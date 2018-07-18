import tensorflow as tf
import numpy as np

import os
import sys
from datetime import datetime
import time

import threading
import multiprocessing as mp
import git

import logging
logger = logging.getLogger('cw2vec')

import data
import graph
import util


def report_thread(started_or_finished):
  logger.info(
    f'{threading.current_thread().name} {threading.current_thread().ident} {started_or_finished}')


def training_thread_body(sess, train_op):
  # for one epoch
  report_thread('started')
  while True:
    try:
      sess.run(train_op)
    except tf.errors.OutOfRangeError:
      break
  report_thread('finished')


def progress_thread_body(
      report_interval, ckpt_interval, summary_interval,
      sess, summary_writer, summary, saver, model_save_prefix, output_dir, id2word,
      words_to_train_est, data_queue, num_skips):
  # for one epoch

  report_thread('started')
  epoch = sess.graph.get_tensor_by_name('current_epoch:0')
  num_words_processed = sess.graph.get_tensor_by_name('num_words_processed:0')
  global_step = sess.graph.get_tensor_by_name('global_step:0')
  lr = sess.graph.get_tensor_by_name('lr:0')
  word_embeddings = sess.graph.get_tensor_by_name('word_embeddings:0')
  stroke_embeddings = sess.graph.get_tensor_by_name('stroke_embeddings:0')

  saving_threads = []

  last_save_time = None

  last_words_processed = 0
  last_rate_time = time.monotonic()

  last_summary_time = None

  initial_epoch = sess.run(epoch)
  while True:
    time.sleep(report_interval)
    try:
      now = time.monotonic()

      num_words_processed_val, epoch_val, global_step_val, lr_val, se_val, we_val = sess.run([
        num_words_processed, epoch, global_step, lr, stroke_embeddings, word_embeddings])

      rate = (num_words_processed_val - last_words_processed) / (now - last_rate_time)
      last_words_processed = num_words_processed_val
      last_rate_time = now

      logger.info(' '.join([
        f'epoch={epoch_val}',
        f'progress={num_words_processed_val/words_to_train_est:.4%}(est)',
        f'global_step={global_step_val}',
        f'data_queue_size={data_queue.qsize()}(est)',
        f'lr={lr_val:.6f}',
        f'words_per_sec={rate:.0f}(wo. skips) {rate*num_skips:.0f}(w. skips)',
        f'∥se∥={np.linalg.norm(se_val, ord=2):.4f}',
        f'∥we∥={np.linalg.norm(we_val, ord=2):.4f}']))


      now = time.monotonic()
      if last_save_time is None or now - last_save_time >= ckpt_interval:
        saver.save(sess, model_save_prefix, global_step=global_step_val)
        thread = threading.Thread(
          target=data.save_word_embeddings,
          kwargs={'arr': sess.run(word_embeddings),
                  'id2word': id2word,
                  'save_dir': output_dir,
                  'global_step': global_step_val,
                  'annotation': 'periodic'})
        thread.start()
        saving_threads.append(thread)
        last_save_time = now

      if summary_interval > 0:
        now = time.monotonic()
        if last_summary_time is None or now - last_summary_time >= summary_interval:
          last_summary_time = now
          global_step_val, summary_val = sess.run(
            [global_step, summary])
          summary_writer.add_summary(summary_val, global_step=global_step_val)

      if epoch_val != initial_epoch:
        break
    except tf.errors.OutOfRangeError:
      break

  logger.info(f'waiting for {len(saving_threads)} saving threads to finish')
  for t in saving_threads:
    t.join()
  logger.info(f'all {len(saving_threads)} saving threads finished')

  report_thread('finished')


def train_one_epoch(sess, num_training_threads):
  training_threads = []
  for i_thread in range(1, num_training_threads+1):
    t = threading.Thread(
      target=training_thread_body,
      name=f'training_thread_{i_thread}',
      kwargs={
        'sess': sess,
        'train_op': sess.graph.get_operation_by_name('NegTrainCw2vec')})
    t.start()
    training_threads.append(t)
  logger.info('all training threads started, waiting for them to finish')

  for t in training_threads:
    t.join()
  logger.info('all training threads finished')


def ensure_dirs(flags, run_id):
  flags.save_dir = os.path.join(flags.save_dir, run_id)
  flags.summary_dir = os.path.join(flags.summary_dir, run_id)
  flags.output_dir = os.path.join(flags.output_dir, run_id)
  for dir_path in [flags.save_dir, flags.summary_dir, flags.output_dir]:
    logger.info(f'making dir: {dir_path}')
    os.makedirs(dir_path, exist_ok=True)
  return flags


def sync_save(saver, sess, model_save_prefix, id2word, output_dir, annotation):
  global_step = sess.graph.get_tensor_by_name('global_step:0')
  word_embeddings = sess.graph.get_tensor_by_name('word_embeddings:0')
  global_step_val, word_embeddings_val = sess.run([global_step, word_embeddings])
  saver.save(sess, model_save_prefix, global_step=global_step_val)
  data.save_word_embeddings(
    arr=word_embeddings_val,
    id2word=id2word,
    save_dir=output_dir,
    global_step=global_step_val,
    annotation=annotation)


def train(flags, run_id):
  logger.info(f'run id {run_id} started in PID {mp.current_process().pid}')

  flags = ensure_dirs(flags, run_id)
  util.print_tf_flags(flags)

  repo = git.Repo('.')
  commit = repo.rev_parse('HEAD')
  logger.info(f'using commit {commit}')
  if repo.is_dirty() and not flags.dirty_ok:
    logger.error(f'repository is dirty, refuse to continue, exiting')
    exit(1)

  logger.info('setup data')
  training_data = data.Data(flags=flags)
  if flags.cmd == 'vocab_size':
    return
  training_data.coord_proc.start()
  training_data.saving_thread.start()

  logger.info('making graph')
  training_graph = graph.Graph(
    data_queue=training_data.data_queue,
    batch_size=flags.batch_size,
    num_skips=flags.num_skips,
    word_vocab_size=training_data.word_vocab_size,
    stroke_vocab_size=training_data.stroke_vocab_size,
    embedding_dim=flags.embedding_dim,
    num_samples=flags.num_samples,
    init_lr=flags.init_lr,
    word_counts=training_data.word_counts,
    prefetch_size=flags.prefetch_size,
    words_to_train_est=training_data.words_to_train_est)
  logger.debug('finished making graph')

  logger.info('starting session')
  session_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(graph=training_graph.graph, config=session_config) as sess, \
       tf.summary.FileWriter(logdir=flags.summary_dir, graph=sess.graph) as summary_writer:

    if flags.debug:
      from tensorflow.python import debug as tf_debug
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=int(1e5), keep_checkpoint_every_n_hours=1)
    model_save_prefix = os.path.join(flags.save_dir, 'model-')

    for i_epoch in range(1, flags.num_epochs+1):
      epoch_begin_time = time.monotonic()

      sess.run(training_graph.dataset_initializer)
      progress_thread = threading.Thread(
        target=progress_thread_body,
        name='progress_thread',
        kwargs={
          'report_interval': flags.report_interval,
          'ckpt_interval': flags.ckpt_interval,
          'summary_interval': flags.summary_interval,
          'sess': sess,
          'summary_writer': summary_writer,
          'summary': training_graph.summary,
          'saver': saver,
          'model_save_prefix': model_save_prefix,
          'output_dir': flags.output_dir,
          'id2word': training_data.id2word,
          'words_to_train_est': training_data.words_to_train_est,
          'data_queue': training_data.data_queue,
          'num_skips': flags.num_skips})
      progress_thread.start()

      train_one_epoch(
        sess=sess, num_training_threads=flags.num_training_threads)

      epoch_end_time = time.monotonic()

      sess.run(training_graph.advance_epoch)

      logger.info('waiting for progress thread to finish')
      progress_thread.join()

      logger.info(' '.join([
        f'epoch={i_epoch}',
        f'seconds={int(epoch_end_time - epoch_begin_time)}']))

    # end for epoch in epochs
    # the entire training process is done
    sync_save(
      saver=saver,
      sess=sess,
      model_save_prefix=model_save_prefix,
      id2word=training_data.id2word,
      output_dir=flags.output_dir,
      annotation='final-save')

  training_data.saving_thread.join()
  training_data.coord_proc.join()
  logger.info(f'run id {run_id} is finished')


def main(argv):
  run_id, = argv
  global FLAGS
  if FLAGS.cmd == 'train' or FLAGS.cmd == 'vocab_size':
    train(flags=FLAGS, run_id=run_id)
  else:
    raise Exception()


if __name__ == '__main__':
  # logging
  tf.app.flags.DEFINE_string(
    'log_level', 'debug',
    'log level')

  tf.app.flags.DEFINE_string(
    'log_file', 'large/output/cw2vec',
    'where to write the logs')

  tf.app.flags.DEFINE_bool(
    'debug', False,
    'use debug cli')

  tf.app.flags.DEFINE_bool(
    'dirty_ok', False,
    'run even if the repository dirty')

  # cmd
  tf.app.flags.DEFINE_string(
    'cmd', 'train',
    'what to do')

  # data files
  tf.app.flags.DEFINE_string(
    'words_txt_path', 'large/input/zhwiki_corpus.txt',
    'training corpus, space separated words')

  tf.app.flags.DEFINE_string(
    'strokes_csv_path', 'large/input/stroke.csv',
    'lookup stroke information from this CSV')

  # training params
  tf.app.flags.DEFINE_integer(
    'num_epochs', 8, # word2vec uses 15
    'how many epochs to train')

  tf.app.flags.DEFINE_integer(
    'batch_size', 18, # word2vec uses 16
    'batch size')

  tf.app.flags.DEFINE_float(
    'init_lr', 0.025, # from JWE
    'initial learning rate')

  tf.app.flags.DEFINE_integer(
    'num_samples', 10, # from JWE, cw2vec uses 5. 5-20 if small dataset, 2-5 if large dataset
    'number of negative samples')

  tf.app.flags.DEFINE_integer(
    'embedding_dim', 200,
    'embedding dim, for both strokes and words')

  tf.app.flags.DEFINE_integer(
    'num_skips', 6, # cw2vec didn't mention about this
    'use each word these much times as the center word')

  tf.app.flags.DEFINE_integer(
    'skip_window', 5, # from cw2vec
    'skip window size')

  tf.app.flags.DEFINE_float(
    'subsampling_threshold', 0.0001, # from JWE
    'subsampling threshold')

  tf.app.flags.DEFINE_integer(
    'drop_if_leq_than', 10, # from cw2vec
    'if word occurrence <= this number, drop this word from the vocab')

  # memory usage
  tf.app.flags.DEFINE_integer(
    'data_queue_size', 1000,
    'max size of the data generation queue')

  tf.app.flags.DEFINE_integer(
    'prefetch_size', 1000,
    'prefetch size of tf.data.Dataset')

  tf.app.flags.DEFINE_integer(
    'num_datagens', 1,
    'how many processes to start to generate batches')

  tf.app.flags.DEFINE_integer(
    'num_training_threads', 12,
    'how many threads to use to train the model')

  tf.app.flags.DEFINE_string(
    'save_dir', 'large/output/checkpoint/',
    'save model to this directory')

  tf.app.flags.DEFINE_string(
    'summary_dir', 'large/output/summary/',
    'where to save summary')

  tf.app.flags.DEFINE_string(
    'output_dir', 'large/output/result',
    'where to save the trained embeddings, vocabs, etc.')

  tf.app.flags.DEFINE_integer(
    'ckpt_interval', 30*60,
    'take a checkpoint every these seconds')

  tf.app.flags.DEFINE_integer(
    'report_interval', 5*60,
    'report progress every these seconds')

  tf.app.flags.DEFINE_integer(
    'summary_interval', 30*60,
    'write TensorBoard summary every these seconds')

  FLAGS = tf.app.flags.FLAGS


  formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(funcName)s] [%(levelname)s] %(message)s')

  numeric_log_level = getattr(logging, FLAGS.log_level.upper(), None)
  if not isinstance(numeric_log_level, int):
    raise ValueError(f'Invalid log level: {numeric_log_level}')

  run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S_UTC')

  FLAGS.log_file = f'{FLAGS.log_file}.{run_id}.log'
  os.makedirs(os.path.dirname(FLAGS.log_file), exist_ok=True)
  open(FLAGS.log_file, 'w').close()

  fileHandler = logging.FileHandler(filename=FLAGS.log_file, encoding='utf8', mode='a')
  fileHandler.setLevel(numeric_log_level)
  fileHandler.setFormatter(formatter)
  logger.addHandler(fileHandler)

  stdoutHandler = logging.StreamHandler(stream=sys.stdout)
  stdoutHandler.setLevel(numeric_log_level)
  stdoutHandler.setFormatter(formatter)
  logger.addHandler(stdoutHandler)

  logger.setLevel(numeric_log_level)

  tf.app.run(argv=[run_id])