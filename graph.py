import tensorflow as tf
import os
from typing import Dict

libtrain = tf.load_op_library(
  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cw2vec_ops.so'))


def make_forward(word_vocab_size,
                 stroke_vocab_size,
                 embedding_dim):
  init_width = 0.5 / embedding_dim

  stroke_embeddings = tf.Variable(
    tf.random_uniform(
      [stroke_vocab_size, embedding_dim], -init_width, init_width),
    name='stroke_embeddings')

  word_embeddings = tf.Variable(
    tf.random_uniform(
      [word_vocab_size, embedding_dim], -init_width, init_width),
    name='word_embeddings')

  tf.summary.histogram(name='summary/stroke_embeddings', values=stroke_embeddings)
  tf.summary.histogram(name='summary/word_embeddings', values=word_embeddings)


def make_train_op(
      stroke_seqs, context_ids, end_indices,
      word_vocab_size, word_counts,
      num_samples, init_lr,
      words_per_batch, words_to_train_est):
  # words_per_batch and words_to_train_est are used to calculate lr decay

  graph = tf.get_default_graph()
  stroke_embeddings = graph.get_tensor_by_name('stroke_embeddings:0')
  word_embeddings = graph.get_tensor_by_name('word_embeddings:0')

  global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
  inc_global_step = tf.assign_add(global_step, 1, name='inc_global_step')

  num_words_processed = tf.Variable(0, trainable=False, dtype=tf.int64, name='num_words_processed')
  inc_num_words_processed = tf.assign_add(num_words_processed, words_per_batch, name='inc_num_words_processed')

  word_counts_aux : Dict[int, int] = dict(word_counts)
  unigrams = [
    word_counts_aux[word_id]
    for word_id in range(word_vocab_size)]

  lr = tf.multiply(
    init_lr,
    tf.maximum(0.0001, 1.0 - tf.cast(tf.cast(num_words_processed, tf.float64) / words_to_train_est, tf.float32)),
    name='lr')

  # ideally, we should increase these two counters after the training op has finished,
  # but we don't live in a perfect world, aren't we?
  with tf.control_dependencies([inc_global_step, inc_num_words_processed]):
    train_op = libtrain.neg_train_cw2vec(
      w_in=stroke_embeddings,
      w_out=word_embeddings,
      examples=stroke_seqs,
      labels=context_ids,
      end_indices=end_indices,
      lr=lr,
      vocab_count=unigrams,
      num_negative_samples=num_samples)

  return train_op


def make_dataset(data_queue, prefetch_size):
  def gen():
    while True:
      data = data_queue.get()
      if data is None:
        break
      else:
        yield data

  dataset = (
    tf.data.Dataset.from_generator(
      generator=gen,
      output_types=(tf.int32, tf.int32, tf.int32))
    .prefetch(buffer_size=prefetch_size))

  iterator = dataset.make_initializable_iterator()

  return iterator.initializer, iterator.get_next()


class Graph:
  def __init__(self,
               data_queue,
               batch_size, num_skips,
               word_vocab_size, stroke_vocab_size,
               embedding_dim, num_samples,
               init_lr,
               word_counts, prefetch_size, words_to_train_est):

    graph = tf.Graph()
    with graph.as_default():
      (dataset_initializer,
       (stroke_seqs, context_ids, end_indices)
       ) = make_dataset(
        data_queue=data_queue,
        prefetch_size=prefetch_size)

      make_forward(
        word_vocab_size=word_vocab_size,
        stroke_vocab_size=stroke_vocab_size,
        embedding_dim=embedding_dim)

      train_op = make_train_op(stroke_seqs=stroke_seqs,
                               context_ids=context_ids,
                               end_indices=end_indices,
                               word_vocab_size=word_vocab_size,
                               word_counts=word_counts,
                               init_lr=init_lr,
                               num_samples=num_samples,
                               words_per_batch=batch_size // num_skips,
                               words_to_train_est=words_to_train_est)

      epoch = tf.Variable(
        initial_value=1, dtype=tf.int32, trainable=False, name='current_epoch')
      advance_epoch = tf.assign_add(
        ref=epoch, value=1, use_locking=True, name='advance_epoch')

      self.graph = graph
      self.train_op = train_op
      self.epoch = epoch
      self.advance_epoch = advance_epoch
      self.dataset_initializer = dataset_initializer
      self.summary = tf.summary.merge_all()