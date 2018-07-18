import numpy as np

import collections
import random
import itertools
from tqdm import tqdm

import os
import multiprocessing as mp
import threading

from typing import Dict, List

import logging
logger = logging.getLogger(f'cw2vec.{__name__}')


import stroke


# NOTE: This module does not handle </s>

def report_thread(started_or_finished):
  logger.info(
    f'{threading.current_thread().name} {threading.current_thread().ident} {started_or_finished}')


def report_process(started_or_finished):
  logger.info(
    f'{mp.current_process().name} {mp.current_process().pid} {started_or_finished}')


Corpus = collections.namedtuple(
  typename='Corpus',
  field_names=['word_ids', 'word2id', 'id2word', 'ratio_kept'])


def count_words(words_txt_path):
  # count the number of occurrences of the words,
  # technically this can be avoided - if your memory is large enough
  counter = collections.Counter()
  line_no = None
  with open(words_txt_path, 'r', encoding='utf8') as f:
    for line_no, line in enumerate(f):
      words = line.strip().split()
      counter.update(words)
      print(f'\rlines processed: {line_no+1}', end='')
    print()
  assert line_no is not None

  return counter, line_no+1


def read_words(words_txt_path, drop_if_leq_than):
  full_word_count, total_lines = count_words(words_txt_path)
  word_count = collections.Counter()
  for word, count in full_word_count.items():
    if count <= drop_if_leq_than:
      continue
    else:
      word_count[word] = count
  del full_word_count

  # sorted by frequency, desc
  word2id = {word: ident
             for ident, (word, _) in enumerate(word_count.most_common())}

  id2word = {ident: word for word, ident in word2id.items()}

  # integer representation of the corpus, with low-count words dropped
  word_ids = []
  # number of words (counting duplicates) in the corpus
  total_words = 0

  logger.debug('start to build word list')
  with open(words_txt_path, 'r', encoding='utf8') as f:
    for line in tqdm(f, total=total_lines, desc='lines processed'):
      for word in line.strip().split():
        total_words += 1
        index = word2id.get(word, None)
        if index is None:
          continue
        else:
          word_ids.append(index)
  logger.debug('word list built')

  ratio_kept = len(word_ids)/total_words

  logger.info(' '.join([
    f'#after_drop/#before_drop',
    f'= {len(word_ids)}/{total_words}',
    f'= {ratio_kept:.4f}']))
  logger.info(f'word_vocab_size: {len(word2id)}')

  return Corpus(
    word_ids=word_ids,
    word2id=word2id,
    id2word=id2word,
    ratio_kept=ratio_kept)


def save_word_ids(word_ids, id2word, save_dir):
  report_thread('started')
  logger.info('saving word_ids')
  with open(os.path.join(save_dir, 'word_ids.txt'), 'w', encoding='utf8') as f:
    buffer = []
    for word_id in word_ids:
      buffer.append(str(word_id))
      if len(buffer) >= 10000:
        f.write(' '.join(buffer))
        f.write(' ')
        buffer = []
    if buffer:
      f.write(' '.join(buffer))

  logger.info(f'saving id2word')
  with open(os.path.join(save_dir, 'id2word.txt'), 'w', encoding='utf8') as f:
    for word_id in range(len(id2word)):
      word = id2word[word_id]
      f.write(f'{word_id} {word}\n')

  logger.info(f'saving vocabs')
  with open(os.path.join(save_dir, 'vocabs.txt'), 'w', encoding='utf8') as f:
    lines = '\n'.join(id2word.values())
    f.write(lines)
  report_thread('finished')


def save_word_embeddings(arr, id2word, save_dir, global_step, annotation):
  if annotation is None:
    fname = f'embeddings-{global_step}.txt'
  else:
    fname = f'embeddings-{global_step}-{annotation}.txt'

  logger.info(f'saving word embeddings to {fname}')

  with open(os.path.join(save_dir, fname), 'w', encoding='utf8') as f:
    f.write(f'{len(arr)} {len(arr[0])}\n')
    for idx in range(len(arr)):
      word = id2word[idx]
      vector = ' '.join([str(x) for x in arr[idx, :]])
      line = f'{word} {vector}\n'
      f.write(line)

  logger.info(f'word embeddings saved to {fname}')

def calc_drop_probs(word_ids, threshold):
  counter = collections.Counter(word_ids)
  total = len(word_ids)

  counts = counter.most_common()
  freqs = [(word_id, num_occur / total)
           for (word_id, num_occur) in counts]

  # this drops high-frequency words, if the frequency of a word
  # exceeds the chosen threshold, the higher the frequency,
  # the more likely it gets dropped, which is what we want
  #
  # if the frequency of a word is higher than the threshold,
  # drop_probs[word], will be a number between (0, 1),
  # so this word will be dropped with the calculated probability
  #
  # if the frequency of a word is smaller than the threshold,
  # drop_probs[word] will be a negative number, so:
  #     np.random.rand() < drop_probs[word] < 0
  # will never be true, so less frequent words never gets dropped
  drop_probs = dict(
    [(word_id, 1 - np.sqrt(threshold / freq))
     for word_id, freq in freqs])

  return counts, drop_probs


def subsample(word_ids, drop_probs):
  new_word_ids = [x for x in word_ids
                  if not (np.random.rand() < drop_probs[x])]

  logger.info(' '.join([
    f'#after_subsample/#before_subsample',
    f'= {len(new_word_ids)}/{len(word_ids)}',
    f'= {len(new_word_ids)/len(word_ids):.4f}']))

  return new_word_ids


# %%
def coord(result_queue, num_workers,
          num_epochs,
          batch_size, skip_window, num_skips,
          word_ids, drop_probs,
          word2stroke):
  report_process('started')

  # for mp.Array 'l': signed long, at least 4 bytes
  array_code = 'l'
  assert np.max(word_ids) < 2 ** 31 - 1

  for i_epoch in range(1, num_epochs+1):
    logger.info(f'starting to generate data for epoch {i_epoch}')
    logger.info(f'subsampling for epoch {i_epoch}')
    sampled_word_ids = subsample(word_ids, drop_probs)
    logger.info(f'finished subsampling for epoch {i_epoch}')
    mp_sampled_word_ids = mp.Array(array_code, sampled_word_ids)
    num_words = len(sampled_word_ids)

    gen_queue = mp.JoinableQueue(maxsize=500)
    master_proc = mp.Process(
      target=master,
      name='datagen_master',
      kwargs={
        'gen_queue': gen_queue,
        'num_workers': num_workers,
        'num_words': num_words,
        'batch_size': batch_size,
        'skip_window': skip_window,
        'num_skips': num_skips})

    worker_procs = []
    for i_worker in range(1, num_workers+1):
      worker_proc = mp.Process(
        target=worker,
        name=f'datagen_worker_{i_worker}',
        kwargs={
          'job_queue': gen_queue,
          'result_queue': result_queue,
          'word_ids': mp_sampled_word_ids,
          'skip_window': skip_window,
          'num_skips': num_skips,
          'word2stroke': word2stroke})
      worker_procs.append(worker_proc)

    master_proc.start()
    for worker_proc in worker_procs:
      worker_proc.start()
    logger.info('all data generation processes started')

    logger.info('waiting data generation processes to finish')
    for worker_proc in worker_procs:
      worker_proc.join()
    master_proc.join()

    gen_queue.close()

    # poison
    result_queue.put(None)

    logger.info(f'finished generating data for epoch {i_epoch}')

  report_process('finished')

def master(gen_queue, # :: (begin_center_index_inclusive, length, epoch_done)
           num_workers,
           num_words, batch_size,
           skip_window, num_skips):

  report_process('started')

  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  # left end point of the window, inclusive
  left_index = 0
  epoch_done = False

  max_index = num_words - 1

  while not epoch_done:
    num_centers = batch_size // num_skips

    # when left_index = 1, batch_size = 8, skip_window = 3, num_skips = 2
    # num_centers = batch_size // num_skips = 8 // 2 = 4
    # 0 1 2 3 4 5 6 7 8 9 10 11
    #   _ _ _ c _ _ _
    #     _ _ _ c _ _ _
    #       _ _ _ c _ _ _
    #         _ _ _ c _ _ _
    # 1 + 3 + 4 + 3 - 1 = 10
    right_index_inclusive = left_index + skip_window + num_centers + skip_window - 1
    assert right_index_inclusive >= 0

    # begin_center_index = 4 = 1 + 3 = left_index + skip_window
    # new_left_index = 5 = 1 + 4 = left_index + num_centers

    if right_index_inclusive == max_index:
      epoch_done = True
    elif right_index_inclusive > max_index:
      epoch_done = True
      # backtrack to avoid skipping examples in the end of the data
      # 10 11 12 13 14 15 16 17 18 19 xx xx xx xx xx xx xx
      #                _  _  _  c  _  _  _
      #                   _  _  _  c  _  _  _
      #                      _  _  _  c  _  _  _
      #                         _  _  _  c  _  _  _
      # _  _  _  c  _  _  _
      #    _  _  _  c  _  _  _
      #       _  _  _  c  _  _  _
      #          _  _  _  c  _  _  _
      #
      # when left_index = 15, max_index = 19
      # right_index_inclusive = 15 + 3 + 4 + 3 - 1 = 24
      # new_left_index should equal to 10
      # 15 - (24 - 19)
      left_index = left_index - (right_index_inclusive - max_index)
      right_index_inclusive = left_index + skip_window + num_centers + skip_window - 1
      assert left_index >= 0
      assert right_index_inclusive == max_index
    else:
      pass

    begin_center_index = left_index + skip_window
    gen_queue.put((begin_center_index, num_centers))

    left_index = left_index + num_centers

  # poison
  for _ in range(num_workers):
    gen_queue.put(None)

  gen_queue.join()

  report_process('finished')


def worker(job_queue, result_queue,
           word_ids, skip_window, num_skips,
           word2stroke):
  report_process('started')
  while True:
    job = job_queue.get()
    if job is None:
      job_queue.task_done()
      break
    else:
      (begin_center_index_inclusive, num_centers) = job
      xs, ys = [], []
      for center_index in range(begin_center_index_inclusive, begin_center_index_inclusive+num_centers):
        # when skip_window = 3, center_index = 5
        # 1 2 3 4 5 6 7 8 9
        #   _ _ _ c _ _ _
        left_index = center_index - skip_window
        right_index = center_index + skip_window
        context_ids_aux = word_ids[left_index:center_index] + word_ids[center_index+1:right_index+1]
        context_ids = random.sample(context_ids_aux, num_skips)
        center_ids = itertools.repeat(word_ids[center_index], times=num_skips)
        xs.extend(center_ids)
        ys.extend(context_ids)
      new_xs, ends = expand_word_ids(
        word_ids=xs,
        word2stroke=word2stroke)
      result_queue.put((new_xs, ys, ends))
      job_queue.task_done()

  report_process('finished')


# %%
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
# (6, [1-5, 7-11]) x 2
# (7, [2-6, 8-12]) x 2
# (8, [3-7, 9-13]) x 2
# word_ids = list(range(1, 13+1))
# batch_size = 6
# num_skips = 2
# skip_window = 5
#
# num_epochs = 2
# data_queue = mp.Queue(maxsize=10)
# coord_proc = mp.Process(
#   target=coord,
#   name='datagen_coord',
#   kwargs={
#     'result_queue': data_queue,
#     'num_workers': 3,
#     'num_epochs': num_epochs,
#     'batch_size': batch_size,
#     'skip_window': skip_window,
#     'num_skips': num_skips,
#     'word_ids': word_ids,
#     'drop_probs': None})
# coord_proc.start()
# for _ in range(num_epochs):
#   print('BEGIN')
#   while True:
#     data = data_queue.get()
#     if data is None:
#       break
#     else:
#       print(data)
#   print('END')

# %%
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
# batch 1
# (5, [1-4, 6-9]) x 3
# (6, [2-5, 7-10]) x 3
# batch 2
# (7, [3-6, 8-11]) x 3
# (8, [4-7, 9-12]) x 3
# batch 3
# (9, [5-8, 10-13]) x 3
# (5, [1-4, 6-9]) x 3  -- differ, but also make sense
# word_ids = list(range(1, 13+1))
# batch_size = 6
# num_skips = 3
# skip_window = 4
#
# num_epochs = 2
# data_queue = mp.Queue(maxsize=10)
# coord_proc = mp.Process(
#   target=coord,
#   name='datagen_coord',
#   kwargs={
#     'result_queue': data_queue,
#     'num_workers': 3,
#     'num_epochs': num_epochs,
#     'batch_size': batch_size,
#     'skip_window': skip_window,
#     'num_skips': num_skips,
#     'word_ids': word_ids,
#     'drop_probs': None})
# coord_proc.start()
# for _ in range(num_epochs):
#   print('BEGIN')
#   while True:
#     data = data_queue.get()
#     if data is None:
#       break
#     else:
#       print(data)
#   print('END')
# %%

def expand_word_ids(word_ids, word2stroke: Dict[int, List[int]]):
  nstroke_ids = []
  ends = []
  cursor = None
  # 1 2 3 | 4 5 | 6 7 8
  # ends = [2, 4, 7]
  for word_id in word_ids:
    seqs = word2stroke[word_id]
    nstroke_ids.extend(seqs)
    if cursor is None:
      cursor = len(seqs) - 1
    else:
      # [cursor + len(seqs)] - [cursor + 1] + 1 = len(seqs)
      cursor = cursor + len(seqs)
    ends.append(cursor)
  assert len(ends) == len(word_ids)
  assert ends[-1] == len(nstroke_ids) - 1
  return nstroke_ids, ends


class Data:
  def __init__(self, flags):

    logger.info('loading training corpus')
    corpus = read_words(flags.words_txt_path, flags.drop_if_leq_than)
    self.word_ids = corpus.word_ids
    self.word2id = corpus.word2id
    self.id2word = corpus.id2word
    del corpus

    self.word_vocab_size = len(self.word2id)
    logger.info(f'word_vocab_size={self.word_vocab_size}')

    self.saving_thread = threading.Thread(
      target=save_word_ids,
      kwargs={
        'word_ids': self.word_ids,
        'id2word': self.id2word,
        'save_dir': flags.output_dir},
      name='save_word_ids')

    self.word_counts, self.drop_probs = calc_drop_probs(
      word_ids=self.word_ids,
      threshold=flags.subsampling_threshold)

    logger.info('calculating words_to_train_est')
    self.words_to_train_est = flags.num_epochs * len(subsample(
      word_ids=self.word_ids, drop_probs=self.drop_probs))

    logger.info('loading strokes')
    self.stroke_vocab_size, self.word2stroke = stroke.build_word2stroke(
      id2word=self.id2word,
      strokes_csv_path=flags.strokes_csv_path,
      min_width=3,
      max_width=12)
    logger.info(f'stroke_vocab_size={self.stroke_vocab_size}')

    self.data_queue = mp.Queue(maxsize=flags.data_queue_size)
    self.coord_proc = mp.Process(
      target=coord,
      name='datagen_coord',
      kwargs={
        'result_queue': self.data_queue,
        'num_workers': flags.num_datagens,
        'num_epochs': flags.num_epochs,
        'batch_size': flags.batch_size,
        'skip_window': flags.skip_window,
        'num_skips': flags.num_skips,
        'word_ids': self.word_ids,
        'drop_probs': self.drop_probs,
        'word2stroke': self.word2stroke})