import itertools
import collections
from tqdm import tqdm

import csv

from typing import List, Dict, Tuple

import logging
logger = logging.getLogger(f'cw2vec.{__name__}')

import util

def build_char2stroke(csv_path):
  with open(csv_path, newline='', encoding='utf8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(reader)
    char_index = header.index('汉字')
    stroke_index = header.index('笔顺')

    result = {}
    for row in reader:
      result[row[char_index]] = row[stroke_index]

  return result


def word_to_stroke_grams(
      word: str, char2stroke,
      min_width, max_width,
      padding_id_str) -> List[str]:

  # imposing sliding windows with stride 1 on strokes of the given word.
  # this is the S(w) in the original paper.

  # a string, each char is a stroke id
  strokes = ''.join([char2stroke[c] for c in word])
  if len(strokes) < min_width:
    # what if the word has fewer strokes than the minimal window width?
    # - the paper didn't say,
    # so we make an executive decision here and pad it to have the minimal length,
    # luckily there are not too many of these, (wikipedia has about 60 of these),
    # so we might as well drop them
    pad = itertools.repeat(padding_id_str, times=min_width - len(strokes))
    strokes += ''.join(pad)
    assert len(strokes) == min_width

  grams = []
  num_strokes = len(strokes)

  for width in range(min_width, min(num_strokes, max_width)+1):
    # a b c d e     num_strokes=5, width=3
    # abc bcd cde   start_index=[0, 1, 2], 2 = 5 - 3, so range(3)
    for start_index in range(num_strokes-width+1):
      piece = strokes[start_index:(start_index+width)]
      grams.append(piece)

  return grams


# to get a sense of the magnitude:
# number of words: 3158224
# number of stroke grams: 9875240
def collect_all_stroke_grams(
      words: List[str],
      char2stroke,
      min_width, max_width,
      padding_id_str) -> Tuple[Dict[str, int], Dict[int, str]]:

  counter = collections.Counter()
  for word in tqdm(words, desc='collect n-grams'):
    grams = word_to_stroke_grams(
      word=word,
      char2stroke=char2stroke,
      min_width=min_width,
      max_width=max_width,
      padding_id_str=padding_id_str)
    counter.update(grams)

  grams2id = {grams: i for i, (grams, _) in enumerate(counter.most_common())}
  id2grams = {i: grams for grams, i in grams2id.items()}

  return grams2id, id2grams


def build_word2stroke(
      id2word, strokes_csv_path,
      min_width, max_width):

  word_ids, words = util.unzip(id2word.items())

  padding_id_str = '0'
  char2stroke = build_char2stroke(strokes_csv_path)
  gram2id, id2gram = collect_all_stroke_grams(
    words=words,
    char2stroke=char2stroke,
    min_width=min_width,
    max_width=max_width,
    padding_id_str=padding_id_str)

  word_id2stroke_ngrams_ids = {
    word_id: [
      gram2id[gram]
      for gram in word_to_stroke_grams(
        word=id2word[word_id],
        char2stroke=char2stroke,
        min_width=min_width,
        max_width=max_width,
        padding_id_str=padding_id_str)]
    for word_id in tqdm(word_ids, desc='word to stroke')}

  logger.info(f'stroke_vocab_size: {len(gram2id)}')
  return len(gram2id), word_id2stroke_ngrams_ids

# %%
# stroke_csv_path = 'large/dataset/stroke.csv'
# char2stroke = build_char2stroke(stroke_csv_path)
# print(word_to_stroke_grams('大人', char2stroke, min_width=3, max_width=12, padding_id_str='0'))
# print(word_to_stroke_grams('人', char2stroke, min_width=3, max_width=12, padding_id_str='0'))