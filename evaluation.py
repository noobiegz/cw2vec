import argparse
import tempfile
import sys
from gensim.models import KeyedVectors


def main(args):
  if args.fubar:
    assert args.dim is not None
    assert not args.binary
    with tempfile.NamedTemporaryFile(mode='wb') as tmp:
      print('counting lines...', end='')
      sys.stdout.flush()
      lines = 0
      with open(args.emb, mode='rb') as f:
        for _ in f:
          lines += 1
      print(f'{lines} lines')
      tmp.write(f'{lines} {args.dim}\n'.encode('utf8'))

      print('rewriting file...')
      with open(args.emb, mode='rb') as f:
        while True:
          bs = f.read(1024 * 1024 * 50)
          if bs:
            tmp.write(bs)
          else:
            break
      print('loading embeddings...')
      emb = KeyedVectors.load_word2vec_format(tmp.name, binary=False)
  else:
    print('loading embeddings...')
    emb = KeyedVectors.load_word2vec_format(args.emb, binary=args.binary)

  if args.word_pair:
    print('======= result for word pair =======')
    x = emb.evaluate_word_pairs(args.word_pair)
    print(x)
    print('====================================')

  if args.analogy:
    print('======= result for analogy =======')
    acc = emb.accuracy(args.analogy)
    for section in acc:
      num_correct = len(section['correct'])
      num_incorrect = len(section['incorrect'])
      print(f'{section["section"]}: {num_correct/(num_correct + num_incorrect):.4f}')
    print('==================================')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--emb', type=str)
  parser.add_argument('--binary', action='store_true')
  parser.add_argument('--word_pair', type=str, default='large/input/297.txt')
  parser.add_argument('--analogy', type=str, default='large/input/analogy.txt')
  parser.add_argument('--fubar', action='store_true')
  parser.add_argument('--dim', type=int)

  args = parser.parse_args()

  main(args)
