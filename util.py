import tabulate

import logging
logger = logging.getLogger(f'cw2vec.{__name__}')


def unzip(xys):
  # be a good lad and be explicit and do not use argument unpacking
  xs, ys = [], []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return xs, ys


def unzip3(ts):
  xs, ys, zs = [], [], []
  for x, y, z in ts:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return xs, ys, zs


def print_tf_flags(flags):
  rows = flags.flag_values_dict().items()
  table = tabulate.tabulate(rows, headers=['flag name', 'value'], tablefmt='grid')
  logger.info(f'flags:\n{table}')