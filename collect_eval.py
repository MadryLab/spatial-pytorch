DATA = '/data/theory/robustopt/engstrom/store/spatial_eval'
import itertools
from cox.readers import CollectionReader
from cox.store import Store

r = CollectionReader(DATA)

metadata_df = r.df('metadata')
res_df = r.df('results')

def get(orig, attack):
    orig_model = metadata_df['resume'].apply(lambda x:x[46:-19])

    col = 'advacc'
    keys = ['tries', 'use_best', 'attack_type']

    if attack == 'standard':
        values = [1, 0, 'random']
        col = 'natacc'

    elif attack == 'random':
        values = [1, 0, 'random']

    elif attack == 'worst10':
        values = [10, 1, 'random']

    elif attack == 'grid':
        values = [1, 1, 'grid']

    should_keep = orig_model == orig
    for k, v in zip(keys, values):
        should_keep = should_keep & (metadata_df[k] == v)

    exp_ids = metadata_df[should_keep]['exp_id'].tolist()
    assert len(exp_ids) == 1, str(exp_ids)
    eid = exp_ids[0]

    out = res_df[res_df['exp_id'] == eid][col].tolist()[0]
    return out

# poss_origs = itertools.product(['30', '40'], ['worst10', 'random'])
# origs = [y + '_' + x for x, y in poss_origs] + ['nocrop_30', 'standard_30']

origs = ['standard_40', 'nocrop_40', 'random_30', 'random_40', 'worst10_30',
         'worst10_40']

# attacks = ['standard', 'random', 'worst10', 'grid']

attacks = ['standard', 'random', 'worst10', 'grid']
row_names = [
    'Standardly Trained Model',
    'No Crop',
    'Data Aug. (30 deg/24px)',
    'Data Aug. (40 deg/32px)',
    'Worst-of-10 (30 deg/24px)',
    'Worst-of-10 (40 deg/32px)'
]

col_labels = [
    '',
    'Natural',
    'Random (30 deg/24px)',
    'Worst-of-10 Random',
    'Exhaustive Search'
]

lines = [col_labels]
for orig, row_name in zip(origs, row_names):
    values = [row_name]
    for attack in attacks:
        val = get(orig, attack)
        values.append('%.1f%%' % val)

    lines.append(values)

from collections import defaultdict
col_widths = defaultdict(lambda : 2)

for line in lines:
    for i, row_item in enumerate(line):
        col_widths[i] = max(col_widths[i], len(row_item) + 2)

padded_lines = []
for line in lines:
    padded_line = []
    for i, row_item in enumerate(line):
        width = col_widths[i]
        padded_value = ('{: ^%s}' % width).format(row_item)
        padded_line.append(padded_value)

    padded_lines.append(padded_line)

barrier_lines = []
for i in range(len(col_widths.keys())):
    width = col_widths[i]
    if i == 0:
        barrier = '-' * width
    else:
        barrier = ':' + ('-' * (width - 2)) + ':'

    barrier_lines.append(barrier)

barriered_lines = [padded_lines[0]] + [barrier_lines] + padded_lines[1:]
final_lines = ['|' + '|'.join(b) + '|' for b in barriered_lines]

final_string = '\n'.join(final_lines)

print(final_string)
