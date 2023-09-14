from collections import defaultdict
fn = '/mnt/d/Competition/猛犸杯2023-DNA合成赛道/code-v4.0/scores/0005[1.718922] copy.csv'
score_list = defaultdict(float)
with open(fn, 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines[1:]:
        tokens = line.split(',')
        if len(tokens) > 0:
            score_list[(int(tokens[-2]), int(tokens[-1]))] += float(tokens[-3])

sorted_list = sorted(score_list.items(), key=lambda x: x[1])
print(sorted_list)