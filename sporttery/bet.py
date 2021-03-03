# 学校附近的投注站都比较远，去一次麻烦，所以设计方案时注意参与度不宜太高

import glob

import pandas
import torch
from bs4 import BeautifulSoup


info = pandas.read_csv('csv/total.csv')[-14:]
period = info['period'].unique()
assert len(period) == 1, '期数异常'
period = period[0]

pred = torch.from_numpy(
    pandas.read_csv('pred/results.csv').to_numpy()).float()[-14:]


def pred2probs(pred, prob=0.67, bet_lim=20):
    pred = pred.clone()
    n = pred.shape[0]
    probs = torch.zeros((n, 3))
    val, idx = pred.max(dim=1)
    probs[torch.arange(n), idx] = val
    pred[torch.arange(n), idx] = -1
    pred = pred.flatten()
    for i in pred.sort(descending=True).indices[:-n]:
        x = i // 3
        y = i % 3
        probs[x, y] = pred[i]
        if probs.sign().sum(dim=1).prod() > bet_lim:
            probs[x, y] = 0
            break
        if probs.sum(dim=1).prod() > prob:
            break
    return probs


print('第 %s 期对阵：' % period)
with open(glob.glob('html/%s-*.html' % period)[0]) as file:
    web = BeautifulSoup(file.read())
table = web.find(text='第%s期' % period).parent.parent.next_sibling.next_sibling
index = list(info['id'])
order = torch.zeros(14, dtype=int)
for i, a in enumerate(table.select('td a')):
    order[i] = index.index(int(a['href'].split('-')[-1].split('.')[0]))
    assert order[i] >= 0
    host, guest = a.text.split('VS' if ('VS' in a.text) else ':')
    print('  %2d. %s\tVS\t%s' % (i, host, guest))
pred = pred[order]


print('\n第 %s 期方案：' % period)
# 胜负彩方案
prob = 0.35
bet_lim = 12
thres_rate = 0.618
probs = pred2probs(pred, prob, bet_lim)
p = probs.sum(dim=1).prod()
if p < prob * thres_rate:
    print('  1. 胜负彩复投[-]')
else:
    bet = probs.sign().bool()
    count = bet.sum(dim=1).prod().item()
    times = bet_lim // count
    print('  1. 胜负彩复投[%.1f%%] %dx%d' % (100 * p, count, times))
    for i, c in enumerate('310'):
        print('     ' + '|'.join(b and c or '-' for b in bet[:, i]))

# 任选九方案1
comb = torch.combinations(torch.arange(14), 5)
prob = 0.55
k = 5
val, idx = pred.max(dim=1)
ps = val.prod()
rest = val[comb].prod(dim=1).topk(k, largest=False)
p = (ps / rest.values).mean().item()
if p < prob:
    print('  2. 任选九单投[-]')
else:
    print('  2. 任选九单投[%.1f%%] 5x2' % (100 * p))
    rest = comb[rest.indices]
    bet = idx.view(1, -1).repeat(k, 1)
    bet[torch.arange(k).view(-1, 1), rest] = 3
    line = []
    for bs in bet:
        line.append('     ' + '|'.join('310-'[b] for b in bs))
    print('\n'.join(sorted(line, reverse=True)))

# 任选九方案2
prob = 0.95
bet_lim = 12
thres_rate = 0.9
top9 = pred.std(dim=1).sort(descending=True).indices[:9]
probs = pred2probs(pred[top9], prob, bet_lim)
p = probs.sum(dim=1).prod()
if p < prob * thres_rate:
    print('  3. 任选九复投[-]')
else:
    bet = probs.sign().bool()
    count = bet.sum(dim=1).prod().item()
    times = bet_lim // count
    print('  3. 任选九复投[%.1f%%] %dx%d' % (100 * p, count, times))
    bet14 = torch.zeros((14, 3), dtype=bool)
    bet14[top9] = bet
    for i, c in enumerate('310'):
        print('     ' + '|'.join(b and c or '-' for b in bet14[:, i]))
