import sys

import pandas
import torch


csv = pandas.read_csv('csv/total.csv')
if len(sys.argv) == 1:
    period = csv[-28:14]['period'].unique()
    assert len(period) == 1, '期数异常'
    period = period[0]
else:
    period = int(sys.argv[1])

scores = torch.from_numpy(
    csv[csv['period'] == period][['h_score', 'g_score']].to_numpy())
pred = pandas.read_csv('pred/%s.csv' % period)

bet14 = torch.from_numpy(pred[['lose', 'draw', 'win']].to_numpy())
win14 = (scores[:, 0] == -1) | (bet14[
    torch.arange(14),
    1 + (scores[:, 0] - scores[:, 1]).sign().long()
] == 1)
n_win14 = win14.sum().item()
if n_win14 == 14:
    print('胜负彩一等奖x1')
elif n_win14 == 13:
    print('胜负彩二等奖x%d' % bet14[~win14].sum().item())

any9 = pred['any9']
win9 = (pred['any9'] == [
    '013'[i + 1] for i in (scores[:, 0] - scores[:, 1]).sign().long()])
n_win9 = win9.sum()
if n_win9 >= 9:
    print('任选九可能中奖x%d' % n_win9)

if n_win9 < 9 and n_win14 < 13:
    print('%s 期没中奖' % period)
