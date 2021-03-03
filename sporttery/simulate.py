# 学校附近的投注站都比较远，去一次麻烦，所以设计方案时注意参与度不宜太高

import pandas
import torch


csv = pandas.read_csv('csv/total.csv')
info = csv[-14:]
csv = csv[:-14]
period = info['period'].unique()
assert len(period) == 1, '期数异常'
period = period[0]

award = pandas.read_csv('csv/award.csv')
cols = ['period',
        'count1', 'award1', 'total1',
        'count2', 'award2', 'total2',
        'count9', 'award9', 'total9']
for col in cols:
    award[col] = award[col].map(lambda x: int(str(x).replace(',', '')))
award = torch.from_numpy(award[cols].to_numpy())

pred = torch.from_numpy(pandas.read_csv('pred/results.csv').to_numpy()).float()
new_pred = pred[-14:]
pred = pred[:-14]
scores = torch.from_numpy(
    csv[['h_score', 'g_score']].to_numpy()).float()[-pred.shape[0]:]
results = 1 - (scores[:, 0] - scores[:, 1]).sign().long()
csv = csv[-pred.shape[0]:]
periods = torch.from_numpy(csv['period'].to_numpy())


def pred2probs(pred, prob=0.67, bet_lim=20):
    # 按概率大小依次扩充赛果，直至达到目标概率或限注
    pred = pred.clone()
    n = pred.shape[0]
    probs = torch.zeros((n, 3))
    val, idx = pred.max(dim=1)
    probs[torch.arange(n), idx] = val
    pred[torch.arange(n), idx] = -1
    pred = pred.flatten()
    # 动态规划实现麻烦，边际收益估计不高，先用个近似的方案
    # TODO: 增加概率的效益降低时就停止扩充
    # p = probs.sum(dim=1).prod()
    # b = probs.sign().sum(dim=1).prod()
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


# 组合测试
# 结论：1 或 2 任一可投时就去投注，同时顺便投 3
test = [1, 1, 1]
# 总下注
total_n_bet = 0
# 投注期数
total_bet_day = [0 for _ in range(pred.shape[0] // 14)]
# 一等奖数
total_win1_bet = 0
# 二等奖数
total_win2_bet = 0
# 任九中奖数
total_win_bet = 0
# 中奖期数
total_win_day = [0 for _ in range(pred.shape[0] // 14)]
# 中奖金额
total_real_gain = 0

# 胜负彩方案：不断扩充进概率最高的组合，直到满足概率或达到限制注数
# for prob in [i * 0.05 for i in range(1, 21)]:
prob = 0.35
# 限注对相对收益影响不大，绝对收益呈正比，参与度上升
# for bet_lim in [6, 12, 24, 50, 100, 250, 500]:
bet_lim = 12
# 增加阈值可降低参与度，减少资金量，对绝对收益影响不大
# for thres_rate in [i / 10 for i in range(10)]:
thres_rate = 0.5
if test[0]:
    n_bet, bet_day, win1_bet, win2_bet, win_day, real_gain = 0, 0, 0, 0, 0, 0
    for i in range(0, pred.shape[0], 14):
        probs = pred2probs(pred[i:i + 14], prob, bet_lim)
        p = probs.sum(dim=1).prod()
        if p < prob * thres_rate:
            continue
        bet_day += 1
        total_bet_day[i // 14] = 1
        bet = probs.sign().bool()
        count = bet.sum(dim=1).prod().item()
        # 对确定的投注增加倍数，经测试可提升收益
        times = bet_lim // count
        ret = results[i:i + 14]
        win = bet[torch.arange(14), ret]
        n_win = win.sum().item()
        n_bet += (count * times)
        if n_win == 14:
            win1_bet += times
            win_day += 1
            total_win_day[i // 14] = 1
            gain = award[award[:, 0] == periods[i]][0, 2].item()
            if gain == 0:
                gain = 50000 / times
            real_gain += gain * times
            print(periods[i], 'win1: ', gain, times)
        elif n_win == 13:
            w2_bet = bet[~win].sum().item() * times
            win2_bet += w2_bet
            win_day += 1
            total_win_day[i // 14] = 1
            gain = award[award[:, 0] == periods[i]][0, 5].item()
            real_gain += gain * w2_bet
            print(periods[i], 'win2: ', gain, w2_bet)
    # 保守估计相对收益
    rel_gain = (10 * win1_bet + win2_bet) * 1200 / (n_bet * 2 + 1e-5)
    # 保守估计绝对收益
    abs_gain = (10 * win1_bet + win2_bet) * 1200 - n_bet * 2
    # 按真实数据估计收益
    real_gain -= n_bet * 2
    print(
        (
            '%.2f, %.2f, %d;'
            ' bet: %d, win1: %d, win2: %d;'
            ' bday: %d, wday: %d;'
            ' rel: %.2f, abs: %.2f, real: %d'
        ) % (
            prob, thres_rate, bet_lim,
            n_bet, win1_bet, win2_bet,
            bet_day, win_day,
            rel_gain, abs_gain, real_gain,
        )
    )
    total_n_bet += n_bet
    total_win1_bet += win1_bet
    total_win2_bet += win2_bet
    total_real_gain += real_gain

# 任选九方案1：计算概率最高的 k 个组合（k=10 最好，但为了方便投注，k=5）
comb = torch.combinations(torch.arange(14), 5)
# for prob in [i * 0.05 for i in range(1, 21)]:
prob = 0.55
# for k in list(range(10)) + [10, 15, 20, 50, 100]:
k = 5
if test[1]:
    n_bet, bet_day, win_bet, win_day, real_gain = 0, 0, 0, 0, 0
    for i in range(0, pred.shape[0], 14):
        pr = pred[i:i + 14]
        val, idx = pr.max(dim=1)
        all_p = val.prod()
        rest = val[comb].prod(dim=1).topk(k, largest=False)
        p = (all_p / rest.values).mean().item()
        if p < prob:
            continue
        bet_day += 1
        total_bet_day[i // 14] = 1
        n_bet += k
        rest = comb[rest.indices]
        bet = idx.view(1, -1).repeat(k, 1)
        bet[torch.arange(k).view(-1, 1), rest] = -1
        ret = results[i:i + 14]
        n_win = ((bet == ret.view(1, -1)).sum(dim=1) == 9).sum().item()
        if n_win:
            win_bet += n_win
            win_day += 1
            total_win_day[i // 14] = 1
            gain = award[award[:, 0] == periods[i]][0, 8].item()
            real_gain += gain * n_win
            print(periods[i], 'win9: ', gain, n_win)
    rel_gain = win_bet * 300 / (n_bet * 2 + 1e-5)
    abs_gain = win_bet * 300 - n_bet * 2
    real_gain -= n_bet * 2
    print(
        (
            '%.2f, %d;'
            ' bet: %d, win: %d;'
            ' bday: %d, wday: %d;'
            ' rel: %.2f, abs: %.2f, real: %d'
        ) % (
            prob, k,
            n_bet, win_bet,
            bet_day, win_day,
            rel_gain, abs_gain, real_gain,
        )
    )
    total_n_bet += n_bet
    total_win_bet += win_bet
    total_real_gain += real_gain

# 任选九方案2：将概率按 std 排序，最高的九场比赛按胜负彩方案复式投注
# 这方案参与度比较高调不下来，可配置其它方案，其它方案要下注时就顺便投点
# 三个参数的影响都不大
# for prob in [i * 0.05 for i in range(1, 21)]:
prob = 0.95
# for bet_lim in [3, 6, 12, 24, 50, 100, 250, 500]:
bet_lim = 12
# for thres_rate in [i / 10 for i in range(10)]:
thres_rate = 0.9
if test[2]:
    n_bet, bet_day, win_bet, win_day, real_gain = 0, 0, 0, 0, 0
    for i in range(0, pred.shape[0], 14):
        top9 = pred[i:i + 14].std(dim=1).sort(descending=True).indices[:9]
        probs = pred2probs(pred[i:i + 14][top9], prob, bet_lim)
        p = probs.sum(dim=1).prod()
        if p < prob * thres_rate:
            continue
        bet_day += 1
        total_bet_day[i // 14] = 1
        bet = probs.sign().bool()
        count = bet.sum(dim=1).prod().item()
        times = bet_lim // count
        ret = results[i:i + 14][top9]
        win = bet[torch.arange(9), ret]
        n_win = win.sum().item()
        n_bet += (count * times)
        if n_win == 9:
            win_bet += times
            win_day += 1
            total_win_day[i // 14] = 1
            gain = award[award[:, 0] == periods[i]][0, 8].item()
            real_gain += gain * times
            print(periods[i], 'win9: ', gain, times)
    rel_gain = win_bet * 300 / (n_bet * 2 + 1e-5)
    abs_gain = win_bet * 300 - n_bet * 2
    real_gain -= n_bet * 2
    print(
        (
            '%.2f, %.2f, %d;'
            ' bet: %d, win: %d;'
            ' bday: %d, wday: %d;'
            ' rel: %.2f, abs: %.2f, real: %d'
        ) % (
            prob, thres_rate, bet_lim,
            n_bet, win_bet,
            bet_day, win_day,
            rel_gain, abs_gain, real_gain,
        )
    )
    total_n_bet += n_bet
    total_win_bet += win_bet
    total_real_gain += real_gain

if sum(test) > 1:
    print(
        (
            '%s:'
            ' bet: %d; win1: %d, win2: %d, win9: %d;'
            ' day: %d:%d;'
            ' gain: %d'
        ) % (
            test,
            total_n_bet,
            total_win1_bet,
            total_win2_bet,
            total_win_bet,
            sum(total_bet_day),
            sum(total_win_day),
            total_real_gain,
        )
    )
