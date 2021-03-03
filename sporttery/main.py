import math

import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# 用来评估的数据
n_test = 14 * 1000
epochs = 5
hid = 64
n_batch = 14
# 限定胜负彩每次投注上限
bet_lim = 20
# 可能最后两期的数据不定，如果只有最后一期不定就为 14，两期就为 28
n_trim = 14
# 多头有利于结果的稳定性，大致上每次跑脚本的结果差不多
heads = 16
# 11（年投入约 1 万）, 12, 13 每加一，净收入 x2，但成本 x5 （14 收入也增加，但不多了）
# 不过现在取消了超过 9 单自动拆单的服务，要一单一单手输，为方便限制为 10
max_n_most9 = 10

matches = sorted([
    '挪超', '葡超', '英超', '德乙', '英冠', '日乙', '日职', '瑞超', '欧冠',
    '解放者杯', '巴西杯', '欧罗巴', '德甲', '西甲', '意甲', '意大利杯',
    '国王杯', '巴甲', '世预赛', '国际赛', '挪威杯', '联合会杯', '欧青赛',
    '亚冠', '瑞典杯', '韩职', '金杯赛', '杯赛', '美职足', '日联赛杯',
    '俱乐部赛', '荷超杯', '法超杯', '德国杯', '澳超', '意超杯', '社区盾杯',
    '葡超杯', '英联赛杯', '西超杯', '欧超杯', '苏超', '荷甲', '法甲',
    '苏联赛杯', '法联赛杯', '荷兰杯', '世青赛', '天皇杯', '亚预赛',
    '英足总杯', '英甲', '葡联赛杯', '葡萄牙杯', '世俱杯', '苏足总杯',
    '法国杯', '非洲杯', '阿甲', '四强赛', '日超杯', '世界杯', '德超杯',
    '法乙', '欧预赛', '荷乙', '亚运男足', '东南亚锦', '亚洲杯', '瑞超杯',
    '公开赛杯', '英锦标赛', '中北美冠', '英乙', '女世界杯', '美洲杯',
    '俱乐部杯', '优胜者杯', '欧青预赛', '欧洲杯', '奥运女足', '奥运男足',
    '女四强赛', '阿根廷杯', '圣保罗锦', '非预赛', '亚运女足', '智利甲',
    '墨超', '俄超', '阿超杯', '俄超杯', '国冠杯', '墨冠杯', '墨超杯',
    '澳杯', '墨西哥杯', '智利杯', '俄罗斯杯', '智超杯', '亚奥赛',
    '韩足总杯', '比超杯', '比甲', '比利时杯', '中国杯', '挪超杯',
    '女欧洲杯', '欧国联',
], key=len, reverse=True)
m_matches = {m: i for i, m in enumerate(['null'] + matches)}


def match(x):
    for m in matches:
        if m in x:
            return m_matches[m]
    n, m = 1, 'null'
    for _m in matches:
        _n = len(set(_m).intersection(x))
        if _n > n:
            n, m = _n, _m
    return m_matches[m]


csv = pandas.read_csv('csv/total.csv')
info = csv[-14:][csv.columns[[0, 4, 5]]]
period = info['period'].unique()
assert len(period) == 1, '期数异常'
period = period[0]
# 检查上一次投注结果
try:
    scores = torch.from_numpy(csv[-28:-14][['h_score', 'g_score']].to_numpy())
    pred = pandas.read_csv('pred/%s.csv' % (period - 1))

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
        print('上期没中奖')
except Exception as e:
    print('Result check failed!')
    print(e)

teams = set(csv['host']).union(csv['guest'])
m_teams = {t: i for i, t in enumerate(teams)}
hs = torch.from_numpy(csv['host'].map(m_teams.get).to_numpy()).long()
gs = torch.from_numpy(csv['host'].map(m_teams.get).to_numpy()).long()
ms = torch.from_numpy(csv['match'].map(match).to_numpy()).long()
odds = torch.from_numpy(csv[csv.columns[8:]].to_numpy()).float()
# 考虑到终赔不一定及时，可以仅使用初赔
# 但加终赔数据效果更好点，所以都在截止之前尽量晚下注
# odds = odds[:, [6 * i + 0 + j for i in range(25) for j in range(3)]]
d_odds = odds.shape[1]
# 将赔率转换为概率，通常会使效果更差
# for i in range(0, d_odds, 3):
#     idx = odds[:, i] != 0.0
#     na = odds[idx]
#     ret = 1 / (1 / na[:, i] + 1 / na[:, i + 1] + 1 / na[:, i + 2])
#     for j in range(3):
#         odds[idx, i + j] = ret / na[:, i + j]
scores = torch.from_numpy(csv[['h_score', 'g_score']].to_numpy()).float()
n_size = (odds.shape[0] // 14) * 14
odds = odds[-n_size:]
scores = scores[-n_size:]
vss = [nn.Sequential(
    nn.LayerNorm(d_odds, elementwise_affine=True),
    nn.Linear(d_odds, hid),
    nn.Dropout(0.5),
    nn.LeakyReLU(),
    nn.Linear(hid, hid),
    nn.Dropout(0.5),
    nn.LeakyReLU(),
    nn.Linear(hid, hid),
    nn.Dropout(0.5),
    nn.LeakyReLU(),
    nn.Linear(hid, 2),
    # nn.Softmax(dim=1),
) for _ in range(heads)]
# scales = [nn.Parameter(4 * torch.ones(1, 2)) for _ in range(heads)]
# gatts = [
#     nn.Parameter(-torch.log(torch.rand(1, d_odds))) for _ in range(heads)]
# matt = nn.Parameter(torch.ones(ms.shape[0], d_odds))
# tatt = nn.Parameter(torch.ones(hs.shape[0], d_odds))
opt = optim.Adam([p for vs in vss for p in vs.parameters()])


def predict(perm, i=None):
    h = odds[perm]
    if i is not None:
        pred = vss[i](h)
    else:
        h = [h for i in range(heads)]
        # h = [h * torch.softmax(gatts[i], dim=1) for i in range(heads)]
        # h = h * torch.softmax(matt[ms[perm]], dim=1)
        # h = h * torch.softmax(tatt[hs[perm]] + tatt[gs[perm]], dim=1)
        pred = torch.cat([
            vss[i](h[i]).unsqueeze(0) for i in range(heads)
        ], dim=0).mean(dim=0)
    real = scores[perm]
    loss = (((pred - real) ** 2).sum(axis=1) ** 0.5).mean()
    # loss = ((pred[:, 0] - pred[:, 1] - real[:, 0] + real[:, 1]) ** 2).mean()
    return pred, loss


for vs in vss:
    vs.train()
for epoch in range(epochs):
    for perm in DataLoader(
            range(n_size - n_test - n_trim),
            batch_size=n_batch,
            shuffle=True):
        opt.zero_grad()
        pred, loss = predict(perm, i=torch.randint(0, heads, (1, )).item())
        loss.backward()
        opt.step()


def pred2bet(pred, cover=0.10):
    # cover = cover / 10
    bet = torch.zeros((pred.shape[0], 3), dtype=bool)
    delta = (pred[:, 0] - pred[:, 1])
    bet[delta.abs() <= cover] = True
    bet[delta.abs() <= 5/3.0 * cover, 1] = True
    bet[torch.arange(pred.shape[0]), 1 + delta.sign().long()] = True
    return bet


covers = [i / 100.0 for i in range(50)]
total_win = [0 for _ in covers]
total_bet = [0 for _ in covers]
total_bet_dyn = 0
total_bet_dyn_run = 0
total_bet_dyn_win1 = 0
total_bet_dyn_win2 = 0
total_bet_lim = [0 for _ in covers]
total_bet_lim_run = [0 for _ in covers]
total_bet_lim_win1 = [0 for _ in covers]
total_bet_lim_win2 = [0 for _ in covers]
total_bet_any9 = [0 for _ in covers]
total_bet_any9_win = [0 for _ in covers]
total_bet_most9 = [0 for _ in covers]
total_bet_most9_win = [0 for _ in covers]
total_bet_most9_run = [0 for _ in covers]
total_bet_most9_run_win = [0 for _ in covers]
total_bet_top9 = [0 for _ in covers]
total_bet_top9_win = [0 for _ in covers]
total_bet_top9_run = [0 for _ in covers]
total_bet_top9_run_win = [0 for _ in covers]
total_loss = 0.0
for perm in DataLoader(
        range(n_size - n_test - n_trim, n_size - n_trim),
        batch_size=14,
        shuffle=False):
    real = scores[perm]
    for epoch in range(epochs):
        if epoch == 0:
            for vs in vss:
                vs.eval()
            with torch.no_grad():
                pred, _ = predict(perm)
            for vs in vss:
                vs.train()
        opt.zero_grad()
        _, loss = predict(perm, i=torch.randint(0, heads, (1, )).item())
        total_loss += loss.item()
        loss.backward()
        opt.step()
    last_bet_prod = 0
    top = ((pred[:, 0] - pred[:, 1]) ** 2).sort(descending=True).indices
    for i, cover in enumerate(covers):
        bet = pred2bet(pred, cover=cover)
        win = bet[
            torch.arange(pred.shape[0]),
            1 + (real[:, 0] - real[:, 1]).sign().long()]
        n_win = win.sum().item()
        bet_prod = bet.sum(axis=1).prod().item()
        if bet_prod > bet_lim >= last_bet_prod:
            dbet = pred2bet(pred, cover=covers[i - 1])
            dwin = dbet[
                torch.arange(pred.shape[0]),
                1 + (real[:, 0] - real[:, 1]).sign().long()]
            n_dwin = dwin.sum().item()
            dbet_prod = dbet.sum(axis=1).prod().item()
            times = bet_lim // dbet_prod
            total_bet_dyn += (dbet_prod * times)
            total_bet_dyn_run += 1
            if n_dwin == 14:
                total_bet_dyn_win1 += (1 * times)
            elif n_dwin == 13:
                total_bet_dyn_win2 += (dbet[~dwin].sum().item() * times)
        last_bet_prod = bet_prod
        total_bet[i] += bet_prod
        any9 = (bet.sum(axis=1) == 1.0)
        n_any9 = any9.sum().item()
        total_bet_any9[i] += math.comb(n_any9, 9)
        total_bet_any9_win[i] += math.comb((any9 & win).sum().item(), 9)
        # 任九第一种思路，选比较确定的前几场（如 11），每场只单投
        if n_any9 >= 9:
            n_most9 = max_n_most9
            if n_most9 > n_any9:
                n_most9 = n_any9
            n_most9_win = win[top[:n_most9]].sum()
            # 比较确定时加倍，收益更高
            n_most9_bet = math.comb(n_most9, 9)
            if False and n_most9 == 10:
                # 前五场固定，后五场浮动并 2 倍投注，收益 x1.5
                # 不过这里仅用于测试，选超参还需要原来的设置
                total_bet_most9[i] += 10
                if n_most9_win == 10:
                    total_bet_most9_win[i] += 10
                elif win[top[:5]].sum() == 5:
                    total_bet_most9_win[i] += 2
            else:
                times = 10 // n_most9_bet
                times = 1
                total_bet_most9[i] += (n_most9_bet * times)
                total_bet_most9_win[i] += (math.comb(n_most9_win, 9) * times)
            total_bet_most9_run[i] += 1
            if n_most9_win >= 9:
                total_bet_most9_run_win[i] += 1
        if bet_prod <= bet_lim:
            # 越确定越加倍，容易中二等。测试结果成本x2，收益x4
            times = bet_lim // bet_prod
            # 但是评估效果时（用于选超参）还是用原来设置
            times = 1
            total_bet_lim[i] += (bet_prod * times)
            total_bet_lim_run[i] += 1
            if n_win == 14:
                total_bet_lim_win1[i] += (1 * times)
            elif n_win == 13:
                total_bet_lim_win2[i] += (bet[~win].sum().item() * times)
        total_win[i] += n_win
        # 任九第二种思路，选最确定的前 9 场复投（不赚钱，选 10 场）
        top9 = top[:10]
        tmp9 = bet[top9].sum(axis=1)
        bet9_prod = 0
        for k in range(10):
            v = tmp9[k]
            tmp9[k] = 1
            bet9_prod += tmp9.prod().item()
            tmp9[k] = v
        if bet9_prod <= bet_lim:
            times = bet_lim // bet9_prod
            total_bet_top9[i] += (bet9_prod * times)
            total_bet_top9_run[i] += 1
            win9 = win[top9]
            n_win9 = win9.sum().item()
            if n_win9 >= 9:
                total_bet_top9_win[i] += (math.comb(n_win9, 9) * times)
                total_bet_top9_run_win[i] += 1
    if bet_prod <= bet_lim:
        times = bet_lim // bet_prod
        total_bet_dyn += (bet_prod * times)
        total_bet_dyn_run += 1
        if n_win == 14:
            total_bet_dyn_win1 += (1 * times)
        elif n_win == 13:
            total_bet_dyn_win2 += (bet[~win].sum().item() * times)
print('loss:', total_loss)
best_cover_14, best_cover_9, best_cover_9t = 0, 0, 0
best_win_14, best_win_9, best_win_9t = 0, 0, 0
for i, cover in enumerate(covers):
    # win14 奖金多注数少，所以不看成本。一等太少，所以近似二等处理
    win_14 = (
        total_bet_lim_win1[i] * 1600 + total_bet_lim_win2[i] * 1000
        - total_bet_lim[i] * 0)
    # win9 中奖率高但下注也多，所以要考虑成本
    win_9 = total_bet_most9_win[i] * 400 - total_bet_most9[i] * 2
    win_9t = total_bet_top9_win[i] * 400 - total_bet_top9[i] * 2
    if win_14 > best_win_14:
        best_win_14 = win_14
        best_cover_14 = cover
    if win_9 > best_win_9:
        best_win_9 = win_9
        best_cover_9 = cover
    if win_9t > best_win_9t:
        best_win_9t = win_9t
        best_cover_9t = cover
    print(
        (
            '%s: %s / %s = %.2f%%, %.5f%%, %.2f, %.2f; '
            'lim: %s, %s, %s, %s; '
            # 'any9: %s, %s, %.2f; '
            'most9: %s, %s, %.2f, %s, %s; '
            'top: %s, %s, %.2f, %s, %s; '
        ) % (
            cover,
            total_win[i], n_test, total_win[i] * 100.0 / n_test,
            100 * (float(total_win[i]) / n_test) ** 14,
            14.0 * total_bet[i] / n_test,
            28.0 * total_bet[i] / n_test / (
                (float(total_win[i]) / n_test) ** 14),
            total_bet_lim[i],
            total_bet_lim_run[i],
            total_bet_lim_win1[i],
            total_bet_lim_win2[i],
            # total_bet_any9[i],
            # total_bet_any9_win[i],
            # total_bet_any9[i] / (total_bet_any9_win[i] + 1e-5),
            total_bet_most9[i],
            total_bet_most9_win[i],
            total_bet_most9[i] / (total_bet_most9_win[i] + 1e-5),
            total_bet_most9_run[i],
            total_bet_most9_run_win[i],
            total_bet_top9[i],
            total_bet_top9_win[i],
            total_bet_top9[i] / (total_bet_top9_win[i] + 1e-5),
            total_bet_top9_run[i],
            total_bet_top9_run_win[i],
        )
    )
print(
    'best_cover_14:', best_cover_14,
    'best_cover_9:', best_cover_9,
    'best_cover_9t:', best_cover_9t,
)

for vs in vss:
    vs.eval()
with torch.no_grad():
    pred, _ = predict(perm)
for i, cover in enumerate(covers):
    bet = pred2bet(pred, cover=cover)
    bet_prod = bet.sum(axis=1).prod().item()
    if bet_prod > bet_lim:
        cover = covers[i - 1]
        break
print('dynamic: %s, %s, %s, %s; cover: %s' % (
    total_bet_dyn,
    total_bet_dyn_run,
    total_bet_dyn_win1,
    total_bet_dyn_win2,
    cover
))
# bet14 = pred2bet(pred, cover=cover)
bet14 = pred2bet(pred, cover=best_cover_14)
bet9 = pred2bet(pred, cover=best_cover_9)
top = ((pred[:, 0] - pred[:, 1]) ** 2).sort(descending=True).indices
bet9t = pred2bet(pred, cover=best_cover_9t)
top9 = top[:9]
val9t = bet9t[top9]
bet9t[:] = 0
bet9t[top9] = val9t
bet9_prod = val9t.sum(axis=1).prod().item()
info[['h_pred', 'g_pred']] = pred
info[['lose', 'draw', 'win']] = bet14.long()
# info[['los9', 'dra9', 'wi9']] = bet9t.long()
val9 = ['-'] * bet9.shape[0]
n_any9 = (bet9.sum(axis=1) == 1).sum().item()
n_most9 = max_n_most9
if n_most9 > n_any9:
    n_most9 = n_any9
if n_any9 >= 9:
    most9 = top[:n_most9]
    for i in most9:
        val9[i] = '013'[bet9[i].sort().indices[-1].item()]
info['any9'] = val9
# ret = info[info.columns[[0, 3, 4, 7, 6, 5, 10, 9, 8, 11, 1, 2]]]
ret = info[info.columns[[0, 3, 4, 7, 6, 5, 8, 1, 2]]]
print(ret)
n_bet14 = bet14.sum(axis=1).prod().item()
print('all: %sx%s; any9: %s; top9: %s;' % (
    n_bet14,
    bet_lim // n_bet14,
    '5x2' if n_most9 == 10 else '1x10',
    # math.comb(n_most9, 9),
    bet9_prod,
))
ret.to_csv('pred/%s.csv' % period, index=False)
