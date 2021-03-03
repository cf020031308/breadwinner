import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# 用来评估的数据，目前约有 30k 数据
# 从 2017 年开始，因为这之后就有人介入了
n_test = 14 * 600
epochs = 1
hid = 32
n_batch = 14
# 多头并不提升准确性，只是用来保证（也不能完全保证）每次跑脚本的结果大致差不多
heads = 16
# 去掉最后一部分数据，因为结果可能还没出来
n_trim = 14

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
# 对赛事与队伍增加注意力，实际效果并不好
# teams = set(csv['host']).union(csv['guest'])
# m_teams = {t: i for i, t in enumerate(teams)}
# hs = torch.from_numpy(csv['host'].map(m_teams.get).to_numpy()).long()
# gs = torch.from_numpy(csv['host'].map(m_teams.get).to_numpy()).long()
# ms = torch.from_numpy(csv['match'].map(match).to_numpy()).long()
odds = torch.from_numpy(csv[csv.columns[8:]].to_numpy()).float()
# 考虑到终赔不一定及时，可以仅使用初赔
# 但加终赔数据效果更好点，所以都在截止之前尽量晚下注
odds = odds[:, [
    6 * i + 0 + j for i in range(odds.shape[1] // 6) for j in range(3)]]
d_odds = odds.shape[1]
# 将赔率转换为概率
# for i in range(0, d_odds, 3):
#     idx = odds[:, i] != 0.0
#     na = odds[idx]
#     ret = 1 / (1 / na[:, i] + 1 / na[:, i + 1] + 1 / na[:, i + 2])
#     for j in range(3):
#         odds[idx, i + j] = ret / na[:, i + j]
# odds = odds - torch.tensor(
#     [0.4519, 0.2474, 0.3008]
# ).unsqueeze(0).repeat(1, odds.shape[1] // 3)
odds = odds - 1
scores = torch.from_numpy(csv[['h_score', 'g_score']].to_numpy()).float()
n_size = (odds.shape[0] // 14) * 14
odds = odds[-n_size:]
scores = scores[-n_size:]
vss = [nn.Sequential(
    # nn.LayerNorm(d_odds, elementwise_affine=True),
    nn.Linear(d_odds, hid),
    nn.LeakyReLU(),
    nn.Linear(hid, hid),
    nn.LeakyReLU(),
    nn.Linear(hid, 3),
    nn.Softmax(dim=1),
) for _ in range(heads)]
# scales = [nn.Parameter(4 * torch.ones(1, 2)) for _ in range(heads)]
# gatts = [nn.Parameter(torch.rand(1, d_odds)) for _ in range(heads)]
# matt = nn.Parameter(torch.ones(ms.shape[0], d_odds))
# tatt = nn.Parameter(torch.ones(hs.shape[0], d_odds))
opt = optim.Adam([p for vs in vss for p in vs.parameters()])


def predict(perm, i=None):
    h = odds[perm]
    if i is not None:
        pred = vss[i](h)
    else:
        h = [h for i in range(heads)]
        # h = [h[i] * torch.softmax(gatts[i], dim=1) for i in range(heads)]
        # h = h * torch.softmax(matt[ms[perm]], dim=1)
        # h = h * torch.softmax(tatt[hs[perm]] + tatt[gs[perm]], dim=1)
        pred = torch.cat([
            vss[i](h[i]).unsqueeze(0) for i in range(heads)
        ], dim=0).mean(dim=0)
    real = scores[perm]
    result = 1 - (real[:, 0] - real[:, 1]).sign().long()
    loss = F.cross_entropy(
        pred, result,
        # 减少胜的预测增加平的预测
        # 使预测结果的分布与实际数据一致，但总体准确性降低
        # 策略中会降低下注频率但相对收益升高
        # TODO: 另外我估计这种方式在计算赔率时收益会比较高
        # weight=1/torch.tensor([0.4519, 0.2474, 0.3008])
    )
    return pred, loss


for vs in vss:
    vs.train()
for perm in DataLoader(
        range(n_size - n_test - n_trim),
        batch_size=n_batch,
        shuffle=False):
    opt.zero_grad()
    _, loss = predict(perm, i=torch.randint(0, heads, (1, )).item())
    loss.backward()
    opt.step()


total_loss = 0.0
total_pred = []
for perm in DataLoader(
        range(n_size - n_test - n_trim, n_size),
        batch_size=n_batch,
        shuffle=False):
    for vs in vss:
        vs.eval()
    with torch.no_grad():
        pred, _ = predict(perm)
        total_pred.append(pred)
    for vs in vss:
        vs.train()
    opt.zero_grad()
    _, loss = predict(perm)
    total_loss += loss.item()
    loss.backward()
    opt.step()
print('mean loss:', total_loss / n_test)
total_pred = torch.cat(total_pred, dim=0)
pandas.DataFrame(total_pred.numpy()).to_csv('pred/results.csv', index=False)

pred = total_pred[:-n_trim]
pred_val, pred_idx = pred.max(dim=1)
real = scores[-n_test - n_trim:-n_trim]
result = 1 - (real[:, 0] - real[:, 1]).sign().long()
print('[real] win: %f, draw: %f, lose: %f' % (
    (result == 0).sum() / n_test,
    (result == 1).sum() / n_test,
    (result == 2).sum() / n_test))
print('[pred] win: %f, draw: %f, lose: %f' % (
    (pred_idx == 0).sum() / n_test,
    (pred_idx == 1).sum() / n_test,
    (pred_idx == 2).sum() / n_test))
print('[pred] win: %f, draw: %f, lose: %f' % (
    pred[:, 0].mean(), pred[:, 1].mean(), pred[:, 2].mean()))
print('[pred-acc] win: %.2f%%, draw: %.2f%%, lose: %.2f%%' % (
    100 * ((pred_idx == 0) & (result == 0)).sum() / (pred_idx == 0).sum(),
    100 * ((pred_idx == 1) & (result == 1)).sum() / (pred_idx == 1).sum(),
    100 * ((pred_idx == 2) & (result == 2)).sum() / (pred_idx == 2).sum(),
))
print('[pred-rec] win: %.2f%%, draw: %.2f%%, lose: %.2f%%' % (
    100 * ((pred_idx == 0) & (result == 0)).sum() / (result == 0).sum(),
    100 * ((pred_idx == 1) & (result == 1)).sum() / (result == 1).sum(),
    100 * ((pred_idx == 2) & (result == 2)).sum() / (result == 2).sum(),
))
print('total acc: %.2f%%' % (100 * (pred_idx == result).sum() / n_test))
for conf in list(range(0, 90, 10)) + list(range(90, 100)):
    flt = pred_val >= (conf / 100.0)
    flt = flt & (pred_val < (conf + 10) / 100.0)
    print('%.2f%% acc: %.2f%% in %d' % (
        conf,
        100 * (pred_idx[flt] == result[flt]).sum() / flt.sum(),
        flt.sum(),
    ))
