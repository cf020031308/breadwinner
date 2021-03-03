import glob


# 500.com 上的 25 家“主流公司”
mains = [
    '竞彩官方', '威廉希尔', '澳门', '立博',
    'Bet365', 'Interwetten', 'SNAI', '皇冠',
    '易胜博', '伟德', 'Oddset', 'Bwin',
    'Gamebookers', 'Pinnacle平博', '10BET', 'Coral',
    '利记', 'Unibet (优胜客)', 'SportingBet (博天堂)', 'IBCBET (沙巴)',
    'Mansion88 (明升)', '金宝博', '香港马会', '12BET (壹貳博)',
    'Eurobet',
]

# 根据统计，大多数比赛（>20k）都开盘的 16 家公司
# 并不提升模型准确率
real_mains = [
    'Oddset', '利记', '香港马会', '金宝博',
    'Eurobet', '10BET', 'SNAI', 'Coral',
    '皇冠', '易胜博', '澳门', '伟德',
    'Interwetten', '立博', 'Bwin', '威廉希尔',
]
stat = {company: 0 for company in mains}

fps = sorted([
    fp for fp in glob.glob('csv/*.csv')
    if '-' not in fp and 'total' not in fp])
with open('csv/total.csv', 'w') as total:
    total.write('period,id,date,match,host,guest,h_score,g_score,%s\n' % (
        ','.join(
            '%s-%s' % (company, odd)
            for company in real_mains
            for odd in ('win0', 'draw0', 'lose0', 'win1', 'draw1', 'lose1'))))
    for fp in fps:
        with open(fp) as file:
            rets = file.read().splitlines()
        period = fp.split('/')[-1].split('.')[0]
        for line in rets[1:]:
            try:
                _id, *_ = line.split(',')
                with open('csv/%s-%s.csv' % (period, _id)) as match:
                    bets = match.read().splitlines()
                rest = ['1,1,1,1,1,1'] * len(real_mains)
                for bet in bets[1:]:
                    company, odds = bet.split(',', 1)
                    if company not in mains:
                        continue
                    stat[company] += 1
                    if company not in real_mains:
                        continue
                    rest[real_mains.index(company)] = odds
                total.write('%s,%s,%s\n' % (period, line, ','.join(rest)))
            except Exception as e:
                print(line, e)
for company, count in sorted(stat.items(), key=lambda x: x[1]):
    print('%s\t%d' % (company, count))
