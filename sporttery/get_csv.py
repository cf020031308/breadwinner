# period.csv
# id, date, match name, host name, guest name, host score, guest score
# period-id.csv
# company, win0, draw0, lose0, win1, draw1, lose1

import sys
import glob

from bs4 import BeautifulSoup


if len(sys.argv) == 1:
    patterns = ['*']
else:
    patterns = [('%s-*' % period) for period in sys.argv[1:]]

results = None
for pattern in patterns:
    last_period = -1
    for fp in sorted(glob.glob('html/%s.html' % pattern)):
        try:
            period, fid = fp.split('/')[-1].split('.')[0].split('-')
            with open(fp) as file:
                content = file.read()
            sp = BeautifulSoup(content)
            host, match, guest = [
                x.text.strip() for x in sp.select('.hd_name')]
            date = sp.select_one(
                '.game_time').text.strip().replace('比赛时间', '').split()[0]
            try:
                h_score, g_score = sp.select_one(
                    '.odds_hd_bf').text.strip().split(':')
            except Exception as e:
                print('[%s-%s scores] %s' % (period, fid, e))
                h_score, g_score = '-1', '-1'
            fn = 'csv/%s.csv' % period
            if period != last_period:
                if results is not None:
                    print('[%s] OK' % last_period)
                    results.close()
                results = open(fn, 'w')
                results.write('id,date,match,host,guest,h_score,g_score\n')
                last_period = period
            results.write(','.join([
                fid, date, match, host, guest, h_score, g_score
            ]) + '\n')

            companies = [x.text.strip() for x in sp.select('.quancheng')]
            odds = [
                x.next_sibling.next_sibling.text.split()
                for x in sp.select('td.tb_plgs')]
            with open('csv/%s-%s.csv' % (period, fid), 'w') as file:
                file.write('company,win0,draw0,lose0,win1,draw1,lose1\n')
                file.write('\n'.join([
                    ','.join([company, win0, draw0, lose0, win1, draw1, lose1])
                    for company, (win0, draw0, lose0, win1, draw1, lose1)
                    in zip(companies, odds)
                ]))
            print('[%s-%s] OK' % (period, fid))
        except Exception as e:
            print('[%s] %s' % (fp, e))
if results is not None:
    print('[%s] OK' % last_period)
    results.close()
