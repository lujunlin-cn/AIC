import json

d = json.load(open('/data/aic/experiments_910a/LFM_V9/arms_analysis.json'))
KEY = ('A0_ctrl_vs_A1_raw', 'A0_ctrl_vs_A1_smooth', 'A1_raw_vs_A1_smooth',
       'A0_ctrl_vs_A3_kdv2')
for pool in ('confirm', 'phd2_val', 'rv_dev', 'rv_rot_dev', 'live_dev'):
    if pool not in d['pools']:
        continue
    e = d['pools'][pool]
    print('== %s  cells=%d sources=%d' % (pool, e['n_cells'], e['n_sources']))
    for nm, v in sorted(e['mean_iou'].items(), key=lambda kv: -kv[1]):
        print('   %-16s %.4f' % (nm, v))
    for k, v in e['pairs'].items():
        if k in KEY:
            fl = 'SIG' if v['significant'] else '   '
            print('   %s %-28s %+.4f [%+.4f,%+.4f] W/L=%d/%d'
                  % (fl, k, v['delta'], v['ci95'][0], v['ci95'][1],
                     v['wins'], v['losses']))
    print()