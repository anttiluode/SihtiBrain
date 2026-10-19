"""POST-RESULT diagnostic: spectral-vs-random code sensitivity to random seed.

This was designed AFTER the primary Gate 3 first run showed a narrow 5.28pp
advantage over one frozen random code seed. It is NOT the preregistered gate.
It reuses identical source graph, worlds and update rules; random codes only
change their independently generated initialization.
"""
import argparse,json
from pathlib import Path
import numpy as np
from learned_relations import learned_sources,spectral_codes,random_codes,run_target

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/learned_gate3'))
    p.add_argument('--random-seeds',type=int,default=16)
    a=p.parse_args()
    table,agree=learned_sources()
    if not agree:raise RuntimeError('structure learner diverged')
    spectral,_=spectral_codes(table)
    worlds=tuple(range(1000,1020))
    spectral_results={}
    for n in (18,30):
        vals=[run_target(w,n,table,spectral,random_codes(42))['new_route']
              ['learned_spectral_delta'] for w in worlds]
        spectral_results[str(n)]={'mean':float(np.mean(vals)),'per_world':vals}
    rows=[]
    for code_seed in range(a.random_seeds):
        seed_results={}
        code=random_codes(42000+code_seed)
        for count in (18,30):
            scores=[run_target(w,count,table,spectral,code)['new_route']
                    ['learned_random_delta'] for w in worlds]
            seed_results[str(count)]={'mean':float(np.mean(scores)),
                                      'per_world':scores}
        rows.append({'code_seed':42000+code_seed,'loads':seed_results})
    summary={}
    for n in ('18','30'):
        rs=np.array([z['loads'][n]['mean'] for z in rows])
        sv=spectral_results[n]['mean']
        summary[n]={'spectral':sv,'random_mean':float(rs.mean()),
                    'random_std':float(rs.std(ddof=1)),
                    'random_min':float(rs.min()),'random_max':float(rs.max()),
                    'spectral_minus_random_mean':float(sv-rs.mean()),
                    'spectral_beats_random_seeds':int(np.sum(sv>rs)),
                    'number_random_seeds':len(rs)}
    report={
      'status':'post-result seed sensitivity; not preregistered',
      'source_seeds':list(range(3000,3008)),
      'target_worlds':list(worlds),
      'spectral':spectral_results,
      'random_code_seed_trials':rows,
      'summary':summary}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'random_seed_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Post-result seed sensitivity:',json.dumps(summary,indent=2))
if __name__=='__main__':main()
