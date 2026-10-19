"""Reproduce Gate 3 structure learning and held-out relational memory test."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from learned_relations import benchmark

NAMES={
 'learned_spectral_delta':'Learned graph + spectral delta',
 'learned_random_delta':'Learned graph + random delta',
 'learned_spectral_hebb':'Learned graph + spectral Hebb',
 'content_delta':'Content key + delta',
 'time_delta':'Time key + delta',
 'learned_indexed':'Learned graph + index',
 'observed_edges_only':'Target observed edges',
 'no_memory':'No memory'}

def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(report['summary']['18'])
    x=np.arange(len(methods))
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    fig.suptitle('AIbrainV2 Gate 3: learned action structure, held-out worlds',fontsize=15)
    for ax,(key,title) in zip(axes.ravel(),(
       ('new_route','Unseen route to an observed location'),
       ('rewritten','Recall rewritten values'),
       ('untouched','Retention of unchanged associations'),
       ('never_visited','Unvisited random labels: not inferable'))):
        for delta,load in ((-.2,'18'),(.2,'30')):
            vals=[report['summary'][load][m][key]['mean'] for m in methods]
            ax.bar(x+delta,vals,width=.4,label=load+' visited')
        ax.axhline(.125,ls=':',color='gray',label='1/8 chance')
        ax.set(ylim=(0,1.04),title=title,xticks=x,
               xticklabels=[NAMES[m] for m in methods],ylabel='accuracy')
        ax.tick_params(axis='x',rotation=36,labelsize=7)
        ax.legend(fontsize=7)
    fig.savefig(path,dpi=155)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/learned_gate3'))
    p.add_argument('--quick',action='store_true',help='smoke run, not preregistered test')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report=benchmark(source_seeds=range(3000,3002) if args.quick else range(3000,3008),
                     target_seeds=range(1000,1002) if args.quick else range(1000,1020))
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['visited','method','phase','accuracy','std','stored_bytes'])
        for count,rows in report['summary'].items():
            for method,scored in rows.items():
                for phase in ('new_route','rewritten','untouched','never_visited'):
                    v=scored[phase]
                    w.writerow([count,method,phase,v['mean'],v['std'],scored['bytes_mean']])
    plot(report,args.output/'learned_gate3.png')
    print('Invariant checks:',report['invariants'])
    print('Decisions:',report['decisions'])
    for count,rows in report['summary'].items():
        print(f'VISITED {count}')
        for name,d in rows.items():
            print(name,[(k,round(d[k]['mean'],4)) for k in
                        ('new_route','rewritten','untouched','never_visited')],
                  'bytes',d['bytes_mean'])
    print('Outputs:',args.output)

if __name__=='__main__':main()
