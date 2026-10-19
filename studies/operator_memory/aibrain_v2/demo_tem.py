"""Reproduce Gate 2's 20-world relational-addressing benchmark on CPU."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from tem_addressing import benchmark

METHODS=['phase_delta','phase_hebb','content_delta','time_delta',
         'wrong_action','indexed_coordinate','observed_graph',
         'no_memory','shuffled_labels']
LABELS=['Phase / delta','Phase / Hebb','Content / delta','Time / delta',
        'Wrong action','Indexed structural','Observed-edge graph',
        'No memory','Shuffled labels']

def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    fig.suptitle('AIbrainV2 | TEM-inspired structural addresses',fontsize=16)
    phases=[('first_new_route','New route to seen location'),
            ('rewritten','Updated values after 25% rewrites'),
            ('untouched','Unchanged values after rewrites'),
            ('never_visited','Unvisited: random labels are unknowable')]
    x=np.arange(len(METHODS))
    for ax,(phase,title) in zip(axes.ravel(),phases):
        for off,load in ((-.2,'18'),(.2,'30')):
            vals=[result['summary'][load][m][phase]['mean'] for m in METHODS]
            ax.bar(x+off,vals,width=.4,label=load+' visited')
        ax.axhline(1/8,ls=':',color='gray',label='chance = 1/8')
        ax.set(ylim=(0,1.04),xticks=x,xticklabels=LABELS,
               ylabel='accuracy',title=title)
        ax.tick_params(axis='x',rotation=37,labelsize=7)
        ax.legend(fontsize=7)
    fig.savefig(path,dpi=160)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/tem_gate2'))
    p.add_argument('--quick',action='store_true',help='2-world smoke, not a final result')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    data=benchmark(range(1000,1002) if a.quick else range(1000,1020))
    (a.output/'metrics.json').write_text(json.dumps(data,indent=2)+'\n')
    with (a.output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['visited','method','phase','accuracy','std_across_worlds','bytes'])
        for n,group in data['summary'].items():
            for method,details in group.items():
                for phase in ('first_new_route','never_visited','rewritten','untouched'):
                    v=details[phase]
                    w.writerow([n,method,phase,v['mean'],v['std'],details['bytes_mean']])
    plot(data,a.output/'tem_gate2.png')
    print('Address invariants:',data['invariants'])
    print('Preregistered verdicts:',data['decisions'])
    for n,group in data['summary'].items():
        print('\nVisited',n)
        for method,d in group.items():
            print(f'{method:21s}',
                  *(f'{d[p]["mean"]:.3f}' for p in
                    ('first_new_route','never_visited','rewritten','untouched')),
                  f'{d["bytes_mean"]:.0f} bytes')
    print('Saved:',a.output)

if __name__=='__main__':main()
