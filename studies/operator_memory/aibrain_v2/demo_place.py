"""Run AIbrainV2 Gate 4: sparse place expansion vs odometry + sensory fusion."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from place_expansion import benchmark

LABELS={
    'spectral12':'Spectral 12D',
    'sparse64':'Sparse lift 64D',
    'sparse256':'Sparse lift 256D',
    'dense256':'Dense ReLU 256D',
    'random_state256':'State hash 256D',
    'onehot36':'Full-band one-hot',
    'indexed':'Indexed graph'}
SELECTED=list(LABELS)
NOISE={
    'open_sparse256':'Sparse 256 / open loop',
    'map_sparse256':'Sparse 256 / Bayes MAP',
    'open_index':'Index / open loop',
    'map_index':'Index / Bayes MAP',
    'posterior_index':'Index / Bayes value posterior',
    'oracle_index':'Index / true path (oracle)'}

def figure(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(SELECTED))
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    fig.suptitle('AIbrainV2 Gate 4 | sparse place keys and sensory correction',fontsize=15)
    for ax,stage,name in ((axes[0,0],'new_route','A. New-route recall'),
                           (axes[0,1],'untouched','B. Untouched retention after rewrites')):
        for offset,n in ((-.2,'18'),(.2,'30')):
            data=report['summary'][n][stage]
            vals=[data[key]['mean'] for key in SELECTED]
            ax.bar(x+offset,vals,width=.4,label=n+' visited')
        ax.set(ylim=(0,1.05),ylabel='recall',title=name,
               xticks=x,xticklabels=[LABELS[k] for k in SELECTED])
        ax.tick_params(axis='x',rotation=35,labelsize=7)
        ax.legend(fontsize=8)
    ax=axes[1,0]
    for name,label in NOISE.items():
        vals=[np.mean([report['summary'][str(n)]['noisy'][f'{e:.2f}'][name]['mean']
                       for n in (18,30)]) for e in (0.,.15,.30)]
        ax.plot([0,15,30],vals,marker='o',label=label)
    ax.set(title='C. Noisy logged actions + aliased visual observations',
           xlabel='reported action substitution (%)',ylabel='query recall',ylim=(0,1.05))
    ax.legend(fontsize=7,ncol=2)
    ax=axes[1,1]
    byte_means=[report['summary']['18']['bytes'][k] for k in SELECTED]
    ax.bar(x,byte_means)
    ax.set(title='D. Persistent bytes: graph, code, projection, weights',
           ylabel='bytes',xticks=x,xticklabels=[LABELS[k] for k in SELECTED])
    ax.tick_params(axis='x',rotation=35,labelsize=7)
    fig.savefig(path,dpi=170)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('runs/place_gate4'))
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=benchmark(seeds=(1000,1001) if args.quick else range(1000,1020))
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['visited','stage','method','mean','std','stored_bytes'])
        for load,group in result['summary'].items():
            for stage in ('new_route','changed','untouched','never_visited'):
                for method,record in group[stage].items():
                    writer.writerow([load,stage,method,record['mean'],
                                     record['std'],group['bytes'].get(method,'')])
            for noise,data in group['noisy'].items():
                for method,record in data.items():
                    writer.writerow([load,f'noise_{noise}',method,record['mean'],
                                     record['std'],''])
    figure(result,args.output/'place_gate4.png')
    print('Gate 4 decisions:',result['decisions'])
    for n,g in result['summary'].items():
        print(f'\nVisited={n}')
        for name in SELECTED:
            print(f'{name:19s}',
                *(f'{g[stage][name]["mean"]:.3f}' for stage in
                  ('new_route','changed','untouched')),g['bytes'][name],'bytes')
        for e in ('0.00','0.15','0.30'):
            print(' action corruption',e,
                  {k:round(g['noisy'][e][k]['mean'],3) for k in NOISE})
    print('Outputs:',args.output)

if __name__=='__main__':main()
