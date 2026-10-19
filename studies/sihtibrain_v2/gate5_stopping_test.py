"""Gate 5 preregistered head-to-head: calcium ratio vs simpler three-factor rules.

See docs/GATE5_PREREGISTRATION.md. All active learners receive supervised
delayed tuft feedback in training; none receive labels at test time.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from apical_credit import PAIRS, LABELS, nine_compartment_operator

SEEDS = tuple(range(6))
METHODS = ('two_ping_log', 'three_factor_fractional', 'three_factor_event',
           'three_factor_sign', 'direct_gradient', 'no_context')
LEARNING_RATES = (.02, .05, .10, .20)
NOISE_LEVELS = (0.0, .03, .10)
DOMAINS = {
    'iid': ((.88, 1.12), (0., .06)),
    'low_contrast': ((.68, .92), (0., .12)),
    'high_contrast': ((1.08, 1.35), (0., .12)),
    'background': ((.88, 1.12), (.08, .25)),
}
TEST_DOMAINS = ('low_contrast', 'high_contrast', 'background')
ATTENUATION = np.array([.65, 1.10])
G = np.linalg.inv(nine_compartment_operator())
TRANSFER = float(G[0, 8])


def feature_map(x):
    """Exact vectorized fixed-feature map from V2 Gate 4."""
    x = np.asarray(x, dtype=float)
    return .5 + .5*np.tanh(10*(x @ G[[2,3,5,6]][:, [1,4]] - 2.)/2.)


def make_data(seed, count, domain):
    active, inactive = DOMAINS[domain]
    rng = np.random.default_rng(seed)
    xs, labels = [], []
    for pair, label in zip(PAIRS, LABELS):
        for _ in range(count):
            x = rng.uniform(*inactive, 4)
            x[list(pair)] = rng.uniform(*active, 2)
            xs.append(x)
            labels.append(label)
    order = rng.permutation(len(xs))
    return feature_map(np.asarray(xs)[order]), np.asarray(labels)[order]


def logistic(z):
    return .5 + .5*np.tanh(.5*np.asarray(z))


def metrics(phi, y, w, bias):
    score = logistic(phi@w + bias)
    pred = score >= .5
    pos, neg = y==1, y==0
    p = np.clip(score, 1e-9, 1-1e-9)
    return dict(
        balanced_accuracy=float((np.mean(pred[pos])+np.mean(~pred[neg]))/2),
        accuracy=float(np.mean(pred==y)),
        bce=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
        margin=float(np.min(score[pos])-np.max(score[neg])))


def updates(mode, phi, error, multiplicative_noise_early, multiplicative_noise_late):
    if mode == 'no_context':
        return np.zeros(2)
    if mode == 'direct_gradient':
        return np.array([error, error])
    p1 = 1+.85*np.tanh(4*3*TRANSFER*error)
    if mode == 'three_factor_event':
        return np.array([p1-1]*2)
    if mode == 'three_factor_sign':
        return np.array([0. if abs(p1-1)<.02 else np.sign(p1-1)]*2)
    a = 1.6*ATTENUATION
    ratio = np.expm1(a*p1)/np.expm1(a)*multiplicative_noise_late/multiplicative_noise_early
    if mode == 'two_ping_log':
        return np.log(ratio)
    if mode == 'three_factor_fractional':
        return ratio-1.
    raise ValueError(mode)


def noise_draws(seed, sigma, epochs, n):
    rng = np.random.default_rng(100300+seed)
    return np.maximum(.01, 1+sigma*rng.standard_normal((epochs,n,2,2)))


def train_once(method, seed, sigma, lr, train_set, orders, noises, epochs=35):
    """Slow scalar reference implementation retained for checking batched code."""
    phi, labels = train_set
    rng = np.random.default_rng(seed)
    w = rng.normal(0,.05,2)
    bias = -.20
    for ep in range(epochs):
        for j in orders[ep]:
            error = labels[j]-float(logistic(phi[j]@w+bias))
            signal = updates(method,phi[j],error,noises[ep,j,0],noises[ep,j,1])
            w += lr*phi[j]*signal
            bias += lr*float(np.mean(signal))
    return w,bias


def train_batched(seed, train_set, validation, orders, noises,
                  lr_grid=LEARNING_RATES):
    """Six update rules x four rates, same samples/order/noise and scalar math."""
    phi, y = train_set
    rates = np.asarray(lr_grid,float)
    m, nlr = len(METHODS), len(rates)
    rng = np.random.default_rng(seed)
    w = np.broadcast_to(rng.normal(0,.05,2), (m,nlr,2)).copy()
    bias = np.full((m,nlr),-.20)
    a = (1.6*ATTENUATION)[None,None,:]
    denom = np.expm1(a)
    for epoch, permutation in enumerate(orders):
        for idx in permutation:
            x = phi[idx]
            error = y[idx]-logistic(np.sum(w*x[None,None,:],axis=2)+bias)
            late = 1+.85*np.tanh(4*3*TRANSFER*error)
            ratio = (np.expm1(a*late[:,:,None])/denom) * (
                noises[epoch,idx,1]/noises[epoch,idx,0])[None,None,:]
            signal = np.zeros((m,nlr,2))
            signal[0] = np.log(ratio[0])
            signal[1] = ratio[1]-1.
            signal[2] = late[2,:,None]-1.
            event = late[3]-1.
            signal[3] = np.where(np.abs(event[:,None])<.02,0.,
                                 np.sign(event[:,None]))
            signal[4] = error[4,:,None]
            w += rates[None,:,None]*x[None,None,:]*signal
            bias += rates[None,:]*np.mean(signal,axis=2)

    vx,vy = validation
    scores = logistic(np.einsum('ni,mli->nml',vx,w)+bias[None,:,:])
    p = np.clip(scores,1e-9,1-1e-9)
    losses = np.mean(-vy[:,None,None]*np.log(p)
                     -(1-vy[:,None,None])*np.log(1-p), axis=0)
    selected = np.argmin(losses,axis=1)
    return {mode:(w[i,selected[i]].copy(),float(bias[i,selected[i]]),
                  float(rates[selected[i]]),float(losses[i,selected[i]]))
            for i,mode in enumerate(METHODS)}


def one_seed(seed, sigma, epochs=35, lr_grid=LEARNING_RATES):
    train_set = make_data(1000+seed,24,'iid')
    validation = make_data(2000+seed,24,'iid')
    tests = {domain:make_data(3000+seed+101*i,64,domain)
             for i,domain in enumerate(DOMAINS)}
    rng = np.random.default_rng(6000+seed)
    orders = np.array([rng.permutation(len(train_set[1]))
                       for _ in range(epochs)])
    noises = noise_draws(seed,sigma,epochs,len(train_set[1]))
    fitted = train_batched(seed,train_set,validation,orders,noises,lr_grid)
    return {mode:{
        'chosen_lr':lr,'validation_bce':val_loss,
        'final_weights':list(map(float,w)),'final_bias':bias,
        'domains':{domain:metrics(*data,w,bias) for domain,data in tests.items()}}
        for mode,(w,bias,lr,val_loss) in fitted.items()}


def run(seeds=SEEDS, noises=NOISE_LEVELS, epochs=35):
    runs = {}
    for sigma in noises:
        key = f'{sigma:.2f}'
        runs[key] = {str(seed):one_seed(seed,sigma,epochs) for seed in seeds}
    summary = {}
    for noise, results in runs.items():
        summary[noise] = {}
        for mode in METHODS:
            summary[noise][mode] = {}
            for domain in DOMAINS:
                rows = [results[str(seed)][mode]['domains'][domain]
                        for seed in seeds]
                summary[noise][mode][domain] = {
                    key:float(np.mean([row[key] for row in rows]))
                    for key in ('balanced_accuracy','accuracy','bce','margin')}
            summary[noise][mode]['shifted_bacc'] = float(np.mean([
                summary[noise][mode][d]['balanced_accuracy'] for d in TEST_DOMAINS]))
            summary[noise][mode]['shifted_bce'] = float(np.mean([
                summary[noise][mode][d]['bce'] for d in TEST_DOMAINS]))
    primary = summary['0.03']
    candidate = primary['two_ping_log']
    baselines = [(m,primary[m]) for m in
                 ('three_factor_fractional','three_factor_event','three_factor_sign')]
    best_name,best = max(baselines,key=lambda x:x[1]['shifted_bacc'])
    decision = dict(
        criterion='>=2pp advantage over best non-gradient three-factor in shifted bacc AND lower shifted BCE',
        best_baseline=best_name,
        candidate_shifted_bacc=candidate['shifted_bacc'],
        candidate_shifted_bce=candidate['shifted_bce'],
        baseline_shifted_bacc=best['shifted_bacc'],
        baseline_shifted_bce=best['shifted_bce'],
        passes=bool(candidate['shifted_bacc']>=best['shifted_bacc']+.02 and
                    candidate['shifted_bce']<best['shifted_bce']))
    return {
        'protocol':{'train_per_seed':144,'validation_per_seed':144,
                    'test_per_domain_per_seed':384,
                    'seeds':list(seeds),'epochs':epochs,
                    'learning_rates':list(LEARNING_RATES),
                    'noise_levels':list(noises),'domains':DOMAINS,
                    'transfer':TRANSFER,'methods':METHODS},
        'summary':summary,'decision':decision,'runs':runs}


def plot_result(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(METHODS)
    short = ['Log Ca ratio','Fractional Ca','Graded event',
             'Ternary event','Direct SGD','No context']
    fig,ax = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('SihtiBrainV2 Gate 5 | calcium-ratio stopping test',fontsize=16)
    x = np.arange(len(names))
    for i,noise in enumerate(('0.00','0.03','0.10')):
        vals = [result['summary'][noise][m]['shifted_bacc'] for m in names]
        ax[0,0].bar(x+(i-1)*.24,vals,width=.24,label=f'{float(noise)*100:.0f}% Ca noise')
    ax[0,0].set(xticks=x,xticklabels=short,ylim=(0,1.04),
                ylabel='balanced accuracy',title='A. All shifted domains')
    ax[0,0].tick_params(axis='x',rotation=25,labelsize=8)
    ax[0,0].legend(fontsize=8)

    for name,label in zip(names,short):
        ax[0,1].plot([0,3,10],[result['summary'][noise][name]['shifted_bce']
                        for noise in ('0.00','0.03','0.10')],
                     marker='o',label=label)
    ax[0,1].set(xlabel='Calcium read noise (%)',ylabel='binary cross entropy',
                title='B. Shifted calibration under read noise')
    ax[0,1].legend(fontsize=8,ncol=2)

    domains = ['iid',*TEST_DOMAINS]
    for name,label in zip(names[:5],short[:5]):
        ax[1,0].plot(range(4),
            [result['summary']['0.03'][name][d]['balanced_accuracy']
             for d in domains],marker='o',label=label)
    ax[1,0].set(xticks=range(4),
                xticklabels=['In-domain','Low contrast','High contrast','Background'],
                ylabel='balanced accuracy',ylim=(0,1.04),
                title='C. Shift breakdown (3% noise)')
    ax[1,0].tick_params(axis='x',rotation=14,labelsize=8)
    ax[1,0].legend(fontsize=7,ncol=2)

    ax[1,1].axis('off')
    d = result['decision']
    conclusion = 'PASS' if d['passes'] else 'NO DISTINCTIVE ADVANTAGE'
    lines = ['Preregistered stopping decision',conclusion,'',
             f"Two ping:   {d['candidate_shifted_bacc']*100:.1f}% balanced accuracy",
             f"Best baseline: {d['baseline_shifted_bacc']*100:.1f}% ({d['best_baseline']})",
             '', 'Same cable, same basal features, same contexts.',
             'Learning rates picked ONLY by in-domain validation.',
             'No target information at testing.']
    ax[1,1].text(.03,.91,'\n'.join(lines),va='top',fontsize=12,
                 family='monospace',transform=ax[1,1].transAxes)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/gate5_stopping_test'))
    parser.add_argument('--epochs',type=int,default=35)
    parser.add_argument('--quick',action='store_true',help='smoke run, not final')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result = run(seeds=(0,1) if args.quick else SEEDS,
                 noises=(0.,.03) if args.quick else NOISE_LEVELS,
                 epochs=5 if args.quick else args.epochs)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    if not args.quick:
        plot_result(result,args.output/'gate5_stopping_test.png')
    print('Decision:',json.dumps(result['decision'],indent=2))
    print('Saved:',args.output)


if __name__=='__main__':
    main()
