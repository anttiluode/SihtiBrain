"""Gate 4: place-cell-style sparse lift and noisy-action sensory correction.

A post-Gate-3 controlled test. Learned action graph is reused unchanged;
its 12D spectral keys have distinct state identities, not missing information.
Expanded keys are fixed random nonlinear features, never target-trained.
"""
from dataclasses import dataclass
import numpy as np
from learned_relations import (
    NSTATES,NVALUES,DIM,ACTIONS,learned_sources,spectral_codes,graph_state)
from tem_addressing import (
    SIDE,STEP,make_world,canonical_path,different_path)

PROJECTION_SEED=41
ERROR_RATES=(0.,.15,.30)
SENSOR_ERROR=.05
SOURCE_SEEDS=tuple(range(3000,3008))
TARGET_SEEDS=tuple(range(1000,1020))
LOADS=(18,30)
REPEATS=3
ETA=.75

def normalize_rows(a):
    a=np.asarray(a,dtype=np.float64)
    n=np.linalg.norm(a,axis=1,keepdims=True)
    if np.any(n<1e-12):raise ValueError("zero feature code")
    return (a/n).astype(np.float32)

def sparse_relu(features,topk):
    features=np.maximum(0.,np.asarray(features,dtype=np.float64))
    if topk<features.shape[1]:
        order=np.argsort(features,axis=1)
        keep=order[:,-topk:]
        out=np.zeros_like(features)
        np.put_along_axis(out,keep,np.take_along_axis(features,keep,axis=1),axis=1)
        features=out
    return normalize_rows(features)

def codebooks(spectral):
    """All keys share exactly the same learned state addresses."""
    rng=np.random.default_rng(PROJECTION_SEED)
    r64=rng.normal(0,1/np.sqrt(DIM),size=(64,DIM)).astype(np.float32)
    r256=rng.normal(0,1/np.sqrt(DIM),size=(256,DIM)).astype(np.float32)
    randomstate=np.random.default_rng(33041).normal(size=(NSTATES,256))
    return {
        'spectral12':(normalize_rows(spectral),0,int(spectral.nbytes)),
        'sparse64':(sparse_relu(spectral@r64.T,8),int(r64.nbytes),int(spectral.nbytes)),
        'sparse256':(sparse_relu(spectral@r256.T,16),int(r256.nbytes),int(spectral.nbytes)),
        'dense256':(sparse_relu(spectral@r256.T,256),int(r256.nbytes),int(spectral.nbytes)),
        'random_state256':(sparse_relu(randomstate,16),0,int(NSTATES*256*4)),
        'onehot36':(np.eye(NSTATES,dtype=np.float32),0,0),
    }

@dataclass
class KeyMemory:
    codes:np.ndarray
    graph_bytes:int
    projection_bytes:int
    code_bytes:int
    def __post_init__(self):
        self.w=np.zeros((NVALUES,self.codes.shape[1]),dtype=np.float32)
    def write_state(self,state,label):
        key=self.codes[int(state)]
        target=np.zeros(NVALUES,dtype=np.float32)
        target[int(label)]=1.
        self.w+=np.float32(ETA)*np.outer(target-self.w@key,key)
    def predict_state(self,state):
        return int(np.argmax(self.w@self.codes[int(state)]))
    @property
    def storage_bytes(self):
        return self.graph_bytes+self.projection_bytes+self.code_bytes+self.w.nbytes


class IndexedGraph:
    """Same learned graph, exact associative labels. No oracle target coordinates."""
    def __init__(self,graph_bytes):
        self.labels={}
        self.graph_bytes=graph_bytes
    def write_state(self,state,label):
        self.labels[int(state)]=int(label)
    def predict_state(self,state):
        return self.labels.get(int(state),-1)
    @property
    def storage_bytes(self):
        return self.graph_bytes+len(self.labels)
    def value_posterior(self,belief):
        scores=np.zeros(NVALUES)
        for state,y in self.labels.items():scores[y]+=belief[state]
        if scores.sum()<1e-15:return -1
        return int(scores.argmax())


def path_state(table,path):
    return graph_state(table,path)

def visited_states(table,places,visited,symbols):
    """Only training-observed states acquire symbol metadata."""
    known=np.full(NSTATES,-1,dtype=int)
    for idx in visited:
        state=path_state(table,canonical_path(places[int(idx)]))
        known[state]=int(symbols[int(idx)])
    return known

def corrupt_walk(steps,world_symbols,rng,error_rate,sensor_error=SENSOR_ERROR):
    """WORLD GENERATOR ONLY. The reader gets recorded action and sensory token.

    Actual motion follows the original action. The odometry logger can
    substitute a different action; observations can also be independently
    corrupted. True nodes are returned only for evaluation diagnostics.
    """
    pos=(0,0);records=[]
    for a in steps:
        dx,dy=STEP[a]
        pos=((pos[0]+dx)%SIDE,(pos[1]+dy)%SIDE)
        true_symbol=int(world_symbols[pos[0]*SIDE+pos[1]])
        observed=true_symbol
        if rng.random()<sensor_error:
            observed=int(rng.choice([v for v in range(4) if v!=true_symbol]))
        reported=a
        if rng.random()<error_rate:
            reported=str(rng.choice([v for v in ACTIONS if v!=a]))
        records.append((reported,observed))
    return records,pos

def belief_filter(table,records,known_symbols,error_rate,sensor_error=SENSOR_ERROR):
    """Classical Bayes filter, with exactly the same noisy signals for both readers."""
    belief=np.zeros(NSTATES)
    belief[0]=1.
    known=np.asarray(known_symbols,dtype=int)
    for reported,sensed in records:
        nxt=np.zeros(NSTATES)
        for action in ACTIONS:
            pa=(1-error_rate if action==reported else error_rate/3)
            nxt[table[:,ACTIONS.index(action)]]+=pa*belief
        emissions=np.full(NSTATES,.25)
        recognized=known>=0
        emissions[recognized]=np.where(
            known[recognized]==sensed,1-sensor_error,sensor_error/3)
        nxt*=emissions
        total=nxt.sum()
        if total<1e-18:raise ArithmeticError("Bayesian posterior collapsed")
        belief=nxt/total
    return belief

def trained_memories(table,spectral,places,symbols,labels,visited):
    graph_bytes=int(table.nbytes)
    defs=codebooks(spectral)
    models={name:KeyMemory(code,graph_bytes,projection,codebytes)
            for name,(code,projection,codebytes) in defs.items()}
    index=IndexedGraph(graph_bytes)
    for _ in range(REPEATS):
        for idx in visited:
            state=path_state(table,canonical_path(places[int(idx)]))
            for model in models.values():
                model.write_state(state,labels[int(idx)])
            index.write_state(state,labels[int(idx)])
    return models,index

def accuracy(models,index,table,places,targets,indices):
    stats={}
    for name,memory in {**models,'indexed':index}.items():
        stats[name]=float(np.mean([
            memory.predict_state(path_state(table,different_path(places[int(i)])))
            ==targets[int(i)] for i in indices]))
    return stats

def noisy_trial(table,places,symbols,targets,visited,known,models,index,
                seed,load,error):
    """Identical emitted trajectories and filtering for all methods."""
    rng=np.random.default_rng(46000+seed*101+load*23+int(error*1000))
    results={k:[] for k in ('open_index','open_sparse256','map_index',
                           'map_sparse256','posterior_index','oracle_index',
                           'state_open','state_map','true_state_probability')}
    for i in visited:
        i=int(i)
        route=different_path(places[i])
        records,_=corrupt_walk(route,symbols,rng,error)
        broken=tuple(row[0] for row in records)
        true_state=path_state(table,route)
        open_state=path_state(table,broken)
        posterior=belief_filter(table,records,known,error)
        map_state=int(np.argmax(posterior))
        y=int(targets[i])
        results['open_index'].append(index.predict_state(open_state)==y)
        results['open_sparse256'].append(models['sparse256'].predict_state(open_state)==y)
        results['map_index'].append(index.predict_state(map_state)==y)
        results['map_sparse256'].append(models['sparse256'].predict_state(map_state)==y)
        results['posterior_index'].append(index.value_posterior(posterior)==y)
        results['oracle_index'].append(index.predict_state(true_state)==y)
        results['state_open'].append(open_state==true_state)
        results['state_map'].append(map_state==true_state)
        results['true_state_probability'].append(float(posterior[true_state]))
    return {key:float(np.mean(value)) for key,value in results.items()}

def run_world(table,spectral,seed,load):
    places,symbols,values,order=make_world(seed)
    visited=np.asarray(order[:load]);never=np.asarray(order[load:])
    models,index=trained_memories(table,spectral,places,symbols,values,visited)
    first=accuracy(models,index,table,places,values,visited)
    never_result=accuracy(models,index,table,places,values,never)
    known=visited_states(table,places,visited,symbols)
    noisy={f'{e:.2f}':noisy_trial(
        table,places,symbols,values,visited,known,models,index,seed,load,e)
           for e in ERROR_RATES}
    rng=np.random.default_rng(seed+50000)
    changed=np.sort(rng.choice(visited,size=max(1,len(visited)//4),replace=False))
    unchanged=np.setdiff1d(visited,changed)
    fresh=values.copy()
    for i in changed:
        fresh[i]=(int(values[i])+1+int(rng.integers(0,NVALUES-1)))%NVALUES
    for _ in range(REPEATS):
        for i in changed:
            state=path_state(table,canonical_path(places[int(i)]))
            for m in models.values():m.write_state(state,int(fresh[i]))
            index.write_state(state,int(fresh[i]))
    return {'seed':seed,'visited':load,'changed_count':len(changed),
            'new_route':first,'never_visited':never_result,
            'changed':accuracy(models,index,table,places,fresh,changed),
            'untouched':accuracy(models,index,table,places,values,unchanged),
            'noisy':noisy,
            'bytes':{**{name:m.storage_bytes for name,m in models.items()},
                     'indexed':index.storage_bytes,
                     'indexed_with_sensory_map':index.storage_bytes+len(visited),
                     'sparse256_with_sensory_map':models['sparse256'].storage_bytes+len(visited)}}

def aggregate(cases,load):
    sub=[c for c in cases if c['visited']==load]
    report={}
    for stage in ('new_route','changed','untouched','never_visited'):
        report[stage]={}
        for name in sub[0][stage]:
            x=np.asarray([case[stage][name] for case in sub])
            report[stage][name]={'mean':float(x.mean()),
                                 'std':float(x.std(ddof=1)),
                                 'per_world':list(map(float,x))}
    report['noisy']={}
    for rate in (f'{e:.2f}' for e in ERROR_RATES):
        report['noisy'][rate]={}
        for name in sub[0]['noisy'][rate]:
            x=np.asarray([c['noisy'][rate][name] for c in sub])
            report['noisy'][rate][name]={'mean':float(x.mean()),
                'std':float(x.std(ddof=1)),
                'per_world':list(map(float,x))}
    report['bytes']={name:float(np.mean([c['bytes'][name] for c in sub]))
                     for name in sub[0]['bytes']}
    return report

def benchmark(seeds=TARGET_SEEDS):
    table,agree=learned_sources(SOURCE_SEEDS)
    if not agree:raise RuntimeError("source worlds do not learn the same graph")
    spectral,_=spectral_codes(table)
    cases=[run_world(table,spectral,seed,load)
           for seed in seeds for load in LOADS]
    summary={str(load):aggregate(cases,load) for load in LOADS}
    def score(load,phase,method):
        return summary[str(load)][phase][method]['mean']
    gate_place=(score(30,'new_route','sparse256')-
                score(30,'new_route','spectral12')>=.10 and
                score(30,'untouched','sparse256')-
                score(30,'untouched','spectral12')>=.10 and
                score(30,'changed','sparse256')>=.95 and
                (score(30,'new_route','sparse256')>score(30,'new_route','dense256')
                 or score(30,'untouched','sparse256')>score(30,'untouched','dense256')))
    corrected=np.mean([summary[str(n)]['noisy']['0.15']['map_sparse256']['mean']-
                       summary[str(n)]['noisy']['0.15']['open_sparse256']['mean']
                       for n in LOADS])
    return {'protocol':{
             'learned_structure':"Gate 3 eight-source symbolic automaton",
             'world_seeds':list(seeds),'loads':LOADS,'noise_rates':ERROR_RATES,
             'sensor_noise':SENSOR_ERROR,'projection_seed':PROJECTION_SEED,
             'sparse64_topk':8,'sparse256_topk':16,'learning_rate':ETA,
             'writer_repeats':REPEATS,
             'used_target_coordinates_in_reader':False,
             'index_uses_same_belief_filter':True},
            'decisions':{
              'sparse_place_gate':bool(gate_place),
              'map_correction_gain_over_open_sparse15':float(corrected),
              'correction_gate':bool(corrected>=.05)},
            'summary':summary,'cases':cases}
