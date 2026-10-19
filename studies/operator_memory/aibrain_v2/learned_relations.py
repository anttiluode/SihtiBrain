"""Gate 3: identify a transition automaton from sensory landmark walks.

The generator uses a 6x6 torus to create *source data*, but the learner receives
only opaque landmark-before, action and landmark-after triples. No coordinates
or torus formula are supplied to the learner. On target worlds symbols alias,
while action semantics and the start-anchor remain shared.

This is explicit finite-state system identification + classical spectral
embedding, NOT TEM, gradient-trained structural learning or hippocampal data.
"""
from collections import deque
import numpy as np
from tem_addressing import (
    SIDE, DIM, NVALUES, BETA, MEMORY_BYTES, STEP,
    make_world, canonical_path, different_path, DeltaMemory, ObservedGraph,
    NoMemory, evaluate)

ACTIONS=('E','N','W','S')
NSTATES=36
SOURCE_SEEDS=tuple(range(3000,3008))
TARGET_SEEDS=tuple(range(1000,1020))
WALK_STEPS=2400

def source_observations(seed,walk_steps=WALK_STEPS):
    """WORLD GENERATOR ONLY: randomized landmark identities per hidden node."""
    rng=np.random.default_rng(seed)
    landmarks=[f'landmark-{i:03d}' for i in rng.permutation(NSTATES)]
    current=(0,0)
    observations=[]
    for _ in range(walk_steps):
        a=ACTIONS[int(rng.integers(0,len(ACTIONS)))]
        dx,dy=STEP[a]
        nextpos=((current[0]+dx)%SIDE,(current[1]+dy)%SIDE)
        i=current[0]*SIDE+current[1]
        j=nextpos[0]*SIDE+nextpos[1]
        observations.append((landmarks[i],a,landmarks[j]))
        current=nextpos
    return landmarks[0],observations


def learn_automaton(anchor,observations):
    """Learner sees ONLY landmark strings and action-labeled transitions.

    Canonical BFS numbers states by the shared action language relative
    to the reset landmark, never by a ground-truth location.
    """
    edge={}
    landmarks={anchor}
    for before,action,after in observations:
        if action not in ACTIONS:raise ValueError("unknown action")
        landmarks.update((before,after))
        pair=(before,action)
        if pair in edge and edge[pair]!=after:
            raise ValueError("inconsistent transition observation")
        edge[pair]=after
    if len(landmarks)!=NSTATES:
        raise ValueError(f"only {len(landmarks)} of {NSTATES} landmarks observed")
    if len(edge)!=NSTATES*len(ACTIONS):
        raise ValueError(f"only {len(edge)} of {NSTATES*len(ACTIONS)} edges observed")
    ids={anchor:0}
    todo=deque([anchor])
    while todo:
        old=todo.popleft()
        for action in ACTIONS:
            target=edge[(old,action)]
            if target not in ids:
                ids[target]=len(ids)
                todo.append(target)
    if len(ids)!=NSTATES:raise ValueError("not all states reachable")
    transition=np.zeros((NSTATES,len(ACTIONS)),dtype=np.uint8)
    for state,i in ids.items():
        for j,action in enumerate(ACTIONS):
            transition[i,j]=ids[edge[(state,action)]]
    return transition


def learned_sources(seeds=SOURCE_SEEDS):
    tables=[]
    for seed in seeds:
        anchor,obs=source_observations(seed)
        tables.append(learn_automaton(anchor,obs))
    agreement=all(np.array_equal(tables[0],t) for t in tables[1:])
    return tables[0],agreement


def graph_state(table,actions):
    state=0
    for a in actions:state=int(table[state,ACTIONS.index(a)])
    return state


def spectral_codes(table):
    graph=np.zeros((NSTATES,NSTATES),dtype=float)
    for src in range(NSTATES):
        for dest in table[src]:
            graph[src,int(dest)]=1.
            graph[int(dest),src]=1.
    degree=np.diag(graph.sum(axis=1))
    eigenvalues,eigenvectors=np.linalg.eigh(degree-graph)
    codes=eigenvectors[:,1:DIM+1]
    norms=np.linalg.norm(codes,axis=1)
    if np.any(norms<=1e-10):raise ValueError("spectral key vanished")
    return (codes/norms[:,None]).astype(np.float32),eigenvalues[:DIM+2]


def random_codes(seed):
    rng=np.random.default_rng(seed)
    code=rng.standard_normal((NSTATES,DIM))
    return (code/np.linalg.norm(code,axis=1,keepdims=True)).astype(np.float32)


class LearnedMatrix:
    def __init__(self,table,codes,writer="delta"):
        self.table=np.asarray(table)
        self.codes=np.asarray(codes,dtype=np.float32)
        self.writer=writer
        self.w=np.zeros((NVALUES,DIM),dtype=np.float32)
    def key(self,actions):
        return self.codes[graph_state(self.table,actions)]
    def write(self,actions,symbol,label):
        k=self.key(actions)
        target=np.zeros(NVALUES,dtype=np.float32);target[int(label)]=1.
        update=target-self.w@k if self.writer=='delta' else target
        self.w+=np.float32(BETA)*np.outer(update,k)
    def predict(self,actions,symbol):
        return int(np.argmax(self.w@self.key(actions)))
    @property
    def byte_count(self):
        # Include the *learned state table*, structural codes AND fast weights.
        return int(self.table.nbytes+self.codes.nbytes+self.w.nbytes)


class LearnedIndexed:
    def __init__(self,table):
        self.table=np.asarray(table)
        self.values={}
    def write(self,actions,symbol,label):
        self.values[graph_state(self.table,actions)]=int(label)
    def predict(self,actions,symbol):
        return self.values.get(graph_state(self.table,actions),-1)
    @property
    def byte_count(self):
        return int(self.table.nbytes+len(self.values))


def path_invariants(table,codes):
    changed=[]
    for x in range(SIDE):
        for y in range(SIDE):
            p=(x,y)
            a=graph_state(table,canonical_path(p))
            b=graph_state(table,different_path(p))
            changed.append(a!=b)
    d=np.linalg.norm(codes[:,None,:]-codes[None,:,:],axis=2)
    np.fill_diagonal(d,np.inf)
    return {
        'unique_states':len(set(graph_state(table,canonical_path((x,y)))
                                 for x in range(SIDE) for y in range(SIDE))),
        'route_errors':sum(changed),
        'minimum_code_distance':float(d.min()),
        'closed_loop_returns':graph_state(table,('E','N','W','S'))==0}


def bad_action_control(seed=3000):
    anchor,obs=source_observations(seed)
    rng=np.random.default_rng(seed+947)
    scrambled=[(a,ACTIONS[int(rng.integers(0,4))],b) for a,_,b in obs]
    try:learn_automaton(anchor,scrambled)
    except ValueError:return True
    return False


def run_target(seed,nvisited,table,spectral,random):
    places,symbols,values,order=make_world(seed)
    visited=order[:nvisited]
    missing=order[nvisited:]
    models={
      'learned_spectral_delta':LearnedMatrix(table,spectral,'delta'),
      'learned_random_delta':LearnedMatrix(table,random,'delta'),
      'learned_spectral_hebb':LearnedMatrix(table,spectral,'hebb'),
      'content_delta':DeltaMemory('content'),
      'time_delta':DeltaMemory('time'),
      'learned_indexed':LearnedIndexed(table),
      'observed_edges_only':ObservedGraph(),
      'no_memory':NoMemory()
    }
    def writes(indices,label_array):
        for i in indices:
            path=canonical_path(places[i])
            for m in models.values():
                m.write(path,int(symbols[i]),int(label_array[i]))
    for _ in range(3):writes(visited,values)
    def score(indices,target):
        return {name:evaluate(m,indices,places,symbols,target,different_path)
                for name,m in models.items()}
    first=score(visited,values);unvisited=score(missing,values)
    rng=np.random.default_rng(seed+50000)
    changed=np.sort(rng.choice(visited,size=max(1,len(visited)//4),replace=False))
    untouched=np.setdiff1d(visited,changed)
    labels=values.copy()
    for idx in changed:
        labels[idx]=(int(values[idx])+1+int(rng.integers(0,NVALUES-1)))%NVALUES
    for _ in range(3):writes(changed,labels)
    return {
        'seed':int(seed),'visited':int(nvisited),'changed_count':len(changed),
        'new_route':first,'never_visited':unvisited,
        'rewritten':score(changed,labels),
        'untouched':score(untouched,values),
        'bytes':{name:m.byte_count for name,m in models.items()}
    }


def benchmark(source_seeds=SOURCE_SEEDS,target_seeds=TARGET_SEEDS):
    table,agrees=learned_sources(source_seeds)
    spectral,evals=spectral_codes(table)
    random=random_codes(42)
    invariants=path_invariants(table,spectral)
    invariants['source_agreement']=bool(agrees)
    invariants['source_worlds']=len(source_seeds)
    invariants['transition_states']=len(table)
    invariants['transition_edges']=int(table.size)
    invariants['shuffled_action_rejected']=bad_action_control()
    invariants['passes']=(agrees and invariants['unique_states']==36
                           and invariants['route_errors']==0
                           and invariants['closed_loop_returns']
                           and invariants['minimum_code_distance']>1e-6
                           and invariants['shuffled_action_rejected'])
    if not invariants['passes']:
        raise RuntimeError(f"necessary structure identification failed: {invariants}")
    cases=[run_target(seed,n,table,spectral,random)
           for seed in target_seeds for n in (18,30)]
    methods=list(cases[0]['new_route'])
    summary={}
    for n in (18,30):
        subset=[r for r in cases if r['visited']==n]
        group={}
        for method in methods:
            d={}
            for phase in ('new_route','never_visited','rewritten','untouched'):
                vals=np.array([r[phase][method] for r in subset],float)
                d[phase]={'mean':float(vals.mean()),
                          'std':float(vals.std(ddof=1)) if len(vals)>1 else 0.,
                          'per_world':list(map(float,vals))}
            d['bytes_mean']=float(np.mean([r['bytes'][method] for r in subset]))
            group[method]=d
        summary[str(n)]=group
    def val(n,method):
        return summary[str(n)][method]['new_route']['mean']
    candidate=val(18,'learned_spectral_delta')
    content=val(18,'content_delta');time=val(18,'time_delta')
    random=val(18,'learned_random_delta')
    classic=val(18,'learned_indexed')
    decisions={
      'gate0_structure_invariance':bool(invariants['passes']),
      'gate1_vs_content_time':bool(candidate-max(content,time)>=.20
                          and candidate>=.45 and val(30,'learned_spectral_delta')>=.325),
      'gate2_vs_random_state_codes':bool(candidate-random>=.05),
      'hard_superiority_vs_structured_index':bool(candidate>classic and
         summary['18']['learned_spectral_delta']['bytes_mean'] <=
         summary['18']['learned_indexed']['bytes_mean']),
      'spectral_accuracy_18':candidate,
      'content_accuracy_18':content,
      'time_accuracy_18':time,
      'random_codes_accuracy_18':random,
      'indexed_accuracy_18':classic}
    return {
      'protocol':{'source_worlds':list(source_seeds),
          'walk_steps_per_source':WALK_STEPS,
          'target_worlds':list(target_seeds),'visited':[18,30],
          'action_order':list(ACTIONS),
          'learned_structure':'opaque sensory tokens and action triples',
          'geometry_given_to_learner':False,
          'source_unique_landmarks':True,
          'target_sensory_aliases':4,
          'phase_code_size':DIM,
          'matrix_bytes':MEMORY_BYTES,
          'stored_graph_bytes':int(table.nbytes),
          'spectral_code_bytes':int(spectral.nbytes),
          'action_table':table.tolist(),
          'spectrum_low':list(map(float,evals))},
       'invariants':invariants,'decisions':decisions,
       'summary':summary,'cases':cases}
