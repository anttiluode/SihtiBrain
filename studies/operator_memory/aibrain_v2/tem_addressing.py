"""Gate 2: TEM-inspired *given-structure* relational addressing.
The action algebra is GIVEN, not learned. New-route recall is tested against
equal-byte content/time keys and stronger classical indexed structural memory.
"""
import numpy as np
from dataclasses import dataclass

SIDE, DIM, NVALUES, BETA = 6, 12, 8, .75
MEMORY_BYTES = DIM*NVALUES*4
STEP = {'E':(1,0),'W':(-1,0),'N':(0,1),'S':(0,-1)}
WAVES = np.array([(1,0),(0,1),(1,1),(1,2),(2,1),(2,2)],float)

def unit(x):
    x=np.asarray(x,dtype=float)
    return x/np.linalg.norm(x)

class PhaseIntegrator:
    def __init__(self):
        self.phase=np.ones(6,dtype=np.complex128)
    def step(self,a):
        delta=np.asarray(STEP[a])
        self.phase *= np.exp(2j*np.pi*(WAVES@delta)/SIDE)
    def follow(self,actions):
        for a in actions:self.step(a)
        return self.key()
    def key(self):
        return unit(np.stack((self.phase.real,self.phase.imag),axis=-1).ravel())

class CoordinateIntegrator:
    """Classical baseline: integrate actions with the same known structural prior."""
    def __init__(self):
        self.coord=(0,0)
    def follow(self,actions):
        x,y=self.coord
        for a in actions:
            dx,dy=STEP[a];x=(x+dx)%SIDE;y=(y+dy)%SIDE
        self.coord=x,y
        return self.coord

def time_key(t):
    a=2*np.pi*np.arange(1,7)*t/19.
    return unit(np.stack((np.cos(a),np.sin(a)),axis=-1).ravel())

def content_key(symbol):
    k=np.zeros(DIM);k[int(symbol)]=1.;return k

def canonical_path(pos):
    x,y=pos
    return ('E',)*x+('N',)*y

def different_path(pos):
    x,y=pos
    return ('N','S','W','E')+('N',)*y+('E',)*x

@dataclass
class DeltaMemory:
    mode:str
    writer:str='delta'
    def __post_init__(self):
        self.w=np.zeros((NVALUES,DIM),dtype=np.float32)
    def key(self,actions,symbol):
        if self.mode=='phase':return PhaseIntegrator().follow(actions)
        if self.mode=='content':return content_key(symbol)
        if self.mode=='time':return time_key(len(actions))
        if self.mode=='wrong_action':
            # A constant action permutation would STILL preserve a group law,
            # so corrupt every other action to make paths inconsistent.
            flip={'E':'N','W':'S','N':'E','S':'W'}
            actions=tuple(a if i%2==0 else flip[a] for i,a in enumerate(actions))
            return PhaseIntegrator().follow(actions)
        raise ValueError(self.mode)
    def write(self,actions,symbol,label):
        k=self.key(actions,symbol)
        target=np.zeros(NVALUES);target[int(label)]=1.
        error=target-self.w@k if self.writer=='delta' else target
        self.w += np.float32(BETA)*np.outer(error,k).astype(np.float32)
    def predict(self,actions,symbol):
        return int(np.argmax(self.w@self.key(actions,symbol)))
    @property
    def byte_count(self):return self.w.nbytes

class NoMemory:
    def write(self,actions,symbol,label):pass
    def predict(self,actions,symbol):return 0
    @property
    def byte_count(self):return 0

class IndexedCoordinate:
    def __init__(self):self.table={}
    def write(self,actions,symbol,label):
        self.table[CoordinateIntegrator().follow(actions)]=int(label)
    def predict(self,actions,symbol):
        return self.table.get(CoordinateIntegrator().follow(actions),-1)
    @property
    def byte_count(self):return 3*len(self.table)

class ObservedGraph:
    """Records only edges seen along training paths; missing edge = no recall.
    Uses same known torus structure to merge observed path endpoints.
    """
    def __init__(self):
        self.edges={};self.values={};self.seen_paths={}
    def observe_path(self,actions):
        curr=(0,0);self.seen_paths.setdefault(curr,0)
        for a in actions:
            nxt=CoordinateIntegrator();nxt.coord=curr
            nextcoord=nxt.follow((a,))
            self.seen_paths.setdefault(nextcoord,len(self.seen_paths))
            self.edges[(self.seen_paths[curr],a)]=self.seen_paths[nextcoord]
            curr=nextcoord
        return self.seen_paths[curr]
    def write(self,actions,symbol,label):
        self.values[self.observe_path(actions)]=int(label)
    def predict(self,actions,symbol):
        node=self.seen_paths.get((0,0),-1)
        for a in actions:
            node=self.edges.get((node,a),-1)
            if node<0:return -1
        return self.values.get(node,-1)
    @property
    def byte_count(self):return 3*len(self.values)+5*len(self.edges)

def make_world(seed):
    rng=np.random.default_rng(seed)
    locations=[(x,y) for x in range(SIDE) for y in range(SIDE)]
    symbols=rng.integers(0,4,size=36)
    values=rng.integers(0,NVALUES,size=36)
    order=np.r_[0,rng.permutation(np.arange(1,36))]
    return locations,symbols,values,order

def evaluate(model,indices,locations,symbols,values,route):
    if len(indices)==0:return float('nan')
    return float(np.mean([
        model.predict(route(locations[i]),int(symbols[i]))==int(values[i])
        for i in indices]))

def run_world(seed,nvisited):
    places,symbols,values,order=make_world(seed)
    visited=order[:nvisited];unvisited=order[nvisited:]
    models={
        'phase_delta':DeltaMemory('phase'),
        'phase_hebb':DeltaMemory('phase','hebb'),
        'content_delta':DeltaMemory('content'),
        'time_delta':DeltaMemory('time'),
        'wrong_action':DeltaMemory('wrong_action'),
        'indexed_coordinate':IndexedCoordinate(),
        'observed_graph':ObservedGraph(),
        'no_memory':NoMemory(),
        'shuffled_labels':DeltaMemory('phase'),
    }
    shuffled=values.copy()
    rng=np.random.default_rng(seed+70000)
    shuffled[visited]=rng.permutation(values[visited])
    def write(indices,current):
        for i in indices:
            path=canonical_path(places[i])
            for name,m in models.items():
                y=shuffled[i] if name=='shuffled_labels' else current[i]
                m.write(path,int(symbols[i]),int(y))
    for _ in range(3):write(visited,values)
    def score(indices,route,target):
        return {name:evaluate(m,indices,places,symbols,target,route)
                for name,m in models.items()}
    before=score(visited,different_path,values)
    never=score(unvisited,different_path,values)
    rng=np.random.default_rng(seed+50000)
    rewritten=np.sort(rng.choice(visited,size=max(1,len(visited)//4),replace=False))
    untouched=np.setdiff1d(visited,rewritten)
    next_values=values.copy()
    for i in rewritten:
        alt=int(rng.integers(0,NVALUES-1))
        next_values[i]=(int(values[i])+1+alt)%NVALUES
    for _ in range(3):write(rewritten,next_values)
    return {'seed':seed,'visited':nvisited,'rewrites':len(rewritten),
        'first_new_route':before,'never_visited':never,
        'rewritten':score(rewritten,different_path,next_values),
        'untouched':score(untouched,different_path,values),
        'bytes':{name:m.byte_count for name,m in models.items()}}

def address_invariants():
    errors=[];keys=[]
    for x in range(SIDE):
        for y in range(SIDE):
            a=PhaseIntegrator().follow(canonical_path((x,y)))
            b=PhaseIntegrator().follow(different_path((x,y)))
            errors.append(float(np.max(np.abs(a-b))));keys.append(a)
    keys=np.asarray(keys)
    distance=np.linalg.norm(keys[:,None,:]-keys[None,:,:],axis=2)
    distance+=100*np.eye(36)
    loop=PhaseIntegrator().follow(('E','N','W','S'))
    anchor=PhaseIntegrator().key()
    max_route=max(errors)
    max_loop=float(np.max(abs(loop-anchor)))
    min_dist=float(distance.min())
    return {'max_route_error':max_route,'max_closed_loop_error':max_loop,
        'minimum_distinct_address_distance':min_dist,
        'passes':bool(max_route<1e-9 and max_loop<1e-9 and min_dist>1e-6)}

def summarize(cases):
    methods=list(cases[0]['first_new_route'])
    summary={}
    for n in (18,30):
        subset=[c for c in cases if c['visited']==n]
        summary[str(n)]={}
        for name in methods:
            d={}
            for phase in ('first_new_route','never_visited','rewritten','untouched'):
                values=[c[phase][name] for c in subset]
                d[phase]={'mean':float(np.mean(values)),
                          'std':float(np.std(values,ddof=1)),
                          'per_world':list(map(float,values))}
            d['bytes_mean']=float(np.mean([c['bytes'][name] for c in subset]))
            summary[str(n)][name]=d
    a=summary['18'];candidate=a['phase_delta']['first_new_route']['mean']
    cont=a['content_delta']['first_new_route']['mean']
    tempo=a['time_delta']['first_new_route']['mean']
    candidate30=summary['30']['phase_delta']['first_new_route']['mean']
    return summary, {
        'gate0':address_invariants()['passes'],
        'gate1_structured_vs_unstructured':bool(candidate-max(cont,tempo)>=.2 and
            candidate>=.425 and candidate30>=.425),
        'algorithmic_advantage_over_indexed':bool(
            candidate>a['indexed_coordinate']['first_new_route']['mean'] and
            a['phase_delta']['bytes_mean']<=a['indexed_coordinate']['bytes_mean']),
        'phase_delta_visited18':candidate,
        'content_visited18':cont,'time_visited18':tempo}

def benchmark(seeds=range(1000,1020)):
    cases=[run_world(seed,n) for seed in seeds for n in (18,30)]
    summary,decisions=summarize(cases)
    return {'protocol':{'grid':'6x6 torus','learned_structure':False,
        'sensory_symbols':4,'possible_values':NVALUES,
        'seeds':list(seeds),'visited_counts':[18,30],
        'writer_repeats':3,'beta':BETA,'phase_memory_bytes':MEMORY_BYTES,
        'wave_vectors':WAVES.astype(int).tolist()},
        'invariants':address_invariants(),'decisions':decisions,
        'summary':summary,'cases':cases}
