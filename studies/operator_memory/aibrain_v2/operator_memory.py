"""AIbrainV2: queryable passive response geometry vs equal-byte fast weights.

Graph memory is a grounded bipartite Laplacian. Its equilibrium transfer is
linear in the cue for fixed connections, and NONLINEAR in stored conductances.
Fast weights contain exactly the same stored matrix and use direct readout.
"""
from dataclasses import dataclass
import numpy as np

CUE_BITS = 16
VALUE_BITS = 8
LEAK = .25
STEP_SIZE = .1
MATRIX_BYTES = CUE_BITS * VALUE_BITS * 4


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x-np.max(x,axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def encode_key(active_positions):
    k = np.zeros(CUE_BITS, dtype=float)
    k[np.asarray(active_positions,dtype=int)] = 1.
    if int(k.sum()) != 4:
        raise ValueError("key must have four unique active bits")
    return k


def make_operator(w, leak=LEAK):
    """SPD grounded Laplacian on 16 cue + 8 value nodes."""
    w = np.asarray(w,dtype=float)
    if w.shape != (CUE_BITS,VALUE_BITS) or np.any(w<0):
        raise ValueError("conductances must be a nonnegative 16x8 matrix")
    d = CUE_BITS + VALUE_BITS
    h = np.zeros((d,d))
    h[np.arange(CUE_BITS),np.arange(CUE_BITS)] = leak+w.sum(axis=1)
    j = np.arange(VALUE_BITS)+CUE_BITS
    h[j,j] = leak+w.sum(axis=0)
    h[:CUE_BITS,CUE_BITS:] = -w
    h[CUE_BITS:,:CUE_BITS] = -w.T
    return h


def response_matrix(w, leak=LEAK):
    """Exact Schur complement: z_values = T(W) @ cue."""
    w=np.asarray(w,dtype=float)
    if np.any(w<0) or w.shape!=(CUE_BITS,VALUE_BITS):
        raise ValueError("expected nonnegative 16x8 conductance matrix")
    inv_dk=1./(leak+w.sum(axis=1))
    coupling=w.T*inv_dk[None,:]
    schur=np.diag(leak+w.sum(axis=0))-coupling@w
    return np.linalg.solve(schur,coupling)


def euler_relaxation(w,key,dt=.03,steps=10000,tol=1e-9):
    """Passive voltage gradient descent; optional convergence check."""
    h=make_operator(w)
    drive=np.zeros(CUE_BITS+VALUE_BITS)
    drive[:CUE_BITS]=key
    z=np.zeros_like(drive)
    for _ in range(steps):
        residual=drive-h@z
        if np.max(np.abs(residual))<tol:
            break
        z=z+dt*residual
    return z


@dataclass
class ConnectionMemory:
    graph: bool
    w: np.ndarray

    @classmethod
    def empty(cls,graph=True):
        return cls(graph,np.zeros((CUE_BITS,VALUE_BITS),dtype=np.float32))

    def write(self,key,label,scale=STEP_SIZE):
        """Shared add-only Hebbian writer, identical for both read architectures."""
        if int(np.sum(key))!=4:
            raise ValueError("key must contain exactly four active bits")
        if not (0<=label<VALUE_BITS):
            raise ValueError("invalid label")
        self.w[:,label]+=np.asarray(key,dtype=np.float32)*(scale/4.)

    def read(self,keys):
        q=np.asarray(keys,dtype=float)
        if q.shape[-1]!=CUE_BITS:
            raise ValueError("cue must have 16 positions")
        t=response_matrix(self.w) if self.graph else self.w.T.astype(float)
        return q@t.T

    def predict(self,keys):
        return np.argmax(self.read(keys),axis=-1)

    def reset_activity(self):
        """Queries never persist in this state; only connections are stored."""
        pass

    def erase_connections(self):
        self.w[:]=0.

    @property
    def storage_bytes(self):
        return self.w.nbytes



class DegreeNormalizedMemory(ConnectionMemory):
    """Post-result diagnostic: first-order, no-relaxation Schur approximation.

    Approximates the graph Green's function as Dv^-1 G^T Dk^-1.
    Stores EXACTLY the same W and uses the same Hebbian writer. It does
    NOT claim to be part of the preregistered primary two-way comparison.
    """
    def __init__(self):
        super().__init__(False,np.zeros((CUE_BITS,VALUE_BITS),dtype=np.float32))

    def read(self,keys):
        q=np.asarray(keys,dtype=float)
        dk=LEAK+self.w.sum(axis=1)
        dv=LEAK+self.w.sum(axis=0)
        t=(self.w.T/dk[None,:])/dv[:,None]
        return q@t.T


class IndexedMemory:
    """Strong packed-key control: two cue bytes + one label byte per binding."""
    def __init__(self):
        self.table={}

    @staticmethod
    def pack(key):
        return sum(int(bit)<<i for i,bit in enumerate(key))

    def write(self,key,label):
        self.table[self.pack(key)]=int(label)

    def predict(self,keys):
        if not self.table:
            return np.zeros(len(np.atleast_2d(keys)),dtype=int)
        stored=np.asarray(list(self.table.keys()),dtype=np.uint16)
        values=np.asarray(list(self.table.values()),dtype=int)
        out=[]
        for key in np.atleast_2d(keys):
            target=self.pack(key)
            distances=np.array([int(target^int(s)).bit_count() for s in stored])
            out.append(values[int(np.argmin(distances))])
        return np.asarray(out)

    def erase_connections(self):
        self.table.clear()

    def reset_activity(self):
        pass

    @property
    def storage_bytes(self):
        return 3*len(self.table)


def dataset(seed,load):
    """Unique four-hot keys with random independent labels."""
    rng=np.random.default_rng(seed)
    keys_set=set()
    while len(keys_set)<load:
        keys_set.add(tuple(sorted(rng.choice(CUE_BITS,4,replace=False))))
    keys=np.stack([encode_key(bits) for bits in sorted(keys_set)])
    labels=rng.integers(0,VALUE_BITS,size=load,dtype=int)
    return keys,labels


def swapped_queries(keys,seed,repeats=3):
    """Held-out 1-for-1 bit swaps, preserving four active bits."""
    rng=np.random.default_rng(seed)
    q=[]
    for key in keys:
        for _ in range(repeats):
            x=key.copy()
            a=int(rng.choice(np.flatnonzero(x)))
            b=int(rng.choice(np.flatnonzero(1-x)))
            x[a]=0.
            x[b]=1.
            q.append(x)
    return np.stack(q)


def _score_by_class(pred,labels):
    """Macro accuracy over labels present in the queried set."""
    labels=np.asarray(labels)
    pred=np.asarray(pred)
    present=np.unique(labels)
    return float(np.mean([np.mean(pred[labels==k]==k) for k in present]))


def evaluate(models,keys,labels,seed):
    noisy=swapped_queries(keys,seed)
    correct_noisy=np.repeat(labels,3)
    return {name:{
        'exact':float(np.mean(memory.predict(keys)==labels)),
        'one_swap':float(np.mean(memory.predict(noisy)==correct_noisy)),
        'exact_macro':_score_by_class(memory.predict(keys),labels),
        'one_swap_macro':_score_by_class(memory.predict(noisy),correct_noisy)}
        for name,memory in models.items()}


def run_case(load,seed):
    keys,labels=dataset(10100+seed*100+load,load)
    graph=ConnectionMemory.empty(True)
    direct=ConnectionMemory.empty(False)
    indexed=IndexedMemory()
    degree=DegreeNormalizedMemory()
    models={'operator':graph,'fast_weight':direct,
            'degree_normalized':degree,'indexed':indexed}
    for _ in range(3):
        for key,label in zip(keys,labels):
            for memory in models.values():
                memory.write(key,int(label))
    assert np.array_equal(graph.w,direct.w), "paired writer mismatch"
    assert np.array_equal(graph.w,degree.w), "diagnostic writer mismatch"
    before=evaluate(models,keys,labels,seed=22000+seed*100+load)

    rng=np.random.default_rng(30100+seed*100+load)
    rewritten=rng.choice(load,size=max(1,load//4),replace=False)
    untouched=np.setdiff1d(np.arange(load),rewritten)
    next_labels=labels.copy()
    for idx in rewritten:
        candidates=[i for i in range(VALUE_BITS) if i!=labels[idx]]
        next_labels[idx]=int(rng.choice(candidates))
    for _ in range(3):
        for idx in rewritten:
            key=keys[idx]
            for memory in models.values():
                memory.write(key,int(next_labels[idx]))
    assert np.array_equal(graph.w,direct.w), "paired rewrite mismatch"
    assert np.array_equal(graph.w,degree.w), "diagnostic rewrite mismatch"
    new=evaluate(models,keys[rewritten],next_labels[rewritten],
                 seed=41000+seed*100+load)
    retention=evaluate(models,keys[untouched],labels[untouched],
                       seed=51000+seed*100+load)

    no_activity={}
    for name,memory in models.items():
        old=evaluate({name:memory},keys,next_labels,seed=61000+seed*100+load)
        memory.reset_activity()
        after=evaluate({name:memory},keys,next_labels,seed=61000+seed*100+load)
        no_activity[name]=old==after

    shuffled=ConnectionMemory.empty(True)
    shuffled_labels=rng.permutation(labels)
    for _ in range(3):
        for key,label in zip(keys,shuffled_labels):
            shuffled.write(key,int(label))
    shuffled_score=float(np.mean(shuffled.predict(keys)==labels))
    bytes_by_model={k:v.storage_bytes for k,v in models.items()}
    return dict(load=load,seed=seed,before=before,
                rewritten_count=int(len(rewritten)),
                remaining_count=int(len(untouched)),
                rewritten=new,untouched=retention,
                stored_bytes=bytes_by_model,
                activity_reset_invariant=no_activity,
                shuffled_label_exact_accuracy=shuffled_score,
                matrix_identical_after_rewrite=bool(np.array_equal(graph.w,direct.w)))


def benchmark(loads=(8,16,32,64),seeds=tuple(range(12))):
    cases=[run_case(load,seed) for load in loads for seed in seeds]
    methods=('operator','fast_weight','degree_normalized','indexed')
    summary={}
    for load in loads:
        subset=[c for c in cases if c['load']==load]
        summary[str(load)]={}
        for mode in methods:
            summary[str(load)][mode]={}
            for phase in ('before','rewritten','untouched'):
                for query in ('exact','one_swap','exact_macro','one_swap_macro'):
                    v=np.array([c[phase][mode][query] for c in subset])
                    summary[str(load)][mode][f'{phase}_{query}']={
                        'mean':float(v.mean()),
                        'std':float(v.std(ddof=1)) if len(v)>1 else 0.,
                        'min':float(v.min()),'max':float(v.max())}
            summary[str(load)][mode]['storage_bytes']=subset[0]['stored_bytes'][mode]
    op=np.mean([c['untouched']['operator']['one_swap'] for c in cases])
    direct=np.mean([c['untouched']['fast_weight']['one_swap'] for c in cases])
    op_new=np.mean([c['rewritten']['operator']['one_swap'] for c in cases])
    direct_new=np.mean([c['rewritten']['fast_weight']['one_swap'] for c in cases])
    decision={'gate':'operator read must improve noisy untouched retention by >=0.05 '
                     'and not lose >0.02 on new noisy bindings',
              'operator_retention':float(op),'direct_retention':float(direct),
              'operator_rewrites':float(op_new),'direct_rewrites':float(direct_new),
              'passes':bool(op-direct>=.05 and op_new-direct_new>=-.02)}
    return dict(spec={'loads':list(loads),'seeds':list(seeds),
                      'matrix_bytes':MATRIX_BYTES,'writer':'identical add-only Hebb',
                      'rewrite_fraction':.25,'noisy_query':'one active bit swap',
                      'degree_normalized':'POST-RESULT diagnostic, not primary gate'},
                summary=summary,decision=decision,cases=cases)
