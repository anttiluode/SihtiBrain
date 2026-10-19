"""Gate 4: context at the tuft.

Gate 3 injected the context (teacher) at the soma.  In a pyramidal cell the
context arrives at the apical tuft, far up a trunk.  This module adds that
second zone to the same basal tree:

    tuft  T  ── trunk n_L ── ... ── trunk n_1 ── soma 0 ── basal branches (as before)
                  ^ optional regenerative "hotspot" at the top of the trunk

The trunk compartments are passive except the hotspot, which carries the same
local regenerative current as the basal roots (an abstract Ca-hotspot, not a
channel model).  Every term is reciprocal or local, so the dynamics are still
gradient flow on an energy and the Jacobian H(v) is symmetric.

Key fact used throughout: the soma is a CUT VERTEX between the trunk and the
basal tree.  For any symmetric H whose graph has that cut, the inverse
factorises across it,

    [H^-1]_{iT} = [H^-1]_{i0} [H^-1]_{0T} / [H^-1]_{00}     (i basal, T apical),

so context injected anywhere apical reaches every basal synapse exactly as if
it had been injected at the soma, scaled by one number
alpha = [H^-1]_{0T} / [H^-1]_{00}.  Apical geometry cannot distort the
DIRECTION of basal credit; it only sets its SIZE.  The question becomes
signal against read noise.
"""

import numpy as np

BASAL_EDGES = ((0, 1, 0.8), (0, 4, 0.8), (1, 2, 0.5), (1, 3, 0.5), (4, 5, 0.5), (4, 6, 0.5))
ROOTS = (1, 4)
LEAVES = (2, 3, 5, 6)


def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _softplus(z):
    return np.logaddexp(0.0, z)


class TuftTree:
    def __init__(self, trunk=4, g_trunk=0.5, leak=0.15, a_root=0.27, k=6.0,
                 theta_root=1.1, a_hot=0.0, k_hot=6.0, theta_hot=1.0, w=None, b=0.0):
        self.trunk = int(trunk)
        self.n = 7 + self.trunk + 1
        self.tuft = self.n - 1
        self.hot = self.n - 2 if self.trunk >= 1 else None
        edges = list(BASAL_EDGES)
        chain = [0] + list(range(7, 7 + self.trunk)) + [self.tuft]
        edges += [(chain[i], chain[i + 1], g_trunk) for i in range(len(chain) - 1)]
        self.edges = edges
        self.leak, self.k = leak, k
        self.a_root, self.theta_root = a_root, theta_root
        self.a_hot, self.k_hot, self.theta_hot = a_hot, k_hot, theta_hot
        self.w = np.ones(4) if w is None else np.array(w, float)
        self.b = float(b)
        lap = np.zeros((self.n, self.n))
        for i, j, g in edges:
            lap[i, i] += g; lap[j, j] += g; lap[i, j] -= g; lap[j, i] -= g
        self.h0 = lap + leak * np.eye(self.n)

    # local regenerative sites: (node, gain, steepness, threshold)
    def _sites(self):
        s = [(r, self.a_root, self.k, self.theta_root) for r in ROOTS]
        if self.hot is not None and self.a_hot > 0:
            s.append((self.hot, self.a_hot, self.k_hot, self.theta_hot))
        return s

    def drive(self, x):
        cur = np.zeros(self.n)
        cur[list(LEAVES)] = self.w * np.asarray(x, float)
        cur[0] += self.b
        return cur

    def grad(self, v, x, extra=None):
        g = self.h0 @ v - self.drive(x)
        for node, a, k, th in self._sites():
            g[node] -= a * _sig(k * (v[node] - th))
        if extra is not None:
            g -= extra
        return g

    def hessian(self, v):
        h = self.h0.copy()
        for node, a, k, th in self._sites():
            s = _sig(k * (v[node] - th))
            h[node, node] -= a * k * s * (1 - s)
        return h

    def relax(self, x, v0=None, extra=None, dt=0.05, steps=600):
        v = np.zeros(self.n) if v0 is None else np.array(v0, float)
        for _ in range(steps):
            g = self.grad(v, x, extra)
            v = v - dt * g
            if np.max(np.abs(g)) < 1e-9:
                break
        for _ in range(60):
            g = self.grad(v, x, extra)
            if np.max(np.abs(g)) < 1e-13:
                break
            v = v - np.linalg.solve(self.hessian(v), g)
        return v

    def min_curvature(self, v):
        return float(np.linalg.eigvalsh(self.hessian(v)).min())


# ---- the calcium readout of gate 3, on this tree -----------------------------
class Readout:
    def __init__(self, amp=2.0, width=0.5, ca_thr=5.0, ca_k=4.0, noise=0.0, t_end=6.0, nt=241):
        self.amp, self.width, self.ca_thr, self.ca_k, self.noise = amp, width, ca_thr, ca_k, noise
        self.t = np.linspace(0.0, t_end, nt)

    def calcium(self, tree, v, x, rng=None):
        lam, phi = np.linalg.eigh(tree.hessian(v))
        t = self.t[:, None]
        on = (1 - np.exp(-lam * np.minimum(t, self.width))) / lam
        off = np.where(t > self.width, np.exp(-lam * (t - self.width)), 1.0)
        delta = (self.amp * on * off * phi[0]) @ phi.T
        d = delta[:, list(LEAVES)]
        rest = v[list(LEAVES)]
        g = _sig(self.ca_k * (rest + d - self.ca_thr)) - _sig(self.ca_k * (rest - self.ca_thr))
        ca = np.asarray(x, float) * np.trapezoid(g, self.t, axis=0)
        if self.noise and rng is not None:
            ca = ca * (1 + self.noise * rng.standard_normal(4))
        return ca


def event_code(err, kind, dead=0.0):
    """Few-level code for the context after it has travelled as a regenerative
    event: 'ternary' = burst (+1) / nothing (0) / pause (-1); 'burst' = one-sided."""
    if kind == "ternary":
        return float(np.sign(err)) if abs(err) > dead else 0.0
    if kind == "burst":
        return 1.0 if err > dead else 0.0
    raise ValueError(kind)


def context_states(tree, x, y, beta, where="tuft", dead=0.0):
    """Free state, and the state with context added.

    where = "soma"     graded current beta*(y - v0_free) at the soma
            "tuft"     the same graded current at the tuft (passive trunk)
            "ternary"  an event that reaches the soma at full size: beta * {+1, 0, -1}
            "burst"    one-sided event: beta * {+1, 0}
    The event variants assume the trunk is regenerative, so the event does not
    attenuate with trunk length; only its code survives."""
    v0 = tree.relax(x)
    extra = np.zeros(tree.n)
    err = y - v0[0]
    if where == "soma":
        extra[0] = beta * err
    elif where == "tuft":
        extra[tree.tuft] = beta * err
    else:
        extra[0] = beta * event_code(err, where, dead)
    vb = tree.relax(x, v0=v0, extra=extra)
    return v0, vb


def local_update(tree, ro, x, y, beta, where="tuft", rng=None, dead=0.0):
    v0, vb = context_states(tree, x, y, beta, where, dead)
    ca0, cab = ro.calcium(tree, v0, x, rng), ro.calcium(tree, vb, x, rng)
    act = (np.asarray(x) > 0) & (ca0 > 0) & (cab > 0)
    dw = np.zeros(4)
    dw[act] = np.log(cab[act] / ca0[act]) / (ro.ca_k * beta)
    db = (vb[0] - v0[0]) / beta
    return dw, db, v0, vb


def true_descent(tree, x, y):
    """-d/dw 1/2 (v0* - y)^2 via the implicit function theorem (exact)."""
    v = tree.relax(x)
    hinv0 = np.linalg.solve(tree.hessian(v), np.eye(tree.n)[0])
    err = v[0] - y
    return -err * hinv0[list(LEAVES)] * np.asarray(x, float), -err * hinv0[0]


def alignment(tree, ro, xs, ys, beta, where="tuft", rng=None, dead=0.0):
    upd, tru = [], []
    for x, y in zip(xs, ys):
        upd.append(local_update(tree, ro, x, y, beta, where, rng, dead)[0])
        tru.append(true_descent(tree, x, y)[0])
    u, t = np.concatenate(upd), np.concatenate(tru)
    nu, nt = np.linalg.norm(u), np.linalg.norm(t)
    return float(u @ t / (nu * nt)) if nu > 0 and nt > 0 else 0.0


def attenuation(tree, x):
    """alpha = [H^-1]_{0T} / [H^-1]_{00} at the free state, and the
    cut-vertex identity residual for the basal leaves."""
    v = tree.relax(x)
    hinv = np.linalg.inv(tree.hessian(v))
    alpha = hinv[0, tree.tuft] / hinv[0, 0]
    lhs = hinv[list(LEAVES), tree.tuft]
    rhs = hinv[list(LEAVES), 0] * hinv[0, tree.tuft] / hinv[0, 0]
    return float(alpha), float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)))


def train_exact(tree, xs, ys, epochs, lr=0.05, seed=0):
    """Adam on the exact gradient (used to make probe states)."""
    rng = np.random.default_rng(seed)
    p = np.r_[tree.w, tree.b]; m = np.zeros(5); s = np.zeros(5)
    for ep in range(epochs):
        step = np.zeros(5)
        for i in rng.permutation(len(xs)):
            dw, db = true_descent(tree, xs[i], ys[i])
            step += np.r_[dw, db] / len(xs)
        m = 0.9 * m + 0.1 * step; s = 0.999 * s + 0.001 * step ** 2
        p = p + lr * (m / (1 - 0.9 ** (ep + 1))) / (np.sqrt(s / (1 - 0.999 ** (ep + 1))) + 1e-8)
        tree.w, tree.b = p[:4].copy(), float(p[4])


def train_local(tree, ro, xs, ys, beta, where="tuft", epochs=300, lr=0.05, seed=0, dead=0.0):
    rng = np.random.default_rng(seed)
    p = np.r_[tree.w, tree.b]; m = np.zeros(5); s = np.zeros(5)
    for ep in range(epochs):
        step = np.zeros(5)
        for i in rng.permutation(len(xs)):
            dw, db, _, _ = local_update(tree, ro, xs[i], ys[i], beta, where, rng, dead)
            step += np.r_[dw, db] / len(xs)
        m = 0.9 * m + 0.1 * step; s = 0.999 * s + 0.001 * step ** 2
        p = p + lr * (m / (1 - 0.9 ** (ep + 1))) / (np.sqrt(s / (1 - 0.999 ** (ep + 1))) + 1e-8)
        tree.w, tree.b = p[:4].copy(), float(p[4])
    outs = np.array([tree.relax(x)[0] for x in xs])
    thr = 0.5 * (max(ys) + min(ys))
    correct = int(np.sum((outs > thr) == (np.array(ys) > thr)))
    return float(0.5 * np.mean((outs - np.array(ys)) ** 2)), outs, correct


def task(hi=0.6, lo=0.3):
    pairs = [((2, 3), 1), ((5, 6), 1), ((2, 5), 0), ((2, 6), 0), ((3, 5), 0), ((3, 6), 0)]
    xs, ys = [], []
    for pair, lab in pairs:
        x = np.zeros(4)
        for leaf in pair:
            x[LEAVES.index(leaf)] = 1.0
        xs.append(x); ys.append(hi if lab else lo)
    return xs, ys


def fresh(seed, **kw):
    r = np.random.default_rng(100 + seed)
    return TuftTree(w=r.uniform(0.3, 1.2, 4), b=r.uniform(-1, 0), **kw)
