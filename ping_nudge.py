"""The ping as a nudge: a dendritic tree as an energy-based model.

Same seven-compartment geometry as ``sihtibrain.Circuit`` (soma 0; branch
roots 1, 4; leaves 2, 3 on the left branch and 5, 6 on the right), but the
branch roots now carry a *voltage-dependent* regenerative current, so the
somatic readout and a somatic ping can both depend on hidden branch state.

Energy (unit capacitance, rest = 0):

    E(v) = 1/2 sum_edges g_ij (v_i - v_j)^2 + g_leak/2 |v|^2
           + sum_{r in roots} Phi(v_r) - sum_i w_i x_i v_i - b v_0

    Phi(v) = -(a/k) log(1 + exp(k (v - theta)))     (so -Phi'(v) = a sigma(k(v-theta)))

The regenerative current a*sigma(...) is an abstract stand-in for an NMDA-like
inward current that switches on with depolarisation.  It is NOT an NMDA
channel model (no Mg2+ block kinetics, no driving force, no conductance
dynamics).  Because every term is either reciprocal (g_ij = g_ji) or local,
the dynamics are an exact gradient flow  dv/dt = -grad E.

Equilibrium propagation (Scellier & Bengio 2017) then applies directly: add a
somatic current  -beta (v_0 - y)  (the "ping" carrying the error), let the
tree relax again, and the change in each local quantity dE/dtheta gives the
gradient of the somatic loss C = 1/2 (v_0 - y)^2 with respect to theta.
"""

import numpy as np

EDGES = ((0, 1, 0.8), (0, 4, 0.8),
         (1, 2, 0.5), (1, 3, 0.5),
         (4, 5, 0.5), (4, 6, 0.5))
ROOTS = (1, 4)
LEAVES = (2, 3, 5, 6)
N = 7


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _softplus(z):
    return np.logaddexp(0.0, z)


class Tree:
    """Gradient-flow cable tree with optional regenerative branch roots.

    Parameters live in plain attributes so that learning code can read and
    write them: ``g`` (6 axial conductances, order of EDGES), ``w`` (4 leaf
    synaptic weights, order of LEAVES), ``b`` (somatic bias current), ``a``
    (2 regenerative gains, order of ROOTS).
    """

    def __init__(self, a=0.0, k=6.0, theta=0.6, leak=0.15, g=None, w=None, b=0.0):
        self.k = float(k)
        self.theta = float(theta)
        self.leak = float(leak)
        self.g = np.array([e[2] for e in EDGES] if g is None else g, dtype=float)
        self.w = np.ones(4) if w is None else np.array(w, dtype=float)
        self.b = float(b)
        self.a = np.full(2, float(a)) if np.ndim(a) == 0 else np.array(a, dtype=float)

    # ---- operator pieces -------------------------------------------------
    def laplacian(self):
        lap = np.zeros((N, N))
        for (i, j, _), gij in zip(EDGES, self.g):
            lap[i, i] += gij
            lap[j, j] += gij
            lap[i, j] -= gij
            lap[j, i] -= gij
        return lap

    def drive(self, x):
        """External current: synapses on leaves plus somatic bias."""
        current = np.zeros(N)
        current[list(LEAVES)] = self.w * np.asarray(x, dtype=float)
        current[0] += self.b
        return current

    def regen_current(self, v):
        out = np.zeros(N)
        for a_r, r in zip(self.a, ROOTS):
            out[r] = a_r * _sigmoid(self.k * (v[r] - self.theta))
        return out

    # ---- energy, gradient, Hessian --------------------------------------
    def energy(self, v, x, extra=None):
        v = np.asarray(v, dtype=float)
        e = sum(0.5 * gij * (v[i] - v[j]) ** 2 for (i, j, _), gij in zip(EDGES, self.g))
        e += 0.5 * self.leak * v @ v
        for a_r, r in zip(self.a, ROOTS):
            e -= a_r / self.k * _softplus(self.k * (v[r] - self.theta))
        e -= self.drive(x) @ v
        if extra is not None:
            e -= extra @ v
        return float(e)

    def _h0(self):
        key = (self.g.tobytes(), self.leak)
        if getattr(self, "_h0_key", None) != key:
            self._h0_cache = self.laplacian() + self.leak * np.eye(N)
            self._h0_key = key
        return self._h0_cache

    def grad(self, v, x, extra=None):
        """dE/dv.  The physical velocity is dv/dt = -grad."""
        gr = self._h0() @ v
        gr -= self.regen_current(v) + self.drive(x)
        if extra is not None:
            gr -= extra
        return gr

    def hessian(self, v):
        h = self._h0().copy()
        for a_r, r in zip(self.a, ROOTS):
            s = _sigmoid(self.k * (v[r] - self.theta))
            h[r, r] -= a_r * self.k * s * (1 - s)
        return h

    # ---- physics ---------------------------------------------------------
    def relax(self, x, v0=None, nudge=None, dt=0.05, steps=400, polish=True):
        """Settle to an equilibrium by the physical gradient flow, then
        Newton-polish inside the basin that the flow reached.

        ``nudge`` is a callable v -> extra somatic/compartment current, used
        for the EqProp nudged phase (it depends on v_0, so it is part of the
        dynamics, not a fixed pulse).
        """
        v = np.zeros(N) if v0 is None else np.array(v0, dtype=float)
        for _ in range(steps):
            extra = None if nudge is None else nudge(v)
            g = self.grad(v, x, extra)
            v = v - dt * g
            if np.max(np.abs(g)) < 1e-9:
                break
        if polish:
            for _ in range(60):
                extra = None if nudge is None else nudge(v)
                g = self.grad(v, x, extra)
                if np.max(np.abs(g)) < 1e-13:
                    break
                h = self.hessian(v)
                if nudge is not None:            # d(extra)/dv_0 for the linear soma nudge
                    h = h.copy()
                    h[0, 0] += nudge.stiffness
                v = v - np.linalg.solve(h, g)
        return v

    def trajectory(self, x, steps, dt=0.02, v0=None, pulse=None):
        """Explicit gradient flow with an optional fixed somatic current pulse
        (start_step, stop_step, amplitude).  Returns (trace, energies)."""
        v = np.zeros(N) if v0 is None else np.array(v0, dtype=float)
        trace = [v.copy()]
        energy = [self.energy(v, x)]
        for t in range(steps):
            extra = np.zeros(N)
            if pulse is not None and pulse[0] <= t < pulse[1]:
                extra[0] = pulse[2]
            v = v - dt * self.grad(v, x, extra)
            trace.append(v.copy())
            energy.append(self.energy(v, x))
        return np.array(trace), np.array(energy)

    # ---- parameter sensitivities of E (local quantities) -----------------
    def dE_dparams(self, v, x):
        """Partial derivatives of E w.r.t. (g, w, b, a) at fixed v.  Each entry
        only needs quantities available where that parameter lives."""
        x = np.asarray(x, dtype=float)
        d_g = np.array([0.5 * (v[i] - v[j]) ** 2 for (i, j, _) in EDGES])
        d_w = -x * v[list(LEAVES)]                 # presynaptic x local voltage
        d_b = np.array([-v[0]])
        d_a = np.array([-_softplus(self.k * (v[r] - self.theta)) / self.k for r in ROOTS])
        return np.concatenate([d_g, d_w, d_b, d_a])

    def get_params(self):
        return np.concatenate([self.g, self.w, [self.b], self.a])

    def set_params(self, p):
        self.g = np.array(p[:6])
        self.w = np.array(p[6:10])
        self.b = float(p[10])
        self.a = np.array(p[11:13])


class SomaNudge:
    """Error current injected at the soma: -beta (v_0 - y).  This is the
    'ping' in the EqProp reading: graded, signed, proportional to error."""

    def __init__(self, beta, target):
        self.beta = float(beta)
        self.target = float(target)
        self.stiffness = self.beta

    def __call__(self, v):
        out = np.zeros(N)
        out[0] = -self.beta * (v[0] - self.target)
        return out


def eqprop_grad(tree, x, y, beta, v_free=None):
    """Symmetric-nudge EqProp estimate of d/dtheta [1/2 (v0* - y)^2]."""
    v_free = tree.relax(x) if v_free is None else v_free
    v_plus = tree.relax(x, v0=v_free, nudge=SomaNudge(+beta, y))
    v_minus = tree.relax(x, v0=v_free, nudge=SomaNudge(-beta, y))
    grad = (tree.dE_dparams(v_plus, x) - tree.dE_dparams(v_minus, x)) / (2 * beta)
    return grad, v_free, v_plus, v_minus


def loss_at(tree, params, x, y, v_init):
    t = Tree(k=tree.k, theta=tree.theta, leak=tree.leak)
    t.set_params(params)
    v = t.relax(x, v0=v_init)
    return 0.5 * (v[0] - y) ** 2


def finite_difference_grad(tree, x, y, eps=1e-6):
    """Central differences of the equilibrium loss, re-relaxing from the free
    equilibrium so each probe stays in the same basin."""
    p0 = tree.get_params()
    v_free = tree.relax(x)
    out = np.zeros_like(p0)
    for i in range(len(p0)):
        dp = np.zeros_like(p0)
        dp[i] = eps
        out[i] = (loss_at(tree, p0 + dp, x, y, v_free)
                  - loss_at(tree, p0 - dp, x, y, v_free)) / (2 * eps)
    return out


def implicit_grad(tree, x, y):
    """Exact gradient by the implicit function theorem (needs the Hessian
    inverse, i.e. non-local information).  Used only as ground truth."""
    v = tree.relax(x)
    h = tree.hessian(v)
    eps = 1e-7
    p0 = tree.get_params()
    cross = np.zeros((N, len(p0)))       # d(grad_v E)/d theta
    for i in range(len(p0)):
        dp = np.zeros_like(p0); dp[i] = eps
        tree.set_params(p0 + dp); gp = tree.grad(v, x)
        tree.set_params(p0 - dp); gm = tree.grad(v, x)
        cross[:, i] = (gp - gm) / (2 * eps)
    tree.set_params(p0)
    dv_dtheta = -np.linalg.solve(h, cross)
    return (v[0] - y) * dv_dtheta[0]


# ---- the cluster-detection task -----------------------------------------
# Inputs on two leaves.  Same-branch pairs -> high, cross-branch pairs -> low.
PAIRS = [((2, 3), 1.0), ((5, 6), 1.0),
         ((2, 5), 0.0), ((2, 6), 0.0), ((3, 5), 0.0), ((3, 6), 0.0)]


def pattern(pair, amp=1.0):
    x = np.zeros(4)
    for leaf in pair:
        x[LEAVES.index(leaf)] = amp
    return x


def task(amp=1.0, hi=1.2, lo=0.4):
    xs = [pattern(p, amp) for p, _ in PAIRS]
    ys = [hi if label else lo for _, label in PAIRS]
    return xs, ys


def linear_floor(xs, ys):
    """Best possible mean loss for ANY passive (a = 0) tree: at equilibrium the
    soma is affine in x for every choice of g, w, b, so the floor is the
    least-squares residual of an affine model on the six patterns."""
    X = np.column_stack([np.array(xs), np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(X, np.array(ys), rcond=None)
    resid = X @ coef - np.array(ys)
    return float(0.5 * np.mean(resid ** 2)), coef


def train(tree, xs, ys, beta=0.05, lr=0.05, epochs=400, train_g=True,
          train_a=True, seed=0, record_every=10):
    """Adam on EqProp gradients.  Conductances and gains are kept positive."""
    rng = np.random.default_rng(seed)
    p = tree.get_params()
    m = np.zeros_like(p); s = np.zeros_like(p)
    mask = np.ones_like(p)
    if not train_g:
        mask[:6] = 0
    if not train_a:
        mask[11:13] = 0
    history = []
    for ep in range(epochs):
        order = rng.permutation(len(xs))
        grad = np.zeros_like(p)
        for idx in order:
            gr, *_ = eqprop_grad(tree, xs[idx], ys[idx], beta)
            grad += gr / len(xs)
        grad *= mask
        m = 0.9 * m + 0.1 * grad
        s = 0.999 * s + 0.001 * grad ** 2
        mh = m / (1 - 0.9 ** (ep + 1)); sh = s / (1 - 0.999 ** (ep + 1))
        p = p - lr * mh / (np.sqrt(sh) + 1e-8)
        p[:6] = np.clip(p[:6], 0.05, None)        # axial conductances stay physical
        p[11:13] = np.clip(p[11:13], 0.0, None)   # regenerative gain >= 0
        tree.set_params(p)
        if ep % record_every == 0 or ep == epochs - 1:
            history.append((ep, evaluate(tree, xs, ys)[0]))
    return history


def evaluate(tree, xs, ys):
    outs = np.array([tree.relax(x)[0] for x in xs])
    loss = float(0.5 * np.mean((outs - np.array(ys)) ** 2))
    thr = 0.5 * (max(ys) + min(ys))
    correct = int(np.sum((outs > thr) == (np.array(ys) > thr)))
    return loss, outs, correct


# ---- observability of the linear soma ------------------------------------
def soma_observability(tree):
    """Eigenmodes of the passive operator and each mode's weight at the soma.
    A mode with zero somatic weight is invisible to the soma at every time."""
    h = tree.laplacian() + tree.leak * np.eye(N)
    lam, phi = np.linalg.eigh(h)
    return lam, phi, phi[0]


# ---- Sihti identity -------------------------------------------------------
def sihti_residues(h, x0, eta, steps):
    """Run the Sihti purifier x_{k+1} = x_k - eta H x_k and return residues.
    Identity: r_k = eta * H x_k, i.e. eta times (net axial current out of each
    compartment + leak current) for the cable whose operator is H."""
    xs = [np.array(x0, dtype=float)]
    res = []
    for _ in range(steps):
        r = eta * h @ xs[-1]
        res.append(r)
        xs.append(xs[-1] - r)
    return np.array(res), np.array(xs)
