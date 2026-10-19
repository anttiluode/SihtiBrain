"""Gate 3: the ping with state, up and down.

Gate 2 gave the soma a *graded error current* (an EqProp nudge) and let each
synapse read its own voltage change.  Two objections came back (Sol, 2026-10-19):

  * a real back-propagating spike is not an error signal, and
  * "state is not error".

This module drops the error current.  The cell instead has

  UP     inputs -> branch voltages -> soma (inference, gradient flow on E)
  TEACH  a weak top-down current at the soma, beta (y - v0), shifts the
         equilibrium (the apical/context drive; the target lives here, not
         in the spike)
  DOWN   a returning ping (bAP) leaves the soma with amplitude A(v0) that may
         depend on somatic state (Martin-Burgos et al. 2026; Brunner &
         Szabadics 2016) and spreads back through the same cable, linearised
         around the current state:  delta(t) = int exp(-H(v)(t-s)) e0 A(s) ds
  READ   each synapse turns ping + local state into calcium through an
         NMDA-like unblocking nonlinearity:
             Ca_i = x_i * int [ g(v_i + gamma_i delta_i(t)) - g(v_i) ] dt
         (ping-evoked calcium, baseline removed, as measured in bAP imaging)
  WRITE  the synapse compares the calcium of two pings, one sampled before
         the teacher arrives (its stored trace) and one during:
             dw_i = eta * (Ca_i^teach - Ca_i^free)

No signal anywhere encodes the error.  The error appears only as the
difference between two local states sampled by two pings.
"""

import numpy as np

from ping_nudge import Tree, N, LEAVES, task, implicit_grad, evaluate


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class PingReadout:
    """Parameters of the returning ping and the local calcium readout.

    amp      ping amplitude at the soma (current pulse height)
    width    pulse duration
    mod      analog modulation of the ping's LOCAL effect by somatic state,
             per leaf (scalar or 4-vector): local ping x (1 + mod_i (v0 - v_ref)).
             Brunner & Szabadics 2016 found the sign of this flips along the
             dendrite (proximal +, distal -), so it is per-synapse here.
    gamma    per-leaf down-path gain (1 = reciprocal cable; <0 = sign flip)
    ca_thr   calcium (NMDA unblock) threshold in local voltage
    ca_k     steepness of the unblock
    noise    std of multiplicative read noise on each calcium sample
    """

    def __init__(self, amp=2.0, width=0.5, mod=0.0, v_ref=0.45, gamma=None,
                 ca_thr=1.2, ca_k=8.0, noise=0.0, t_end=6.0, nt=241):
        self.amp, self.width, self.v_ref = amp, width, v_ref
        self.mod = np.full(4, float(mod)) if np.ndim(mod) == 0 else np.asarray(mod, float)
        self.gamma = np.ones(4) if gamma is None else np.asarray(gamma, float)
        self.ca_thr, self.ca_k, self.noise = ca_thr, ca_k, noise
        self.t = np.linspace(0.0, t_end, nt)


    def local_ping(self, tree, v):
        """delta_i(t) at the leaves for a rectangular somatic pulse, using the
        cable linearised at state v (symmetric Hessian -> exact eigen form)."""
        lam, phi = np.linalg.eigh(tree.hessian(v))
        a = self.amp
        w = self.width
        t = self.t[:, None]
        # response of each mode to a pulse of height a on [0, w]
        on = (1 - np.exp(-lam * np.minimum(t, w))) / lam
        off = np.where(t > w, np.exp(-lam * (t - w)), 1.0)
        modal = a * on * off * phi[0]              # (nt, modes)
        delta = modal @ phi.T                      # (nt, N)
        state_gain = 1.0 + self.mod * (v[0] - self.v_ref)
        return delta[:, list(LEAVES)] * state_gain

    def calcium(self, tree, v, x, rng=None):
        x = np.asarray(x, float)
        d = self.local_ping(tree, v) * self.gamma
        local = v[list(LEAVES)] + d
        rest = v[list(LEAVES)]
        g = _sigmoid(self.ca_k * (local - self.ca_thr)) - _sigmoid(self.ca_k * (rest - self.ca_thr))
        ca = x * np.trapezoid(g, self.t, axis=0)      # ping-EVOKED calcium (baseline removed)
        if self.noise and rng is not None:
            ca = ca * (1 + self.noise * rng.standard_normal(ca.shape))
        return ca


def teacher_state(tree, x, y, beta, v_free):
    from ping_nudge import SomaNudge
    return tree.relax(x, v0=v_free, nudge=SomaNudge(beta, y))


def ping_update(tree, readout, x, y, beta, rng=None, use_ping=True, compare="log"):
    """Local weight/bias update from two pings (free, then teacher).

    compare="log":  dw_i = [x_i > 0] * log(Ca_i^teach / Ca_i^free) / (k beta)
                    (fractional change of ping-evoked calcium; for an
                    exponential sub-threshold unblock curve this equals
                    k * dv_i exactly when the ping's local shape is
                    state-independent, i.e. it IS the EqProp update)
    compare="diff": dw_i = (Ca_i^teach - Ca_i^free) / beta   (absolute)"""
    v0 = tree.relax(x)
    vb = teacher_state(tree, x, y, beta, v0)
    if use_ping:
        ca0 = readout.calcium(tree, v0, x, rng)
        cab = readout.calcium(tree, vb, x, rng)
    else:  # no ping: nothing is evoked, so nothing is written
        ca0 = cab = np.zeros(4)
    if compare == "log":
        active = (np.asarray(x) > 0) & (ca0 > 0) & (cab > 0)
        dw = np.zeros(4)
        dw[active] = np.log(cab[active] / ca0[active]) / (readout.ca_k * beta)
    else:
        dw = (cab - ca0) / beta
    db = (vb[0] - v0[0]) / beta          # bias lives at the soma: its own voltage
    return dw, db


def eqprop_update(tree, x, y, beta):
    v0 = tree.relax(x)
    vb = teacher_state(tree, x, y, beta, v0)
    x = np.asarray(x, float)
    return x * (vb[list(LEAVES)] - v0[list(LEAVES)]) / beta, (vb[0] - v0[0]) / beta


def true_descent(tree, x, y):
    """-dLoss/d(w, b) from the implicit-function gradient."""
    g = implicit_grad(tree, x, y)
    return -g[6:10], -g[10]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def alignment(tree, readout, xs, ys, beta=0.02, rng=None, use_ping=True, shuffle=False,
              compare="log"):
    """Cosine between local updates and true descent on the weights, with the
    six patterns' updates CONCATENATED (a 24-vector), so that per-pattern
    credit is compared, not just the shared average push.  Bias excluded."""
    ys_used = list(ys)
    if shuffle:
        ys_used = list(np.random.default_rng(7).permutation(ys))
    upd, tru = [], []
    for x, y, yu in zip(xs, ys, ys_used):
        dw, _ = ping_update(tree, readout, x, yu, beta, rng, use_ping, compare)
        upd.append(dw)
        tru.append(true_descent(tree, x, y)[0])
    return cosine(np.concatenate(upd), np.concatenate(tru))


def train_ping(tree, readout, xs, ys, beta=0.02, lr=0.05, epochs=300, seed=0,
               rule="ping", shuffle=False, compare="log"):
    """Adam on the local updates (weights + somatic bias)."""
    rng = np.random.default_rng(seed)
    ys_used = list(np.random.default_rng(seed + 99).permutation(ys)) if shuffle else list(ys)
    p = np.r_[tree.w, tree.b]
    m = np.zeros(5); s = np.zeros(5)
    for ep in range(epochs):
        step = np.zeros(5)
        for i in rng.permutation(len(xs)):
            if rule == "eqprop":
                dw, db = eqprop_update(tree, xs[i], ys_used[i], beta)
            else:
                dw, db = ping_update(tree, readout, xs[i], ys_used[i], beta, rng,
                                     use_ping=(rule == "ping"), compare=compare)
            step += np.r_[dw, db] / len(xs)
        m = 0.9 * m + 0.1 * step
        s = 0.999 * s + 0.001 * step ** 2
        mh = m / (1 - 0.9 ** (ep + 1)); sh = s / (1 - 0.999 ** (ep + 1))
        p = p + lr * mh / (np.sqrt(sh) + 1e-8)      # ascent on the local update
        tree.w, tree.b = p[:4].copy(), float(p[4])
    return evaluate(tree, xs, ys)


def fresh_tree(seed):
    r = np.random.default_rng(100 + seed)
    return Tree(a=0.27, k=6, theta=1.1, w=r.uniform(0.3, 1.2, 4), b=r.uniform(-1, 0))
