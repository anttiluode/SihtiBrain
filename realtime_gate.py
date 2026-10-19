"""Gate 5: real-time credit, and a three-factor Hebbian rule as a serious competitor.

Everything that gates 3-4 scheduled by hand now runs in continuous time on the
two-zone tree of gate 4 (basal tree + trunk + tuft):

  * the soma fires its own pings as an inhomogeneous Poisson process whose rate
    depends on somatic voltage;
  * a ping is a brief somatic current pulse; with a probability set by the TUFT
    voltage it is a burst (extra pulses) - the BAC-style coupling;
  * the context (target information) arrives at the tuft after a delay, as a
    graded current; it reaches the soma only through the trunk and, much more
    strongly, through the burst probability it modulates;
  * each active synapse integrates calcium influx x_i * g(v_i(t)) continuously
    (g: the sub-threshold unblock curve of gate 3); the integral over an early
    window (before context) and a late window (after it) are its two "pings".

Rules compared (all share the same somatic-bias update, driven by the change in
burst count, so only the synaptic credit differs):

  ping-local   dw_i = eta * [x_i>0] * log(Ca_late_i / Ca_early_i) / k
  three-factor dw_i = eta * x_i * (bursts_late - bursts_early) / n_ref
               (presynaptic activity x one global signed signal; no local state)
  exact        the implicit-function gradient of the somatic loss (upper bound)

Why a three-factor rule should be competitive inside ONE cell: the Jacobian H is
an M-matrix in the convex regime, so [H^-1]_{0i} > 0 for every synapse; all
active synapses then share the sign of the true gradient (the sign of the
error), and per-synapse credit can only change magnitudes.
"""

import numpy as np

from tuft_gate import TuftTree, LEAVES, task, true_descent, _sig


class RealTime:
    def __init__(self, tree, dt=0.05, rate_max=1.0, rate_k=6.0, rate_theta=0.45,
                 ping_amp=2.0, ping_width=0.5, burst_extra=2, burst_gap=0.3,
                 burst_k=4.0, burst_theta=None, ctx_gain=3.0, ctx_delay=4.0,
                 t_early=20.0, t_late=20.0, ca_thr=5.0, ca_k=4.0, ca_noise=0.0,
                 bursts=True, pings=True, warmup=30.0):
        self.tree, self.dt = tree, dt
        self.rate_max, self.rate_k, self.rate_theta = rate_max, rate_k, rate_theta
        self.ping_amp, self.ping_width = ping_amp, ping_width
        self.burst_extra, self.burst_gap = burst_extra, burst_gap
        self.burst_k = burst_k
        self.burst_theta = burst_theta
        self.ctx_gain, self.ctx_delay = ctx_gain, ctx_delay
        self.t_early, self.t_late = t_early, t_late
        self.ca_thr, self.ca_k, self.ca_noise = ca_thr, ca_k, ca_noise
        self.bursts, self.pings = bursts, pings
        self.warmup = warmup

    def trial(self, x, y, rng, context=True):
        """One trial: settle, warm-up with pings, early window, context onset,
        delay, late window.
        Returns early/late calcium integrals per leaf, burst counts, and the
        free somatic voltage."""
        t = self.tree
        x = np.asarray(x, float)
        v = t.relax(x)
        v_free0 = v[0]
        if self.burst_theta is None:          # baseline burst probability 0.3 at the free tuft voltage
            self.burst_theta = v[t.tuft] + np.log(1 / 0.3 - 1) / self.burst_k
        ctx = np.zeros(t.n)
        n_w = int(round(self.warmup / self.dt))
        n_e = int(round(self.t_early / self.dt))
        n_d = int(round(self.ctx_delay / self.dt))
        n_l = int(round(self.t_late / self.dt))
        pulses = []                            # list of (start_step, end_step)
        ca_e = np.zeros(4); ca_l = np.zeros(4)
        b_e = b_l = 0
        for step in range(-n_w, n_e + n_d + n_l):
            # phase -1: warm-up with pings running (reach the ping-driven steady state)
            phase = -1 if step < 0 else (0 if step < n_e else (1 if step < n_e + n_d else 2))
            if phase >= 1 and context:
                ctx[t.tuft] = self.ctx_gain * (y - v_free0)
            # spontaneous ping
            if self.pings and rng.random() < self.dt * self.rate_max * _sig(self.rate_k * (v[0] - self.rate_theta)):
                pulses.append((step, step + int(round(self.ping_width / self.dt))))
                burst = self.bursts and rng.random() < _sig(self.burst_k * (v[t.tuft] - self.burst_theta))
                if burst:
                    for j in range(1, self.burst_extra + 1):
                        s0 = step + int(round(j * self.burst_gap / self.dt))
                        pulses.append((s0, s0 + int(round(self.ping_width / self.dt))))
                    if phase == 0:
                        b_e += 1
                    elif phase == 2:
                        b_l += 1
            inj = np.zeros(t.n)
            inj[0] = self.ping_amp * sum(1 for a, b in pulses if a <= step < b)
            pulses = [p for p in pulses if p[1] > step]
            v = v - self.dt * t.grad(v, x, ctx + inj)
            influx = x * _sig(self.ca_k * (v[list(LEAVES)] - self.ca_thr)) * self.dt
            if phase == 0:
                ca_e += influx
            elif phase == 2:
                ca_l += influx
        if self.ca_noise:
            ca_e = ca_e * (1 + self.ca_noise * rng.standard_normal(4))
            ca_l = ca_l * (1 + self.ca_noise * rng.standard_normal(4))
        return ca_e, ca_l, b_e, b_l, v_free0


def updates(rt, x, y, rng, rule, context=True):
    x = np.asarray(x, float)
    ca_e, ca_l, b_e, b_l, _ = rt.trial(x, y, rng, context)
    db = (b_l - b_e) / 5.0
    if rule == "ping-local":
        act = (x > 0) & (ca_e > 0) & (ca_l > 0)
        dw = np.zeros(4)
        dw[act] = np.log(ca_l[act] / ca_e[act]) / rt.ca_k
    elif rule == "three-factor":
        dw = x * (b_l - b_e) / 5.0
    elif rule == "exact":
        dw, db = true_descent(rt.tree, x, y)
    else:
        raise ValueError(rule)
    return dw, db


def jitter_set(rng, n_draws, lo=0.95, hi=1.05, y_hi=0.6, y_lo=0.3):
    xs, ys = task(y_hi, y_lo)
    out_x, out_y = [], []
    for _ in range(n_draws):
        for x, y in zip(xs, ys):
            out_x.append(x * rng.uniform(lo, hi, 4) * (x > 0))
            out_y.append(y)
    return out_x, out_y


def evaluate(tree, xs, ys):
    outs = np.array([tree.relax(x)[0] for x in xs])
    ys = np.array(ys)
    thr = 0.5 * (ys.max() + ys.min())
    acc = float(np.mean((outs > thr) == (ys > thr)))
    hi = outs[ys > thr]; lo = outs[ys <= thr]
    return acc, float(hi.min() - lo.max()), float(np.mean((outs - ys) ** 2) / 2)


def train(seed, rule, trunk=8, epochs=150, lr=0.03, shuffle=False, context=True,
          held_out_draws=5, **rt_kw):
    r = np.random.default_rng(100 + seed)
    tree = TuftTree(trunk=trunk, w=r.uniform(0.3, 1.2, 4), b=r.uniform(-1, 0))
    rt = RealTime(tree, **rt_kw)
    rng = np.random.default_rng(seed)
    p = np.r_[tree.w, tree.b]; m = np.zeros(5); s = np.zeros(5)
    for ep in range(epochs):
        xs, ys = jitter_set(rng, 1)
        if shuffle:
            ys = list(rng.permutation(ys))
        step = np.zeros(5)
        for i in rng.permutation(len(xs)):
            rt.burst_theta = None              # re-anchor the burst baseline to the current free state
            dw, db = updates(rt, xs[i], ys[i], rng, rule, context)
            step += np.r_[dw, db] / len(xs)
        m = 0.9 * m + 0.1 * step; s = 0.999 * s + 0.001 * step ** 2
        p = p + lr * (m / (1 - 0.9 ** (ep + 1))) / (np.sqrt(s / (1 - 0.999 ** (ep + 1))) + 1e-8)
        tree.w, tree.b = p[:4].copy(), float(p[4])
    test_x, test_y = jitter_set(np.random.default_rng(10_000 + seed), held_out_draws)
    return evaluate(tree, test_x, test_y), tree
