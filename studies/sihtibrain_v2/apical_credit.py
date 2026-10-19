"""Gate 4: apical-tuft context changes a basal synapse via a returning ping.

Nine reciprocal cable compartments: 7 basal/soma, trunk 7, tuft 8.
The external target-minus-prediction is applied ONLY as a tuft current.
Plastic branch-readout weights see eligibility and two local calcium values,
never the target. A hand-designed nonlinear branch sieve is held fixed.
This is supervised *later-context* learning, not teacher-free discovery.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import numpy as np
from sihtibrain_v2 import operator, currents, sigmoid

PAIRS = tuple(combinations(range(4), 2))
LABELS = np.array([float(p in ((0, 1), (2, 3))) for p in PAIRS])


def nine_compartment_operator(trunk=.12, tuft=.12):
    h = np.zeros((9, 9))
    h[:7, :7] = operator()
    h[7, 7] = h[8, 8] = .15
    for u, v, g in ((0, 7, trunk), (7, 8, tuft)):
        h[u, u] += g
        h[v, v] += g
        h[u, v] -= g
        h[v, u] -= g
    return h


@lru_cache(maxsize=32)
def tuft_to_soma(trunk=.12, tuft=.12):
    source = np.zeros(9)
    source[8] = 1.
    return float(np.linalg.solve(nine_compartment_operator(trunk, tuft), source)[0])


def basal_state(x, trunk=.12, tuft=.12):
    drive = np.zeros(9)
    drive[:7] = currents(x)
    return np.linalg.solve(nine_compartment_operator(trunk, tuft), drive)


def basal_features(x, threshold=2., steepness=10., trunk=.12, tuft=.12):
    """Preselected nonlinear basal feature; the feature extractor does not learn."""
    v = basal_state(x, trunk, tuft)
    return sigmoid(steepness * (v[[1, 4]] - threshold))


def calcium(ping, local_v, active, attenuation=1., k=1.6):
    """Abstract positive pulse-driven calcium; NOT NMDA channel kinetics."""
    return float(active * np.exp(.30 * local_v) *
                 np.expm1(k * attenuation * ping))


@dataclass
class Gate4:
    weights: np.ndarray
    bias: float
    trunk: float = .12
    tuft: float = .12
    context_gain: float = 3.

    @classmethod
    def new(cls, seed=0, trunk=.12, tuft=.12):
        rng = np.random.default_rng(seed)
        return cls(weights=rng.normal(0, .05, 2), bias=-.20,
                   trunk=trunk, tuft=tuft)

    def output(self, x, linear=False):
        h = (basal_state(x, self.trunk, self.tuft)[[1, 4]]
             if linear else basal_features(x, trunk=self.trunk, tuft=self.tuft))
        return float(sigmoid(self.bias + self.weights @ h))

    def window(self, x, target, *, context=True, ping=True,
               shuffle_target=None, noise=0., rng=None,
               delay=0, max_delay=12, linear=False):
        """Two abstract windows. Delay is a hard eligibility cutoff.

        No target is passed into the local calcium/update calculation.
        Basal voltages are computed with zero teacher current, from the
        identical nine-node cable in both phases.
        """
        v = basal_state(x, self.trunk, self.tuft)
        phi = (v[[1, 4]] if linear else basal_features(
            x, trunk=self.trunk, tuft=self.tuft))
        predicted = self.output(x, linear=linear)
        later_target = target if shuffle_target is None else shuffle_target
        tuft_current = float(later_target - predicted) if context else 0.
        transfer = tuft_to_soma(self.trunk, self.tuft)
        soma_modulation = self.context_gain * transfer * tuft_current
        p_free = 1.
        p_later = (1. + .85 * np.tanh(4. * soma_modulation)) if ping else p_free
        if not ping:
            p_free = p_later = 0.
        eligibility = phi if delay <= max_delay else np.zeros_like(phi)
        if rng is None:
            rng = np.random.default_rng(0)
        gains = (.65, 1.10)
        ca0 = np.array([calcium(p_free, v[root], 1., att)
                        for root, att in zip((1, 4), gains)])
        ca1 = np.array([calcium(p_later, v[root], 1., att)
                        for root, att in zip((1, 4), gains)])
        if noise:
            ca0 *= np.maximum(.01, 1. + rng.normal(0, noise, 2))
            ca1 *= np.maximum(.01, 1. + rng.normal(0, noise, 2))
        local_delta = np.zeros(2) if not ping else np.log(ca1 / ca0)
        return dict(predict=predicted, target=float(target),
                    tuft_current=tuft_current,
                    passive_soma_effect=transfer * tuft_current,
                    ping_free=p_free, ping_later=p_later,
                    calcium_free=ca0, calcium_later=ca1,
                    local_delta=local_delta, eligibility=eligibility,
                    updates=eligibility * local_delta,
                    bias_update=float(np.mean(local_delta)),
                    apical_transfer=transfer)


def samples(seed, n_per_pattern=32, jitter=.08):
    rng = np.random.default_rng(seed)
    data = []
    for pair, target in zip(PAIRS, LABELS):
        for _ in range(n_per_pattern):
            x = np.zeros(4)
            x[list(pair)] = rng.uniform(1-jitter, 1+jitter, size=2)
            data.append((x, float(target)))
    rng.shuffle(data)
    return data


def evaluate(model, data, linear=False):
    values = np.array([model.output(x, linear=linear) for x, _ in data])
    t = np.array([target for _, target in data])
    return _metrics(values, t)


def _metrics(y, t):
    return dict(accuracy=float(np.mean((y >= .5) == t)),
                bce=float(-np.mean(t*np.log(np.clip(y, 1e-9, 1)) +
                                (1-t)*np.log(np.clip(1-y, 1e-9, 1)))),
                positive_mean=float(y[t == 1].mean()),
                negative_mean=float(y[t == 0].mean()),
                margin=float(y[t == 1].min() - y[t == 0].max()))


def _cached_data(data, linear=False, trunk=.12, tuft=.12):
    rows = []
    for x, target in data:
        v = basal_state(x, trunk, tuft)
        phi = (v[[1, 4]] if linear else
               sigmoid(10. * (v[[1, 4]] - 2.)))
        rows.append((phi, float(target)))
    return rows


def _evaluate_cached(model, data):
    phi = np.array([z for z, _ in data])
    t = np.array([y for _, y in data])
    return _metrics(sigmoid(phi @ model.weights + model.bias), t)


def train(seed=0, mode="tuft_ping", epochs=50, learning_rate=.11,
          train_data=None, test_data=None, noise=0., delay=0,
          linear=False, trunk=.12, context_gain=3.):
    """Precompute fixed cable features only; update on each example online."""
    train_data = (samples(120+seed, n_per_pattern=12)
                  if train_data is None else train_data)
    test_data = (samples(880+seed, n_per_pattern=32)
                 if test_data is None else test_data)
    cached_train = _cached_data(train_data, linear=linear, trunk=trunk)
    cached_test = _cached_data(test_data, linear=linear, trunk=trunk)
    m = Gate4.new(seed, trunk=trunk)
    m.context_gain = float(context_gain)
    rng = np.random.default_rng(405+seed)
    history = []
    transfer = tuft_to_soma(trunk, .12)
    attenuation = np.array([.65, 1.10])
    for epoch in range(epochs):
        shuffled = (rng.permutation(len(cached_train))
                    if mode == "shuffled_context" else None)
        order = rng.permutation(len(cached_train))
        for idx in order:
            phi, target = cached_train[idx]
            prediction = float(sigmoid(m.bias + m.weights @ phi))
            later_target = (cached_train[shuffled[idx]][1]
                            if shuffled is not None else target)
            tuft_current = ((later_target - prediction)
                            if mode != "no_context" else 0.)
            soma_modulation = m.context_gain * transfer * tuft_current
            p0 = 1.
            p1 = 1. + .85 * np.tanh(4. * soma_modulation)
            if mode == "no_ping":
                ratios = np.zeros(2)
            else:
                ratios = np.log(np.expm1(1.6*attenuation*p1) /
                                np.expm1(1.6*attenuation*p0))
                if noise:
                    noise0 = np.maximum(.01, 1.+rng.normal(0, noise, 2))
                    noise1 = np.maximum(.01, 1.+rng.normal(0, noise, 2))
                    ratios += np.log(noise1/noise0)
            if mode == "direct_gradient":
                # Supervised direct-output comparator; NOT a ping.
                delta = phi * (target - prediction)
                db = target - prediction
            else:
                delta = (np.zeros(2) if mode == "no_basal_trace" or delay > 12
                         else phi * ratios)
                db = float(np.mean(ratios)) if delay <= 12 else 0.
            m.weights += learning_rate * delta
            m.bias += learning_rate * db
        if epoch in (0, 1, 4, 9, 19, epochs-1):
            history.append(dict(epoch=epoch+1, **_evaluate_cached(m, cached_test)))
    return m, _evaluate_cached(m, cached_test), history


def benchmark(seeds=(0, 1, 2, 3), epochs=50):
    results = {}
    options = {
        "tuft_ping": {}, "no_ping": {}, "no_context": {},
        "shuffled_context": {}, "no_basal_trace": {}, "direct_gradient": {},
        "long_delay": {"delay": 25}, "linear_basal": {"linear": True},
        "no_trunk": {"trunk": 0.}
    }
    for name, kw in options.items():
        readings = []
        for seed in seeds:
            mode = ("tuft_ping" if name in ("long_delay", "linear_basal", "no_trunk")
                    else name)
            m, e, h = train(seed, mode=mode, epochs=epochs, **kw)
            readings.append(dict(seed=seed, **e, weights=m.weights.tolist(),
                                 bias=m.bias, history=h))
        results[name] = readings
    return results
