"""Dimensionless passive cable tree with an optional abstract branch subunit.

Nodes: soma 0; branch roots 1, 4; distal leaves (2, 3), (5, 6).
The nonlinear term represents input coincidence, not an NMDA channel model.
"""

import numpy as np


class Circuit:
    EDGES = ((0, 1, 0.8), (0, 4, 0.8),
             (1, 2, 0.5), (1, 3, 0.5),
             (4, 5, 0.5), (4, 6, 0.5))

    def __init__(self, leak=0.15):
        if not np.isfinite(leak) or leak <= 0:
            raise ValueError("leak must be finite and positive")
        self.leak = float(leak)
        self.laplacian = np.zeros((7, 7), dtype=float)
        for i, j, conductance in self.EDGES:
            self.laplacian[i, i] += conductance
            self.laplacian[j, j] += conductance
            self.laplacian[i, j] -= conductance
            self.laplacian[j, i] -= conductance

    @staticmethod
    def _stimulus(stimulus):
        current = np.asarray(stimulus, dtype=float)
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError("stimulus must be a finite vector of seven currents")
        return current

    def run(self, stimulus, *, coincidence_gain=0., steps=700, dt=0.02,
            soma_pulse=None):
        current = self._stimulus(stimulus).copy()
        if (not np.isfinite(coincidence_gain) or not np.isfinite(dt)
                or coincidence_gain < 0 or steps < 1 or dt <= 0):
            raise ValueError("gain and dt must be finite; gain >= 0, steps >= 1, dt > 0")
        if dt * (max(np.linalg.eigvalsh(self.laplacian)) + self.leak) >= 2:
            raise ValueError("dt is too large for explicit Euler stability")
        if soma_pulse is not None:
            start, stop, amplitude = soma_pulse
            if not (0 <= start < stop <= steps and np.isfinite(amplitude)):
                raise ValueError("soma_pulse must be (start, stop, finite amplitude)")
        current[1] += coincidence_gain * current[2] * current[3]
        current[4] += coincidence_gain * current[5] * current[6]
        trace = np.zeros((steps + 1, 7), dtype=float)
        for t in range(steps):
            drive = current.copy()
            if soma_pulse is not None and start <= t < stop:
                drive[0] += amplitude
            trace[t + 1] = trace[t] + dt * (
                drive - self.laplacian @ trace[t] - self.leak * trace[t]
            )
        return trace

    def energy(self, voltage, stimulus):
        voltage = self._stimulus(voltage)
        current = self._stimulus(stimulus)
        edge_energy = sum(0.5 * g * (voltage[i] - voltage[j]) ** 2
                          for i, j, g in self.EDGES)
        return float(edge_energy + self.leak / 2 * np.dot(voltage, voltage)
                     - np.dot(current, voltage))

    def axial_current(self, voltage, source, target):
        voltage = self._stimulus(voltage)
        for i, j, conductance in self.EDGES:
            if {source, target} == {i, j}:
                return float(conductance * (voltage[source] - voltage[target]))
        raise ValueError("source and target must be adjacent compartments")

    @staticmethod
    def threshold_crossings(soma_trace, threshold):
        values = np.asarray(soma_trace, dtype=float)
        return int(np.count_nonzero((values[:-1] <= threshold)
                                    & (values[1:] > threshold)))
