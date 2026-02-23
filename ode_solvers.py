# ode_solvers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np

Array = np.ndarray
RHS = Callable[[float, Array], Array]  # f(t, y)


@dataclass
class Solution:
    t: Array        # shape (N,)
    y: Array        # shape (N, dim)


def euler_step(f: RHS, t: float, y: Array, h: float) -> Array:
    return y + h * f(t, y)


def rk4_step(f: RHS, t: float, y: Array, h: float) -> Array:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: RHS,
    t_span: Tuple[float, float],
    y0: Array,
    h: float,
    method: str = "rk4",
) -> Solution:
    t0, t1 = t_span
    if h <= 0:
        raise ValueError("Step size h must be > 0.")
    if t1 <= t0:
        raise ValueError("t_span must satisfy t1 > t0.")

    stepper = {"euler": euler_step, "rk4": rk4_step}.get(method.lower())
    if stepper is None:
        raise ValueError("method must be 'euler' or 'rk4'.")

    y0 = np.array(y0, dtype=float)
    if y0.ndim == 0:
        y0 = y0.reshape(1)

    n_steps = int(np.ceil((t1 - t0) / h))
    t = np.empty(n_steps + 1, dtype=float)
    y = np.empty((n_steps + 1, y0.size), dtype=float)

    t[0] = t0
    y[0] = y0

    ti = t0
    yi = y0
    for i in range(n_steps):
        # last step might be shorter so we land exactly on t1
        hi = min(h, t1 - ti)
        yi = stepper(f, ti, yi, hi)
        ti = ti + hi
        t[i + 1] = ti
        y[i + 1] = yi

    return Solution(t=t, y=y)
