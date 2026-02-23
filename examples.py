# examples.py
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import integrate

def harmonic_oscillator(omega: float):
    # y = [x, v], y' = [v, -omega^2 x]
    def f(t, y):
        x, v = y
        return np.array([v, -(omega**2) * x], dtype=float)
    return f

def main():
    omega = 2.0
    y0 = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t_span = (0.0, 10.0)

    sol_euler = integrate(harmonic_oscillator(omega), t_span, y0, h=0.01, method="euler")
    sol_rk4   = integrate(harmonic_oscillator(omega), t_span, y0, h=0.01, method="rk4")

    # exact solution: x(t) = cos(omega t), v(t) = -omega sin(omega t)
    t = sol_rk4.t
    x_exact = np.cos(omega * t)

    plt.figure()
    plt.plot(sol_euler.t, sol_euler.y[:, 0], label="Euler: x(t)")
    plt.plot(sol_rk4.t,   sol_rk4.y[:, 0], label="RK4: x(t)")
    plt.plot(t, x_exact, label="Exact: cos(ωt)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.title("Harmonic Oscillator")
    plt.show()

if __name__ == "__main__":
    main()
