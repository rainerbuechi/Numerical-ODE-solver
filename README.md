# ODE Solvers (Euler + RK4)

Small numerical ODE solver implementations (written for learning + physics simulations).

## Methods
- Explicit Euler
- Runge–Kutta 4 (RK4)

## Example: Harmonic Oscillator
We solve the system

$$
x' = v, \quad v' = -\omega^2 x
$$

and compare Euler vs RK4 to the exact solution  ($$x(t)=\cos(\omega t)$$).

## Run
```bash
python examples.py
