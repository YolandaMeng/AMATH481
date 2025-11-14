import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------
# Function definitions
# ------------------------------------------------

def ffunc1(t, y, lam):
    """Single ODE: dy/dt = λy"""
    return lam * y

def ffunc2(t, y, lam):
    """Vector-valued ODE system"""
    f1 = lam * y[0]
    f2 = y[1] - 0.5 * y[1]**3
    return [f1, f2]

# ------------------------------------------------
# Define parameters and initial conditions
# ------------------------------------------------
tstart = 0
tend = 10
dt = 0.3
tspan = np.arange(tstart, tend + dt, dt)

# alternative implementation with linspace
num_points = int((tend - tstart) / dt) + 1
tspan = np.linspace(tstart, tend, num_points)

lam = 10
y0_scalar = [1]
y0_vector = [1, 1]

# ------------------------------------------------
# Solve ODE system
# ------------------------------------------------
# Uncomment this for scalar ODE (ffunc1)
# sol = solve_ivp(lambda t, y: ffunc1(t, y, lam), [tstart, tend], y0_scalar,
                 # method='RK45', t_eval=tspan, rtol=1e-4, atol=1e-4)

# Solve vector ODE (ffunc2)
sol = solve_ivp(lambda t, y: ffunc2(t, y, lam), [tstart, tend], y0_vector,
                 method='RK45', t_eval=tspan, rtol=1e-4, atol=1e-4)

t = sol.t
y = sol.y.T  # transpose to shape (n_points, n_variables)

y1 = y[:, 0]
y2 = y[:, 1]

# ------------------------------------------------
# Plot results
# ------------------------------------------------
plt.figure()
plt.gca().tick_params(labelsize=16)

plt.plot(t, y1, '-o', label='ODE45 Approx')

# Analytical solution for ffunc1 (dy/dt = λy)
tspanfine = np.linspace(tstart, tend, 10000)
plt.plot(tspanfine, np.exp(lam * tspanfine), 'r', linewidth=3, label='Exact Solution')

plt.xlabel('t', fontsize=20)
plt.ylabel('y(t)', fontsize=20)
plt.legend()
plt.show()
