import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# ODE: y' = lambda * y
# ----------------------------
def ffunc1(t, y, lam):
    return lam * y

# ----------------------------
# parameters and grids
# ----------------------------
tstart = 0.0
tend   = 10.0
dt     = 0.3

# time vector (include endpoint [tstart:dt:tend])
# tspan = np.arange(tstart, tend + 1e-12, dt)  # small epsilon to include tend

# alternative implementation
num_points = int((tend - tstart) / dt) + 1
tspan = np.linspace(tstart, tend, num_points)

N = tspan.size

# solution array and IC
y = np.zeros(N)
y[0] = 1.0

lam = -10.0  # choose lambda

# ----------------------------
# Forward Euler iteration
# ----------------------------
for n in range(N - 1):
    y[n+1] = y[n] + dt * ffunc1(tspan[n], y[n], lam)

# ----------------------------
# Plot Euler approximation vs exact solution
# ----------------------------
plt.figure()
plt.gca().tick_params(labelsize=16)

plt.plot(tspan, y, '-o', label='Euler Approx')

tspanfine = np.linspace(tstart, tend, 10_000)
plt.plot(tspanfine, np.exp(lam * tspanfine), linewidth=3, label='Exact Solution')

plt.xlabel('t', fontsize=20)
plt.ylabel('y(t)', fontsize=20)
plt.legend()
plt.show()