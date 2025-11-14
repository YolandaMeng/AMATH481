import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import time

# -----------------------------
# FitzHugh–Nagumo ODE right-hand side
# -----------------------------
def fitzhugh(t, y, a, b, c, I):
    v, w = y
    dv = -v**3 + (1 + a)*v**2 - a*v - w + I
    dw = b*v - c*w
    return [dv, dw]
    
    
# -----------------------------
# Parameters (choose 'a' per regime comments)
# a = 0.1  # periodic
# a = 0.2  # subthreshold
# a = 0.3  # single spike
# -----------------------------
a = 0.2
b = 0.01
c = 0.01
I = 0.1

# Initial conditions and time grid
vinit, winit = 0.2, 0.0
T = 400.0
deltaT = 0.05
t_eval = np.arange(0.0, T, deltaT)  # matches [0:deltaT:T-deltaT]

# Solver tolerances
rtol = 1e-10
atol = np.array([1e-10, 1e-10])


start = time.time()

# Solve (ode15s analogue → use a stiff solver like BDF)
# sol = solve_ivp(
#     fun=lambda t, y: fitzhugh(t, y, a, b, c, I),
#     t_span=(float(t_eval[0]), float(t_eval[-1])),
#     y0=[vinit, winit],
#     method='BDF',
#     t_eval=t_eval,
#     rtol=rtol,
#     atol=atol
# )

sol = solve_ivp(
    fitzhugh,
    args = (a, b, c, I),
    t_span=(float(t_eval[0]), float(t_eval[-1])),
    y0=[vinit, winit],
    method='LSODA',
    t_eval=t_eval,
    rtol=rtol,
    atol=atol
)

t = sol.t
v = sol.y[0, :]
w = sol.y[1, :]

end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")




# -----------------------------
# Plot: w(t)
# -----------------------------
plt.figure(1)
plt.gca().tick_params(labelsize=18)
plt.box(True)
plt.plot(t, w)
plt.xlabel('t', fontsize=18)
plt.ylabel('w', fontsize=18)

# -----------------------------
# Plot: v(t)
# -----------------------------
plt.figure(2)
plt.gca().tick_params(labelsize=18)
plt.box(True)
plt.plot(t, v)
plt.xlabel('t', fontsize=18)
plt.ylabel('v', fontsize=18)

# -----------------------------
# Phase plane with nullclines
# -----------------------------
plt.figure(3)
plt.gca().tick_params(labelsize=18)

# Nullclines
vval = np.arange(-2, 2.0 + 0.01, 0.01)
wl1 = -vval**3 + (1 + a)*vval**2 - a*vval + I   # dv/dt = 0
wl2 = (b / c) * vval                            # dw/dt = 0

plt.plot(vval, wl1, 'r', label='dv/dt = 0')
plt.plot(vval, wl2, 'm', label='dw/dt = 0')
plt.axis([-0.4, 1.2, -0.1, 0.4])
plt.xlabel('v', fontsize=18)
plt.ylabel('w', fontsize=18)
plt.legend()


plt.plot(v, w, label='trajectory')

plt.show()