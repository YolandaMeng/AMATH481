import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

beta = 99.0

# --- initial mesh and initial guess 
xinit = np.linspace(-1.0, 1.0, 50)

y1_g = np.cos((np.pi / 2) * xinit)
y2_g = -(np.pi / 2) * np.sin((np.pi / 2) * xinit)
y_init = np.array([y1_g, y2_g]) 

def rhs_eigenvpb(x, y):
    f1 = y[1]
    f2 = (beta - 100.0) * y[0] - 10.0 * y[0]**3
    return [f1, f2]  

def bc(yl, yr):
    return [yl[0]-0.1, yr[0]]

# --- solve BVP ---
sol = solve_bvp(rhs_eigenvpb, bc, xinit, y_init)

# --- evaluate and plot ---
x = np.linspace(-1.0, 1.0, 100)
BS = sol.sol(x)  # BS[0] = y1(x), BS[1] = y2(x)

plt.plot(x, BS[0])
plt.xlabel("x")
plt.ylabel("y_1(x)")
plt.title("BVP solution (beta = 99)")
plt.grid(True)
plt.show()