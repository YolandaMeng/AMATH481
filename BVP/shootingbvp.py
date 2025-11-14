import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- RHS function: y' = [ y2 ; (beta - n0) * y1 ] ---
def rhsfunc(t, y, n0, beta):
    y1, y2 = y
    f1 = y2
    f2 = (beta - n0) * y1
    return [f1, f2]

# --- main script ---
tol = 1e-4
n0 = 100
xp = [-1.0, 1.0]     # integration interval
A = 1.0
x0 = [0.0, A]        # initial condition [y1(0); y2(0)]

beta_start = n0

plt.figure()
for modes in range(1, 6):
    beta = beta_start
    dbeta = n0 / 100.0

    for j in range(1000):
        sol = solve_ivp(rhsfunc,xp, x0, method='RK45',args=[n0, beta],rtol=1e-6, atol=1e-6)
        t = sol.t
        y = sol.y.T  
        
        # stopping criterion: |y1(end) - 0| < tol
        if abs(y[-1, 0]) < tol:
            print(f"modes={modes}, beta ≈ {beta:.6f}")
            break

        # sign test: if (-1)^(modes+1) * y1(end) > 0, decrease beta
        if ((-1)**(modes + 1)) * y[-1, 0] > 0:
            beta = beta - dbeta
        else:
            beta = beta + dbeta/2;
            dbeta = dbeta/2;

    beta_start = beta - 0.1

    # plot y1(t)
    plt.plot(t, y[:, 0], label=f"mode {modes}")

plt.xlabel("t")
plt.ylabel("y1(t)")
plt.legend()
plt.tight_layout()
plt.show()