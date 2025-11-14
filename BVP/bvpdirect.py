import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 1.0  # Length of the bar
T0 = 100.0  # Temperature at x=0
TL = 0.0  # Temperature at x=L
N = 50  # Number of interior points (N+2 total points including boundaries)
h = L / (N + 1)  # Step size

# dissipation
#d = 0
d = 5

# Construct the tridiagonal matrix
A = np.zeros((N+2, N+2))
b = np.zeros(N+2)


for i in range(1,N+1):
    A[i, i-1] = 1
    A[i, i] = -2 + d*h**2
    A[i, i+1] = 1

# Apply boundary conditions
A[0,0]=1
A[-1,-1]=1

b[0] = T0
b[-1] = TL

# Solve the system of linear equations
u = np.linalg.solve(A, b)

# Combine with boundary conditions to get the full solution
x = np.linspace(0, L, N + 2)

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(x, u, marker='o', linestyle='-', label='Numerical Solution')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (u)')
plt.title('Steady-State Temperature Distribution in a Bar')
plt.grid(True)
plt.legend()
plt.show()