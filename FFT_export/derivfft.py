import numpy as np
import matplotlib.pyplot as plt

# clear all; close all; % clear all variables and figures
plt.close('all')

L = 20                      # define the computational domain [-L/2, L/2]
n = 128                     # define the number of Fourier modes 2^n

x2 = np.linspace(-L/2, L/2, n+1)  # define the domain discretization
x = x2[:n]                        # consider only the first n points: periodicity

# function to take a derivative of
u = 1 / np.cosh(x)                # sech(x) = 1/cosh(x)

# FFT the function
ut = np.fft.fft(u)

# k rescaled to 2pi domain
k = (2 * np.pi / L) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))

# first, second, third derivatives in Fourier space
ut1 = 1j * k * ut
ut2 = -k**2 * ut
ut3 = -1j * k**3 * ut

# inverse transforms
u1 = np.fft.ifft(ut1)
u2 = np.fft.ifft(ut2)
u3 = np.fft.ifft(ut3)

# analytic first derivative
u1exact = - (1 / np.cosh(x)) * np.tanh(x)

# plot results
plt.figure(1)
plt.plot(x, u, 'r', label='u')
plt.plot(x, u1.real, 'g', label='u1 (FFT)')
plt.plot(x, u1exact, 'go', label='u1 exact')
plt.plot(x, u2.real, 'b', label='u2 (FFT)')
plt.plot(x, u3.real, 'c', label='u3 (FFT)')
plt.legend()
plt.title('Spectral derivatives of sech(x)')
plt.show()