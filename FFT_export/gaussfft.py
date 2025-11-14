import pdb
import numpy as np
import matplotlib.pyplot as plt

# clear all; close all;
plt.close('all')

L = 20                     # computational domain [-L/2, L/2]
n = 128                    # number of Fourier modes

x2 = np.linspace(-L/2, L/2, n+1)  # domain discretization
x = x2[:n]                        # only first n points (periodicity)

u = np.exp(-x**2)                 # function to take a derivative of

ut = np.fft.fft(u)                # FFT of u
pdb.set_trace()
utshift = np.fft.fftshift(ut)     # shift FFT

# plot initial Gaussian
plt.figure(1)
plt.plot(x, u)
plt.title('Initial Gaussian')

# plot unshifted transform
plt.figure(2)
plt.plot(np.abs(ut))
plt.title('Unshifted FFT magnitude')

# plot shifted transform
plt.figure(3)
plt.plot(np.abs(utshift))
plt.title('Shifted FFT magnitude')

plt.show()
