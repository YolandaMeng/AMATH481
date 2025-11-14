import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def visualize_fft(F, log_scale=True, normalize=True, show=True):

    mag = np.abs(F)
    
    if log_scale:
        mag = np.log1p(mag)  # log compression

    if normalize:
        mag = 255 * (mag / (mag.max() + 1e-12))
        
    img_uint8 = np.clip(mag, 0, 255).astype(np.uint8)

    if show:
        plt.figure()
        plt.imshow(img_uint8, cmap='gray')
        plt.axis('off')
        plt.title('Fourier Magnitude (grayscale)')
        plt.show()

    return img_uint8

I = io.imread('2noise.jpg', as_gray=True)
if I.dtype != np.uint8:
    I = (I * 255).astype(np.uint8)

H, W = I.shape
I_out = np.empty_like(I)

F = np.fft.fft2(I.astype(float))

visualize_fft(F, log_scale=True)

F = np.fft.fftshift(F)
    
visualize_fft(F, log_scale=True)

keep = 0.15
hh = int(H * keep / 2)
wh = int(W * keep / 2)
mask = np.zeros((H, W), dtype=float)
mask[H//2 - hh : H//2 + hh, W//2 - wh : W//2 + wh] = 1.0  # center box (fftshifted)

plt.figure(); plt.imshow(mask,cmap='gray'); plt.axis('off'); plt.title('F')

F *= mask

visualize_fft(F, log_scale=True)

F = np.fft.ifftshift(F)
img = np.fft.ifft2(F)

I_out = np.clip(np.abs(img), 0, 255).astype(np.uint8)

plt.figure(); plt.imshow(I); plt.axis('off'); plt.title('Original/Noisy')
plt.figure(); plt.imshow(I_out); plt.axis('off'); plt.title('Low-pass (central 10%)')
plt.show()

