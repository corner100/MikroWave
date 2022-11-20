import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, data, restoration

def wiener_deconvolution(signal, kernel, lambd): # https://gist.github.com/danstowell/f2d81a897df9e23cc1da
    '''
    lambd is the SNR
    '''
    kernel_pad = np.zeros_like(signal)
    kernel_pad[kernel_pad.shape[1]//2-kernel.shape[1]//2:kernel_pad.shape[1]//2-kernel.shape[1]//2+kernel.shape[1],kernel_pad.shape[0]//2-kernel.shape[0]//2:kernel_pad.shape[0]//2-kernel.shape[0]//2+kernel.shape[0]] = kernel
    H = np.fft.fftn(kernel_pad)
    signal_fft = np.fft.fftn(signal)
    deconvolved = np.real(np.fft.ifftn(signal_fft * np.conj(H) / (H * np.conj(H) + lambd ** 2)))
    #deconvolved = np.abs(np.fft.ifftn(signal_fft * np.conj(H) / (H * np.conj(H) * signal_fft + lambd ** 2)))
    return deconvolved

def wiener_helstrom(signal, kernel, lambd):
    # https://www.strollswithmydog.com/deconvolution-by-division-in-the-frequency-domain/
    kernel_pad = np.zeros_like(signal)
    kernel_pad[kernel_pad.shape[1]//2-kernel.shape[1]//2:kernel_pad.shape[1]//2-kernel.shape[1]//2+kernel.shape[1],kernel_pad.shape[0]//2-kernel.shape[0]//2:kernel_pad.shape[0]//2-kernel.shape[0]//2+kernel.shape[0]] = kernel
    H = np.fft.fftn(kernel_pad)
    signal_fft = np.fft.fftn(signal)
    deconvolved = np.fft.fftshift(np.abs(np.fft.ifftn(signal_fft / ((H)*(1+1/(H * np.conj(H)*lambd ** 2))))))
    return deconvolved

def my_wiener(signal, kernel, lambd=0.2):
    SNR = 1 / lambd
    kernel_pad = np.zeros_like(signal)
    kernel_pad[kernel_pad.shape[1]//2-kernel.shape[1]//2:kernel_pad.shape[1]//2-kernel.shape[1]//2+kernel.shape[1],kernel_pad.shape[0]//2-kernel.shape[0]//2:kernel_pad.shape[0]//2-kernel.shape[0]//2+kernel.shape[0]] = kernel
    psf_fft = np.fft.fftn(kernel_pad)
    signal_fft = np.fft.fftn(signal)
    wiener = np.fft.fftshift(np.abs(np.fft.ifft2((psf_fft * np.conjugate(psf_fft) / (
    (psf_fft * np.conjugate(psf_fft) + 1 / SNR))) * signal_fft / psf_fft)))
    return wiener

cmap = "gray"

#gt = cv2.imread(r"C:\Users\veckstein\Downloads\RandS\starting-kit\src\python\data\gt\oct_60_40_0.png",0)
gt = cv2.imread(r"C:\Users\veckstein\Downloads\RandS\download_messung\sorted_img\gt\coin\20221119-141845-033_reco.png",0)
gt = cv2.normalize(gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#measurement = np.load(r"C:\Users\veckstein\Downloads\RandS\starting-kit\src\python\data\images\oct_60_40_0.npy",0)
measurement = cv2.imread(r"C:\Users\veckstein\Downloads\RandS\download_messung\sorted_img\measurement\coin\20221119-141845-033_reco.png",0)
measurement = cv2.normalize(measurement, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
from scipy.signal import convolve2d as conv2


psf_wiener_scipy, _ = restoration.unsupervised_wiener(measurement, gt)

lambd_est = 1e-3  # estimated noise lev
psf_wiener_wiki = wiener_deconvolution(measurement, gt, lambd_est)
psf_wiener_helstrom = wiener_helstrom(measurement, gt, lambd_est)

psf_wiener_my = my_wiener(measurement, gt, lambd=0.05)


plt.figure(), plt.imshow(gt, cmap = cmap), plt.title("img")
plt.figure(), plt.imshow(measurement, cmap = cmap), plt.title("img_blur")
plt.figure(), plt.imshow(psf_wiener_scipy, cmap = cmap), plt.title("psf_wiener scipy")
plt.figure(), plt.imshow(psf_wiener_wiki, cmap = cmap), plt.title("psf_wiener wiki")
plt.figure(), plt.imshow(psf_wiener_helstrom, cmap = cmap), plt.title("psf_wiener helstrom")
plt.figure(), plt.imshow(psf_wiener_my, cmap = cmap), plt.title("psf_wiener_my")

####
gt2 = cv2.imread(r"C:\Users\veckstein\Downloads\RandS\download_messung\sorted_img\measurement\test\20221119-143007-852_reco.png",0)
gt2=cv2.normalize(gt2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
measurement2 = cv2.imread(r"C:\Users\veckstein\Downloads\RandS\download_messung\sorted_img\measurement\test\20221119-143007-852_reco.png",0)
measurement2=cv2.normalize(measurement2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

est_wiener_scipy, _ = restoration.unsupervised_wiener(measurement2, psf_wiener_scipy)

lambd_est = 1e-3  # estimated noise lev
est_wiener_wiki = wiener_deconvolution(measurement2, psf_wiener_wiki, lambd_est)
est_wiener_helstrom = wiener_helstrom(measurement2, psf_wiener_helstrom, lambd_est)

est_wiener_my = my_wiener(measurement2, psf_wiener_my, lambd=0.05)


plt.figure(), plt.imshow(gt2, cmap = cmap), plt.title("img2")
plt.figure(), plt.imshow(measurement2, cmap = cmap), plt.title("img_blur2")
plt.figure(), plt.imshow(est_wiener_scipy, cmap = cmap), plt.title("deconvolved_wiener scipy")
plt.figure(), plt.imshow(est_wiener_wiki, cmap = cmap), plt.title("deconvolved_wiener wiki")
plt.figure(), plt.imshow(est_wiener_helstrom, cmap = cmap), plt.title("deconvolved_wiener helstrom")
plt.figure(), plt.imshow(est_wiener_my, cmap = cmap), plt.title("deconvolved_wiener_my")


print(123)