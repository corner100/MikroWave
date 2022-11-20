import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

h,w = 128,128
N=5
filter_avg = np.ones((N,N),dtype=int)
for i in range(0,1000):
    gt = np.zeros((h,w))
    x = np.random.randint(w)
    y = np.random.randint(h)
    gt[y,x] = 1

    gt = signal.convolve2d(gt, filter_avg, boundary='symm', mode='same')
