import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    path = './data/gt'
    filter = cv.resize(cv.imread('./images/circle.png', 0), (21, 21)).astype(np.float32)
    filter = cv.copyMakeBorder(filter, 54, 54, 54, 54, cv.BORDER_CONSTANT, 0)
    for image_name in tqdm(os.listdir(path)):
        if not image_name.endswith('.png'):
            continue

        img = cv.imread(os.path.join(path, image_name), 0)
        # filter = cv.getGaussianKernel(129, -1)
        # filter /= filter[0, 0]
        # filter = filter @ filter.T
        # circle aperture
        try:
            fft_img = np.fft.fftshift(np.fft.fft2(img), axes=(0, 1))
            fft_img *= filter
            re_img = np.fft.ifft2(np.fft.ifftshift(fft_img))
            re_img = np.abs(re_img)
            re_img /= re_img.max()

            with open(os.path.join('./data/images', image_name.replace('.png', '.npy')), 'wb') as f:
                np.save(f, re_img)
                f.close()

        except Exception as e:
            print(e)
            print(image_name)

