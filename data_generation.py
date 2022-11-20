import numpy as np
import cv2 as cv
import imutils


if __name__ == '__main__':
    img_size = 129
    mask_size = 20

    circle = cv.resize(cv.imread('./images/circle.png', 0), (mask_size, mask_size))
    oct = cv.resize(cv.imread('./images/oct.png', 0), (mask_size, mask_size))
    rect = cv.resize(cv.imread('./images/rect.png', 0), (mask_size, mask_size))
    bg = np.zeros((img_size, img_size), dtype=np.uint8)

    names = ['circle', 'oct', 'rect']

    for idx, mask in enumerate([circle, oct, rect]):
        for x in range(0, img_size - 1 - mask_size, mask_size):
            for y in range(0, img_size - 1 - mask_size, mask_size):
                if names[idx] == 'circle':
                    img = bg.copy()
                    img[x:x + mask_size, y:y + mask_size] = mask
                    cv.imwrite(f'./data/gt/{names[idx]}_{x}_{y}_0.png', img)
                else:
                    for angle in range(0, 360, 30):
                        # M = cv.getRotationMatrix2D((mask_size // 2, mask_size // 2), angle, 1.0)
                        # rotated = cv.warpAffine(mask, M, (mask_size, mask_size))
                        rotated = imutils.rotate_bound(mask, angle)
                        img = bg.copy()
                        img[x:x + rotated.shape[0], y:y + rotated.shape[1]] = rotated
                        cv.imwrite(f'./data/gt/{names[idx]}_{x}_{y}_{angle}.png', img)
