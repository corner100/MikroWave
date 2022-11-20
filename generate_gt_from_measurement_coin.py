import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import nibabel
from scipy import ndimage

def import_volume(file_path):
    """Import 3D volumetric data from file.

    Args:
        file_path (basestring): Absolute path for .img, .hdr or .json file.

    Returns:
        The volume definition given as the coordinate vectors in x, y, and z-direction.
    """

    path, filename = os.path.split(file_path)
    filename = os.path.splitext(filename)[0]

    file_path_img = os.path.join(path, f"{filename}.img")
    file_path_hdr = os.path.join(path, f"{filename}.hdr")
    file_path_json = os.path.join(path, f"{filename}.json")

    if not os.path.exists(file_path_img):
        raise Exception(f"Does not exist file: {file_path_img}")
    if not os.path.exists(file_path_hdr):
        raise Exception(f"Does not exist file: {file_path_hdr}")

    v_mag_phase = nibabel.load(file_path_hdr)
    _volume = v_mag_phase.dataobj[:, :, : v_mag_phase.shape[2] // 2] * np.exp(
        1j * v_mag_phase.dataobj[:, :, v_mag_phase.shape[2] // 2 :]
    )
    if len(_volume.shape) > 3:
        _volume = np.squeeze(_volume)

    return _volume


dir_path = r"C:\Users\veckstein\Downloads\RandS\download_messung\team-12"
paths = glob.glob(os.path.join(dir_path,"*\*.img"))
dest_dir_path = r"C:\Users\veckstein\Downloads\RandS\download_messung\sorted\gt\d1_4ecken"
form_type = "square_rot" # rect_rot # circle d12 # oct_rot # five_corner d12
rect_img = cv2.imread(r"images/rect.png",0)
oct_img_temp = cv2.imread(r"images/oct.png",0)
oct_img = np.zeros((385,385))
oct_img[25:-26,:]=oct_img_temp
d = 12 # 10

rect_img = ndimage.rotate(rect_img, 45)
rect_img = cv2.resize(rect_img,dsize=(2*d+1,2*d+1))
oct_img = ndimage.rotate(oct_img, -45)
oct_img = cv2.resize(oct_img,dsize=(2*d+1,2*d+1),interpolation=cv2.INTER_AREA)

five_corner = np.zeros((d*4+1,d*4+1))
contours = np.array( [ [d,d], [d,d+d*2], [d+d*2,d+d*2], [d+d*2,d], [d+d,d+d] ] )
cv2.fillPoly(five_corner, pts =[contours], color=255)
five_corner = ndimage.rotate(five_corner, 135, reshape=False)# -135

for path in paths:
    measurement = import_volume(path)
    measurement = np.abs(measurement)[:,:,0]
    measurement = measurement/np.max(measurement)
    (T, thresh) = cv2.threshold(measurement, thresh=0.2, maxval=1.0,type=cv2.THRESH_BINARY)
    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    gt = np.zeros(measurement.shape)
    if form_type == "circle":
        cv2.circle(gt, (cX, cY), 12, 255, -1)
    elif form_type == "square_rot":
        #d = 10
        #box = cv2.boxPoints([[cX, cY-d],[cX-d, cY],[cX, cY+d],[cX+d, cY]])
        #box = cv2.boxPoints(thresh)
        #cv2.rectangle(gt, (cX-d, cY-d), (cX, cY+d), 255,-1)
        #cv2.drawContours(gt, [box], 0, 255, -1)

        gt[cY-d:cY+d+1, cX-d:cX+d+1] = rect_img
    elif form_type == "oct_rot":
        #d = 10
        #box = cv2.boxPoints([[cX, cY-d],[cX-d, cY],[cX, cY+d],[cX+d, cY]])
        #box = cv2.boxPoints(thresh)
        #cv2.rectangle(gt, (cX-d, cY-d), (cX, cY+d), 255,-1)
        #cv2.drawContours(gt, [box], 0, 255, -1)

        gt[cY-d:cY+d+1, cX-d:cX+d+1] = oct_img
    elif form_type == "five_corner":
        #d = 10
        #box = cv2.boxPoints([[cX, cY-d],[cX-d, cY],[cX, cY+d],[cX+d, cY]])
        #box = cv2.boxPoints(thresh)
        #cv2.rectangle(gt, (cX-d, cY-d), (cX, cY+d), 255,-1)
        #cv2.drawContours(gt, [box], 0, 255, -1)

        gt[cY-d*2:cY+d*2+1, cX-d*2:cX+2*d+1] = five_corner
    dest_path = os.path.join(dest_dir_path,os.path.basename(path).replace(".img", ".png"))
    plt.imsave(fname=dest_path,arr=gt)
    dest_path = os.path.join(dest_dir_path,os.path.basename(path).replace(".img", "_thresh.png"))
    plt.imsave(fname=dest_path,arr=thresh)
    dest_path = os.path.join(dest_dir_path,os.path.basename(path).replace(".img", "_diff.png"))
    plt.imsave(dest_path,np.square(gt-measurement*255))
    dest_path = os.path.join(dest_dir_path,os.path.basename(path).replace(".img", "_measurement.png"))
    plt.imsave(dest_path,measurement)
print(123)
# plt.figure(), plt.imshow(measurement)
# plt.figure(), plt.imshow(thresh)
# plt.figure(), plt.imshow(gt)
# plt.figure(), plt.imshow(np.square(gt-measurement*255))