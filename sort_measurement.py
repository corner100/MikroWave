import numpy as np
import os
import glob
from shutil import copyfile
import nibabel
import cv2

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
dir_path = r"C:\Users\veckstein\Downloads\RandS\messung2\rest"
paths = glob.glob(os.path.join(dir_path,"*"))
dest_path_dir = r"C:\Users\veckstein\Downloads\RandS\messung2\sorted\measurement\rest"

for path in paths:
    base = os.path.basename(path)
    source = os.path.join(path,base+"_reco.img")
    dest = os.path.join(dest_path_dir,base+"_reco.img")
    copyfile(source,dest)
    source = os.path.join(path,base+"_reco.hdr")
    dest = os.path.join(dest_path_dir,base+"_reco.hdr")
    copyfile(source,dest)


    measurement = import_volume(source)
    measurement = np.abs(measurement)[:,:,0]
    measurement = measurement/np.max(measurement)*255
    dest_path_measurement = os.path.join(dest_path_dir,base+"_reco.png")
    cv2.imwrite(dest_path_measurement, measurement)

print(123)