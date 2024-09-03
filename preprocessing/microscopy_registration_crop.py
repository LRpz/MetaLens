import numpy as np
import tifffile as tif
from skimage import transform, io
from pystackreg import StackReg
import glob
import os
import tqdm
import sys

def scale(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def contrast(im, min_percentile, max_percentile):
    min_val = np.percentile(im, min_percentile)
    max_val = np.percentile(im, max_percentile)
    return np.clip(im, min_val, max_val)

def process_images(base_path, ds_name):
    im_pre_path = base_path + rf"\{ds_name}_preMALDI_channel1"
    im_post_path = base_path + rf"\{ds_name}_postMALDI_channel1"
    data_path = base_path + rf"\{ds_name}\transformedMarks.npy"
    params_path = base_path + rf"\{ds_name}\optimized_params.npy"

    im_pre = tif.imread(im_pre_path)
    im_post = tif.imread(im_post_path)
    data = np.load(data_path)
    tx, ty, angle = np.load(params_path)

    # Apply affine transformation
    affine_transform = transform.SimilarityTransform(translation=(ty, tx), rotation=-angle)
    im_post_tf = transform.warp(im_post, affine_transform.inverse) 

    # StackReg transformation
    scaling = 1 # Optional scaling factor
    crop_shape = np.min([im_pre.shape, im_post.shape], axis=0)
    ref = transform.resize(im_pre[:crop_shape[0], :crop_shape[1]], crop_shape // scaling, anti_aliasing=True)
    mov = transform.resize(im_post_tf[:crop_shape[0], :crop_shape[1]], crop_shape // scaling, anti_aliasing=True)

    sr = StackReg(StackReg.RIGID_BODY)
    out = sr.register_transform(ref, mov)

    # Calculate bounds for cropping
    window = 100
    min_y, min_x = np.min(data, axis=1) - window
    max_y, max_x = np.max(data, axis=1) + window
    min_x, min_y, max_x, max_y = [max(0, int(coord)) for coord in [min_x, min_y, max_x, max_y]]

    # Crop the images
    # crop_pre = ref[min_y:max_y, min_x:max_x]
    crop_post = scale(out[min_y:max_y, min_x:max_x])

    file_list = glob.glob(base_path + f"\{ds_name}_preMALDI_*")
    file_list = sorted(file_list)

    channels = [scale(tif.imread(file)[min_y:max_y, min_x:max_x]) for file in file_list]

    if not channels:
        raise FileNotFoundError("No suitable files found in the directory.")

    crop_pre = np.stack(channels, axis=0)

    return crop_pre, crop_post

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python microscopy_registration_crop.py <dataset_name>")
        sys.exit(1)
    
    ds_name = sys.argv[1]

    root = r'MetaLens\data\raw_data'
    
    crop_pre, crop_post = process_images(root, ds_name)

    # Save the processed images
    tif.imwrite(root + rf'{ds_name}_cells.tif', crop_pre.astype(np.float32))
    tif.imwrite(root + rf'{ds_name}_ablation_marks_tf.tif', crop_post.astype(np.float32))