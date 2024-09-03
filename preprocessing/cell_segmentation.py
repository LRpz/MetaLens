import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread
from skimage import filters
import glob, tqdm
import tifffile as tif
import sys

def scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def contrast(arr, low, high):
    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, high))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cell_segmentation.py <dataset_name>")
        sys.exit(1)
    
    ds_name = sys.argv[1]

    f = rf'MetaLens\data\raw_data\{ds_name}_cells.tif'
    imgs = [imread(f)]
    channels = [[2,3]]
    model = models.Cellpose(model_type='cyto2', gpu=True)
    masks, flows, styles, diams = model.eval(
        imgs, 
        diameter=50, 
        channels=channels,
        flow_threshold=1)

    tif.imwrite(f.replace('cells', 'cells_mask'), masks[0].astype(np.uint16))
