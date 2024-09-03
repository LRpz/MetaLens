import torch
import os
import sys
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
import h5py
from MetaLens.dl.utils import process_annotations, define_transforms, load_regressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_annotation_stats(annotations, metabolites):
    means = []
    stds = []
    for metabolite in metabolites:
        data = annotations[metabolite].values
        data = data[data != 0]
        means.append(data.mean())
        stds.append(data.std())
    return stds, means

def scale_cuda(patch):
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val)

def contrast_cuda(patch, min_percentile, max_percentile):
    min_val = torch.quantile(patch, min_percentile / 100.0)
    max_val = torch.quantile(patch, max_percentile / 100.0)
    return torch.clamp((patch - min_val) / (max_val - min_val), 0.0, 1.0)

def load_annotations(folder_path, split=True, test_size=0.3):
    annotations = pd.read_csv(os.path.join(folder_path, 'ion_intensities.csv'))
    df_normalized, metabolites, weights = process_annotations(annotations)    
    if split:
        train_annotations, val_annotations = train_test_split(annotations, test_size=test_size, random_state=42)
    else:
        val_annotations = annotations
    annot_stats = collect_annotation_stats(annotations, metabolites)
    return metabolites, val_annotations, annot_stats

def pred_images(test_image,model, am_test, annot_stats, metabolites, start_x=0, start_y=0, eval_range=500, step=1, batch_size=128):

    def process_patches(patches, model):

        with torch.no_grad():
            output = model(patches)

        # output_norm = output * std_tensor + mean_tensor
        # output_norm = torch.clamp(output_norm, min=0.4)
        output_norm = output * mask
        
        return output_norm
    
    if len(test_image.shape) == 2: test_image = test_image[..., None]
    test_image_tensor = torch.from_numpy(np.moveaxis(test_image, -1, 0)).unsqueeze(0).to(device)

    patches = torch.empty(0, in_chans, patch_size, patch_size, device='cuda')
    coords = []

    stds, means = annot_stats
    mean_tensor = torch.tensor(means).view(1, len(metabolites), 1, 1).to(device)
    std_tensor = torch.tensor(stds).view(1, len(metabolites), 1, 1).to(device)    
    
    am_image = torch.from_numpy(np.moveaxis(am_test, -1, 0)).to(device).unsqueeze(0)  # Convert to tensor and send to GPU
    mask = am_image[0, -1:, ...] > 0.5
    mask = mask.to(torch.float32)

    # NEG_am_image = torch.from_numpy(np.moveaxis(POS_am_test, -1, 0)).to(device).unsqueeze(0)  # Convert to tensor and send to GPU
    pred_image_r = torch.zeros((1, len(metabolites), eval_range+patch_size, eval_range+patch_size), device='cuda')  # Keep on GPU
    pred_image_counts = torch.zeros((eval_range+patch_size, eval_range+patch_size), device='cuda')  # Keep on GPU

    patches = torch.empty(0, in_chans, patch_size, patch_size, device='cuda')
    coords = []

    x_eval = np.arange(start_x, start_x+eval_range, step)
    y_eval = np.arange(start_y, start_y+eval_range, step)

    for i in tqdm.tqdm(x_eval):
        for j in y_eval:

            patch = test_image_tensor[:, :, i:i+patch_size, j:j+patch_size]

            if patch.shape[1] == 1:
                cell_patch = scale_cuda(contrast_cuda(patch, 0.1, 99.9))
            else:
                cell_patch = torch.stack([scale_cuda(contrast_cuda(patch[0, chan, ...], 0.1, 99.9)) for chan in range(patch.shape[1])], dim=0).unsqueeze(0)
            
            patch_data = torch.cat([cell_patch, am_image], dim=1)
            # patch_data = torch.from_numpy(np.moveaxis(transforms[0](image=np.moveaxis(patch_data.squeeze().cpu().numpy(), 0, -1))['image'], -1, 0)[None, ...]).to(device)

            patches = torch.cat([patches, patch_data], dim=0)
            coords.append((i, j))

            if patches.shape[0] == batch_size:
                
                output = process_patches(patches, model)
                   
                for k, (x, y) in enumerate(coords):
                    x, y = x-start_x, y-start_y
                    pred_image_r[..., x:x+patch_size, y:y+patch_size] += output[k]  # Keep on GPU
                    pred_image_counts[x:x+patch_size, y:y+patch_size] += patches[k, -1, ...] > 0.5 # Keep on GPU

                patches = torch.empty(0, in_chans, patch_size, patch_size, device='cuda')
                coords = []

    # Process remaining patches
    if patches.shape[0] > 0:

        output = process_patches(patches, model)

        for k, (x, y) in enumerate(coords):
            pred_image_r[..., x:x+patch_size, y:y+patch_size] += output[k]  # Keep on GPU
            pred_image_counts[x:x+patch_size, y:y+patch_size] += patches[k, -1, ...] > 0.5 # Keep on GPU

        patches = torch.empty(0, in_chans, patch_size, patch_size, device='cuda')
        coords = []

    # Move final image to CPU for visualization or saving
    pred_image_r_cpu = np.moveaxis(pred_image_r.cpu().numpy()[0], 0, -1)
    pred_image_counts_cpu = pred_image_counts.cpu().numpy()

    return pred_image_r_cpu, pred_image_counts_cpu

def save_data(data, h5_filepath):
    with h5py.File(h5_filepath, 'w') as file:
        for key, value in data.items():
            # Create a dataset for each item
            file.create_dataset(key, data=np.array(value))

plt.style.use('default')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: eval.py <dataset_name> <model_path>")
        sys.exit(1)
    
    sample_id = sys.argv[1]
    checkpoint_path = sys.argv[2]

    in_chans = 4
    patch_size = 128
    folder_path = rf'MetaLens\data\training_data'
    encoder = 'resnet152'
    metabolites, val_annotations, annot_stats = load_annotations(os.path.join(folder_path))

    test_image = tif.imread(rf"MetaLens\data\raw_data\{sample_id}_cells.tif")
    test_image[..., 2] = np.roll(test_image[..., 2], shift=3, axis=0)  # Shift down by 3 pixels
    test_image[:3, :, 2] = np.median(test_image[..., 2])  # Fill the top 3 rows with the median value
    cell_mask = tif.imread(rf"MetaLens\data\raw_data\{sample_id}_cells_mask.tif")
    am_prob = tif.imread(rf"MetaLens\data\raw_data\{sample_id}_ablation_marks_tf_pred.tif")

    am_test = tif.imread(r'MetaLens\data\am_eval.tif')[..., -1] # Only take the last channel (AM probablity map)

    model = load_regressor(checkpoint_path, len(metabolites), encoder=encoder, in_chans=in_chans)
    transforms = define_transforms(patch_size=patch_size)

    eval_range = 4000
    pred_image_r_cpu, pred_image_counts_cpu = pred_images(
        test_image=test_image,
        model=model, 
        am_test=am_test,
        start_x=0, 
        start_y=0, 
        eval_range=eval_range, 
        step=1, 
        annot_stats=annot_stats,
        metabolites=metabolites, 
        batch_size=128)

    data = {
        'pred': pred_image_r_cpu,  
        'metabolites': metabolites
    }

    h5_filepath = r"MetaLens\data\output\output.h5"
    save_data(data, h5_filepath)