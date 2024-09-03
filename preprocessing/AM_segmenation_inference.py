import glob
import numpy as np
import tifffile as tif
import torch
import segmentation_models_pytorch as smp
from monai.inferers import SlidingWindowInferer
import math
import sys

def adapt_input_conv(in_chans, conv_weight):
    """
    This function adapts the input channels of a convolutional layer's weights based on the number of input channels 
    provided. It handles cases where the input channels are 1 (grayscale), 3 (RGB), or other values. 
    The function ensures that the weight tensor is in the correct format for the given number of input channels and 
    adjusts the weights accordingly.

    Args:
    in_chans (int): The number of input channels.
    conv_weight (torch.Tensor): The convolutional layer's weights.

    Returns:
    torch.Tensor: The adapted convolutional layer's weights.
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight

def adapt_input_model(model):
    # Adapt first layer to take 1 channel as input - timm approach = sum weights
    new_weights = adapt_input_conv(in_chans=1, conv_weight=model.encoder.patch_embed1.proj.weight)
    model.encoder.patch_embed1.proj = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

    with torch.no_grad():
        model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
    
    return model

def scale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: AM_segmenation_inference.py <dataset_name>")
        sys.exit(1)
    
    ds_name = sys.argv[1]

    f = rf'MetaLens\data\raw_data\{ds_name}_ablation_marks_tf.tif'
        
    # Model trained on DHB and DAN
    model_path = r"MetaLens\models\AM_segmentation.pth"

    device='cuda'

    model = smp.Unet(encoder_name='mit_b5', classes=1, in_channels=3, encoder_weights=None)
    model = adapt_input_model(model) # Adapt model to accept 1 channel input
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.to(device)

    inferer = SlidingWindowInferer(
        roi_size=(1024, 1024), 
        sw_batch_size=1, 
        progress=True, 
        mode="gaussian",
        overlap=0.5,
        device='cpu', # Stithcing is done on CPU
        sw_device=device # Patch inference is done on GPU
        )

    im_np = scale(tif.imread(f))
    im_tensor = torch.tensor(im_np[None, None, ...]).float().to(device)

    with torch.no_grad():
        pred_tensor = inferer(inputs=im_tensor, network=model)
        pred_tensor = torch.sigmoid(pred_tensor)

    pred = pred_tensor.squeeze().cpu().numpy()
    tif.imwrite(f.replace('.tif', '_pred.tif'), pred.astype(np.float32))