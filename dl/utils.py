import os
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.spatial.distance import cdist
from skimage import measure, morphology
from sklearn.model_selection import train_test_split
from tifffile import imread
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from scipy.stats import zscore
from torchmetrics import PearsonCorrCoef, R2Score, Accuracy, F1Score
import math 

def define_transforms(patch_size=128):

    hard_transform = A.Compose([
        A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(1, 1), p=1)
        ])

    transform = A.Compose([
        # A.RandomResizedCrop(height=224, width=224, scale=(1, 1), p=1),
        A.Affine(scale=[0.75, 1], rotate=[-45, 45], p=0.5, mode=cv2.BORDER_REFLECT), #scale=[0.25, 1.5]
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        # ToTensorV2()  # Converts the image back to PyTorch format
        ])

    post_transform = A.Compose([ 
        ToTensorV2()
        ])

    return [hard_transform, transform, post_transform]

def process_annotations(annotations, lod_threshold=0.0, remove_zeros_samples=True):
    # Load dataset and split into train and validation sets
    metabolites = np.array(annotations.drop('filename', axis=1).columns)

    # Assuming `df` is your DataFrame
    df_normalized = annotations.copy()
    for col in metabolites:  # Assuming the last column is 'filename' or a non-feature column
        if annotations[col].dtype in ['float64', 'int64']:
            df_normalized.loc[:, col] = -10
            df_normalized.loc[annotations[col] > lod_threshold, col] = zscore(annotations.loc[annotations[col] > lod_threshold, col])

    # Remove rows filled with 0
    if remove_zeros_samples: df_normalized = df_normalized.loc[df_normalized[metabolites].sum(axis=1) != -10*len(metabolites)]

    # Compute the frequency of zero values for each column
    label_frequencies = df_normalized.drop('filename', axis=1).apply(lambda x: (x != -10).mean(), axis=0).values

    inv_freq = 1 - label_frequencies
    weights = (inv_freq - inv_freq.min()) / (inv_freq.max() - inv_freq.min())

    return df_normalized, metabolites, weights

class ImageDataset(Dataset):
    def __init__(self, annotations, root_dir, metabolites, hard_transform=None, transform=None, post_transform=None, task='classification'):
        self.annotations = annotations
        self.root_dir = root_dir
        self.hard_transform = hard_transform
        self.transform = transform
        self.post_transform = post_transform
        self.metabolites = metabolites
        self.task = task

    def __len__(self):
        return len(self.annotations)

    def mask_central_am(self, image):

        binary_image = image[..., -1] > 0.5

        # Label the image
        labeled_image, num_labels = measure.label(binary_image, return_num=True)
        
        if num_labels > 1:
        
            image_center = np.array([[labeled_image.shape[0] / 2, labeled_image.shape[1] / 2]])
            properties = measure.regionprops(labeled_image)
            centroids = np.array([prop.centroid for prop in properties])
            distances = cdist(centroids, image_center)
            centermost_label = properties[np.argmin(distances)].label
            centermost_mask = labeled_image == centermost_label
            dilated_mask = morphology.dilation(centermost_mask, morphology.disk(3))
            image[..., -1] = image[..., -1] * dilated_mask

        return image

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.loc[idx, 'filename'])
        image = imread(img_name)
        
        # Experiment - skip BF of AM -- Final solution
        # image = image[..., [0, 1, 2, 4]]


        image = np.moveaxis(image, -1, 0)
        image = image.transpose((1, 2, 0))  # Convert to HWC format for Albumentations

        metabolite_intensities = self.annotations.loc[idx, self.metabolites].values.astype(np.float32)
        if self.task == 'regression':
            label = torch.from_numpy(metabolite_intensities).float() # Regression
        elif self.task == 'classification':
            label = torch.from_numpy(metabolite_intensities > 0) # Classification

        if self.hard_transform:
            image = self.hard_transform(image=image)['image']

        if self.transform:
            image = self.transform(image=image)['image']
            image = self.mask_central_am(image)

        if self.post_transform:
            image = self.post_transform(image=image)['image']
        
        return image, label

class ImageRegressor(pl.LightningModule):
    def __init__(self, num_classes, metabolite_weights=False, learning_rate=1e-3, n_epochs=200, encoder='resnet152', in_chans=4):
        super().__init__()

        if encoder == 'mit_b5': 
            weights = 'imagenet'
            in_channels = 3
        else: 
            weights = None
            in_channels = in_chans

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder, 
            encoder_weights=weights, 
            in_channels=in_channels, 
            classes=num_classes,
        )
        
        if encoder == 'mit_b5': self.adapt_input_model(in_chans)

        self.learning_rate = learning_rate
        self.epochs = n_epochs
        self.weights = metabolite_weights

        self.pearson_corrcoef = PearsonCorrCoef(num_outputs=1)
        self.r2_score = R2Score(num_outputs=1)

    def adapt_input_conv(self, in_chans, conv_weight):
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

    def adapt_input_model(self, in_channels):
        """
        Adapts first layer to take a specified number of input channels.
        
        Args:
            model: The segmentation model to be adapted.

        Returns:
            The adapted segmentation model.
        """
        # Adapt first layer to take 1 channel as input - timm approach = sum weights
        new_weights = self.adapt_input_conv(in_chans=in_channels, conv_weight=self.model.encoder.patch_embed1.proj.weight)
        self.model.encoder.patch_embed1.proj = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        with torch.no_grad():
            self.model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
        
        # return model

    def custom_weighted_mse_loss(self, y_hat, y, weights, non_zero_mask):
        effective_loss = weights + (y_hat - y) ** 2 * non_zero_mask
        loss = effective_loss.sum() / non_zero_mask.sum()
        return loss

    def forward(self, x):
        x = self.model(x)
        return x

    def compute_loss(self, image, y, non_zero_mask):
        image_pred = self(image).squeeze()

        mask = image[:, -1, ...].unsqueeze(1) #> 0.5 # Binary instead of probability map - less stable somehow
        masked_preds = image_pred * mask
        y_hat = torch.sum(masked_preds.view(masked_preds.size(0), masked_preds.size(1), -1), dim=-1)

        # if not self.weights:
        #     loss = F.mse_loss(y_hat, y)

        # else:
        loss = self.custom_weighted_mse_loss(
            y_hat,
            y, 
            torch.from_numpy(self.weights).to('cuda'), 
            non_zero_mask
            )

        return loss, y_hat

    def training_step(self, batch):
        image, y = batch
        non_zero_mask = y != -10 

        loss, y_hat = self.compute_loss(image, y, non_zero_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('train_pearson_corrcoef', self.pearson_corrcoef(y_hat[non_zero_mask], y[non_zero_mask]).mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_r2_score', self.r2_score(y_hat[non_zero_mask], y[non_zero_mask]).mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch):
        image, y = batch
        non_zero_mask = y != -10 

        loss, y_hat = self.compute_loss(image, y, non_zero_mask)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        self.log('val_pearson_corrcoef', self.pearson_corrcoef(y_hat[non_zero_mask], y[non_zero_mask]).mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2_score', self.r2_score(y_hat[non_zero_mask], y[non_zero_mask]).mean(), on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):

        # optimizer = Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=5e-3)
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # On weight decay: https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8

        scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_dataloader()) * self.trainer.max_epochs),
            # 'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'scheduler': StepLR(optimizer, step_size=self.epochs//4, gamma=0.5), #gamma 0.2 leads to overfitting with mse at first reduction (start lr=1e-3)
            'monitor': 'val_loss',
            'name': 'step_lr'
        }

        return [optimizer], [scheduler]

def get_loaders(folder_path, annotations, metabolites, batch_size, train_annotations, val_annotations, hard_transform, transform, post_transform, task, num_workers=2):
    
    train_dataset = ImageDataset(
        annotations=train_annotations.reset_index(), 
        metabolites=metabolites, 
        root_dir=folder_path, 
        hard_transform=hard_transform,
        transform=transform, 
        post_transform=post_transform,
        task=task
        )

    if num_workers == 0:
        persistant = False
    else:
        persistant = True

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers,
        persistent_workers=persistant,
        drop_last=True,
        )

    val_dataset = ImageDataset(
        val_annotations.reset_index(), 
        metabolites=metabolites, 
        root_dir=folder_path, 
        hard_transform=hard_transform,
        transform=None, 
        post_transform=post_transform,
        task=task
        )    
        
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        num_workers=num_workers, 
        persistent_workers=persistant,
        drop_last=True
        )

    return train_dataloader, val_dataloader, annotations

def train_regressor(folder_path, model_path, batch_size, learning_rate, epochs, encoder, transform_collection, in_chans, test_size=0.33, metabolite_oi=None):

    annotations = pd.read_csv(os.path.join(folder_path, 'ion_intensities.csv'))
    df_normalized, metabolites, weights = process_annotations(annotations, remove_zeros_samples=True)

    if metabolite_oi is not None: 
        metabolites = metabolite_oi

    train_annotations, val_annotations = train_test_split(df_normalized, test_size=test_size, random_state=42) # for reproducibility

    hard_transform, transform, post_transform = transform_collection

    train_dataloader, val_dataloader, annotations = get_loaders(
        folder_path, 
        annotations, 
        metabolites, 
        batch_size, 
        train_annotations, 
        val_annotations, 
        hard_transform, 
        transform, 
        post_transform,
        task='regression', 
        num_workers=2
        )

    model = ImageRegressor(num_classes=len(metabolites), metabolite_weights=weights, n_epochs=epochs, encoder=encoder, learning_rate=learning_rate, in_chans=in_chans)
    
    logger_dirname = f'{encoder}_lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_pearson_corrcoef',
        dirpath=os.path.join(model_path, logger_dirname), 
        filename='{epoch:02d}-{val_loss:.4f}-{val_pearson_corrcoef:.3f}-{val_r2_score:.3f}',
        save_top_k=1, 
        mode='max'
    )

    # Create the trainer with the checkpoint callback
    logger = TensorBoardLogger(
        save_dir=model_path, 
        name=logger_dirname
        )
    
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    return model, trainer

def load_regressor(checkpoint_path, num_classes, encoder, in_chans):
    model = ImageRegressor.load_from_checkpoint(checkpoint_path, num_classes=num_classes, encoder=encoder, in_chans=in_chans)
    model.eval()
    return model
