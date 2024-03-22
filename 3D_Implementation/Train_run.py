from tqdm import tqdm
import os
import time
from random import randint

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from torch.cuda.amp import autocast, GradScaler


import albumentations as A
from albumentations import Compose, HorizontalFlip
from albumentations.pytorch import ToTensor, ToTensorV2 

import warnings

from Meter import BCEDiceLoss, dice_coef_metric_per_classes, jaccard_coef_metric_per_classes
warnings.simplefilter("ignore")

import random as random


from Dataset import BratsDataset
from DoubleConv import DoubleConv, Down, Up, UNet3d
from GlobalConfig import GlobalConfig
from GlobalConfig import *
from Image3dToGIF3d import Image3dToGIF3d, ShowResult, merging_two_gif
from Trainer import Trainer

def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 8,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader

def compute_scores_per_classes(model,
                               dataloader,
                               classes):
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:
        model: neural net for make predictions.
        dataloader: dataset object to load data from.
        classes: list with classes.
        Returns: dictionaries with dice and jaccard coefficients for each class for each slice.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes

def compute_results(model,
                    dataloader,
                    treshold=0.33):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {"Id": [],"image": [], "GT": [],"Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            id_, imgs, targets = data['Id'], data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            predictions = (probs >= treshold).float()
            predictions =  predictions.cpu()
            targets = targets.cpu()
            
            results["Id"].append(id_)
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)
            
            # only 5 pars
            if (i > 5):    
                return results
        return results


def main():

#     sample_filename = '/data/mm510/UCSF-PDGM-v3/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_FLAIR.nii'
#     sample_filename_mask = '/data/mm510/UCSF-PDGM-v3/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_tumor_segmentation.nii'

    sample_filename = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_FLAIR.nii'
    sample_filename_mask = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_tumor_segmentation.nii'

    sample_img = nib.load(sample_filename)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    sample_mask = nib.load(sample_filename_mask)
    sample_mask = np.asanyarray(sample_mask.dataobj)
    sample_mask = np.rot90(sample_mask)
    print("img shape ->", sample_img.shape)
    print("mask shape ->", sample_mask.shape)


    # sample_filename2 = '/data/mm510/UCSF-PDGM-v3/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1.nii'
    sample_filename2 = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1.nii'
    sample_img2 = nib.load(sample_filename2)
    sample_img2 = np.asanyarray(sample_img2.dataobj)
    sample_img2  = np.rot90(sample_img2)

    # sample_filename3 = '/data/mm510/UCSF-PDGM-v3/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T2.nii'
    sample_filename3 = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T2.nii'
    sample_img3 = nib.load(sample_filename3)
    sample_img3 = np.asanyarray(sample_img3.dataobj)
    sample_img3  = np.rot90(sample_img3)

    # sample_filename4 = '/data/mm510/UCSF-PDGM-v3/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1c.nii'
    sample_filename4 = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1c.nii'
    sample_img4 = nib.load(sample_filename4)
    sample_img4 = np.asanyarray(sample_img4.dataobj)
    sample_img4  = np.rot90(sample_img4)

    mask_WT = sample_mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1

    mask_TC = sample_mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 4] = 1

    mask_ET = sample_mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 4] = 1

    fig = plt.figure(figsize=(20, 10))

    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    flair = ax0.imshow(sample_img[:,:,65], cmap='bone')
    ax0.set_title("FLAIR", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(flair)

    #  Varying density along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    t1 = ax1.imshow(sample_img2[:,:,65], cmap='bone')
    ax1.set_title("T1", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1)

    #  Varying density along a streamline
    ax2 = fig.add_subplot(gs[0, 2])
    t2 = ax2.imshow(sample_img3[:,:,65], cmap='bone')
    ax2.set_title("T2", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t2)

    #  Varying density along a streamline
    ax3 = fig.add_subplot(gs[0, 3])
    t1ce = ax3.imshow(sample_img4[:,:,65], cmap='bone')
    ax3.set_title("T1 contrast", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1ce)

    #  Varying density along a streamline
    ax4 = fig.add_subplot(gs[1, 1:3])

    #ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
    l1 = ax4.imshow(mask_WT[:,:,65], cmap='summer',)
    l2 = ax4.imshow(np.ma.masked_where(mask_TC[:,:,65]== False,  mask_TC[:,:,65]), cmap='rainbow', alpha=0.6)
    l3 = ax4.imshow(np.ma.masked_where(mask_ET[:,:,65] == False, mask_ET[:,:,65]), cmap='winter', alpha=0.6)

    ax4.set_title("", fontsize=20, weight='bold', y=-0.1)

    _ = [ax.set_axis_off() for ax in [ax0,ax1,ax2,ax3, ax4]]

    colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
    labels = ['Non-Enhancing tumor core', 'Peritumoral Edema ', 'GD-enhancing tumor']
    patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 'xx-large',
               title='Mask Labels', title_fontsize=18, edgecolor="black",  facecolor='#c5c6c7')

    plt.suptitle("Multimodal Scans -  Data | Manually-segmented mask - Target", fontsize=20, weight='bold')

    fig.savefig("data_sample.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("data_sample.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

    # Directory containing your data
#     data_directory = '/data/mm510/UCSF-PDGM-v3/'
    data_directory = '/data'

    # Exclude the specific path you want to skip
#     excluded_path = '/data/mm510/UCSF-PDGM-v3/UCSF-PDGM-0541_nifti'
    excluded_path = '/data/UCSF-PDGM-0541_nifti'
    excluded_dirname = os.path.basename(excluded_path)
    if excluded_path in data_directory:
        data_directory = data_directory.replace(excluded_path, '')

    # List all files in the directory
    all_files = os.listdir(data_directory)

    # Exclude the specific path you want to skip
    # excluded_path = '/data/UCSF-PDGM-0541_nifti'
    # if excluded_path in all_files:
    #     all_files.remove(excluded_path)

    # Shuffle the list of files
    random.shuffle(all_files)

    # Define the number of files for training and validation
    num_train_files = 400

    # Split the shuffled list into training and validation sets
    train_files = all_files[:num_train_files]
    val_files = all_files[num_train_files:]

    # Define the paths for training and validation datasets
    TRAIN_DATASET_PATH = [os.path.join(data_directory, file) for file in train_files]
    VAL_DATASET_PATH = [os.path.join(data_directory, file) for file in val_files]

    # Print the first few file paths for verification
    print("Training Dataset Paths:")
    print(TRAIN_DATASET_PATH[:5])  # Print the first 5 paths
    print("\nValidation Dataset Paths:")
    print(VAL_DATASET_PATH[:5])    # Print the first 5 paths

    # Load the dataset
#     survival_info_df = pd.read_csv('/home/mm510/Dissertation/UCSF-PDGM-metadata_v2.csv')

    config = GlobalConfig()
    seed_everything(config.seed)
    
    survival_info_df = pd.read_csv('metadata.csv')

    # Further processing, including age ranking and removing specific IDs
    survival_info_df["Age_rank"] = survival_info_df["Age at MRI"] // 10 * 10
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-541'].reset_index(drop=True)
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-181'].reset_index(drop=True)
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-315'].reset_index(drop=True)
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-289'].reset_index(drop=True)
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-138'].reset_index(drop=True)
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-175'].reset_index(drop=True)
    survival_info_df = survival_info_df[survival_info_df['ID'] != 'UCSF-PDGM-278'].reset_index(drop=True)

    # Adding path information with zero-padded ID
#     survival_info_df['path'] = survival_info_df['ID'].apply(lambda id_: f"/data/mm510/UCSF-PDGM-v3/{'-'.join(id_.split('-')[:-1]) + '-' + id_.split('-')[-1].zfill(4)}_nifti/{'-'.join(id_.split('-')[:-1]) + '-' + id_.split('-')[-1].zfill(4)}_FLAIR.nii")
    survival_info_df['path'] = survival_info_df['ID'].apply(lambda id_: f"/data/{'-'.join(id_.split('-')[:-1]) + '-' + id_.split('-')[-1].zfill(4)}_nifti/{'-'.join(id_.split('-')[:-1]) + '-' + id_.split('-')[-1].zfill(4)}_FLAIR.nii")


    # Initializing StratifiedKFold and assigning fold numbers
    skf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)
    for fold, (_, val_index) in enumerate(skf.split(survival_info_df, survival_info_df["Age_rank"])):
        survival_info_df.loc[val_index, "fold"] = fold

    # Assigning 'phase' based on 'BraTS21 Segmentation Cohort' and 'fold'
    survival_info_df['phase'] = 'test'
    survival_info_df.loc[survival_info_df['BraTS21 Segmentation Cohort'] == 'Training', 'phase'] = 'train'
    survival_info_df.loc[survival_info_df['fold'] == 0, 'phase'] = 'val'

    # Splitting data into train, val, and test sets
    train_df = survival_info_df[survival_info_df['phase'] == 'train']
    val_df = survival_info_df[survival_info_df['phase'] == 'val']
    test_df = survival_info_df[survival_info_df['phase'] == 'test']

    # Print the shapes of the split data
    print("Training data shape:", train_df.shape)
    print("Validation data shape:", val_df.shape)
    print("Testing data shape:", test_df.shape)

    # Saving the entire DataFrame with phases to a CSV file
    survival_info_df.to_csv('train_data.csv', index=False)

    print("Data saved to train_data.csv")

#     dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='/home/mm510/Dissertation/new_model/train_data.csv', phase='valid', fold=0, num_workers= 0)
    dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='train_data.csv', phase='valid', fold=0, num_workers= 8)
    len(dataloader)

    data = next(iter(dataloader))
    data['Id'], data['image'].shape, data['mask'].shape

    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
    mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor)) 

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask == False, mask),
               cmap='cool', alpha=0.6)
    plt.savefig("segmentMap.png")

    nodel = UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
    trainer = Trainer(net=nodel,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=25,
                  batch_size=1,
                  fold=0,
                  num_epochs=50,
                  path_to_csv = config.path_to_csv,)

    if config.pretrained_model_path is not None:
        trainer.load_predtrain_model(config.pretrained_model_path)
        
        # if need - load the logs.      
        train_logs = pd.read_csv(config.train_logs_path)
        trainer.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()
        trainer.losses["val"] =  train_logs.loc[:, "val_loss"].to_list()
        trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
        trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
        trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
        trainer.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()

#     torch.cuda.empty_cache()
    
#     trainer.run()

#     trainer.display_plot()

    val_dataloader = get_dataloader(BratsDataset, 'train_data.csv', phase='valid', fold=0)
    len(dataloader)

    nodel.eval()

    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
    nodel, val_dataloader, ['WT', 'TC', 'ET']
    )

    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard', 
                                          'TC dice', 'TC jaccard', 
                                          'ET dice', 'ET jaccard']]
    val_metics_df.sample(5)

    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6));
    sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax);
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15);
    ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=20)

    for idx, p in enumerate(ax.patches):
            percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
            x = p.get_x() + p.get_width() / 2 - 0.15
            y = p.get_y() + p.get_height()
            ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig("result1.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("result1.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

    results = compute_results(
        nodel, val_dataloader, 0.33)

    for id_, img, gt, prediction in zip(results['Id'][4:],
                    results['image'][4:],
                    results['GT'][4:],
                    results['Prediction'][4:]
                    ):
    
        print(id_)
        break

    show_result = ShowResult()
    show_result.plot(img, gt, prediction, 'unet_3d_prediction.png')

if __name__ == "__main__":
    main()
