import numpy as np
import nibabel as nib
import glob
import os
import sys
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
import tensorflow as tf
from tensorflow.keras.losses import Loss
import segmentation_models_3D as sm
# segmentation_models = '/exp1/experiment1Softmax/segmentation_models/'


# sys.path.append(segmentation_models)
import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from tensorflow.keras.backend import clear_session
clear_session()

TRAIN_DATASET_PATH = '/data'


sample_filename = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_FLAIR.nii'
sample_filename2 = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1.nii'
sample_filename3 = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T2.nii'
sample_filename4 = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1c.nii'
sample_filename_mask = '/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_tumor_segmentation.nii'

test_image_flair=nib.load(sample_filename).get_fdata()
print(test_image_flair.max())
#Scalers are applied to 1D so let us reshape and then reshape back to original shape. 
# test_image_flair=scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)


test_image_t1=nib.load('/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1.nii').get_fdata()
# test_image_t1=scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce=nib.load('/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T1c.nii').get_fdata()
# test_image_t1ce=scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2=nib.load('/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_T2.nii').get_fdata()
# test_image_t2=scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask=nib.load('/data/UCSF-PDGM-0014_nifti/UCSF-PDGM-0014_tumor_segmentation.nii').get_fdata()
test_mask=test_mask.astype(np.uint8)

print(np.unique(test_mask))  #0, 1, 2, 4 (Need to reencode to 0, 1, 2, 3)
test_mask[test_mask==4] = 3  #Reassign mask values 4 to 3
print(np.unique(test_mask)) 

import os
import random

# Directory containing your data
data_directory = '/data'

# Exclude the specific path you want to skip
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

train_t2_list = []
train_t1ce_list = []
train_flair_list = []
train_mask_list = []

# Iterate through all directories in TRAIN_DATASET_PATH
for directory in TRAIN_DATASET_PATH:
    # Extract the base directory name, removing '_nifti' if it is part of the directory name
    base_directory_name = os.path.basename(directory)
    
    # Assuming the ID is correctly represented in the directory name without '_nifti'
    ID = base_directory_name.replace('_nifti', '')

    # Construct file paths without '_nifti' in the filename part
    t2_path = glob.glob(os.path.join(directory, f'{ID}_T2.nii'))
    t1ce_path = glob.glob(os.path.join(directory, f'{ID}_T1c.nii'))
    flair_path = glob.glob(os.path.join(directory, f'{ID}_FLAIR.nii'))
    mask_path = glob.glob(os.path.join(directory, f'{ID}_tumor_segmentation.nii'))

    # Check if all modalities are present for the ID and add them at the same index
    if t2_path and t1ce_path and flair_path and mask_path:
        train_t2_list.append(t2_path[0])  
        train_t1ce_list.append(t1ce_path[0])
        train_flair_list.append(flair_path[0])
        train_mask_list.append(mask_path[0])
        
import os
import glob

# Initialize lists for different types of files
val_t2_list = []
val_t1ce_list = []
val_flair_list = []
val_mask_list = []

# Iterate through all directories in TRAIN_DATASET_PATH
for directory in VAL_DATASET_PATH:
    # Extract the base directory name, removing '_nifti' if it is part of the directory name
    base_directory_name = os.path.basename(directory)
    
    # Assuming the ID is correctly represented in the directory name without '_nifti'
    ID = base_directory_name.replace('_nifti', '')

    # Construct file paths without '_nifti' in the filename part
    t2_path = glob.glob(os.path.join(directory, f'{ID}_T2.nii'))
    t1ce_path = glob.glob(os.path.join(directory, f'{ID}_T1c.nii'))
    flair_path = glob.glob(os.path.join(directory, f'{ID}_FLAIR.nii'))
    mask_path = glob.glob(os.path.join(directory, f'{ID}_tumor_segmentation.nii'))

    # Check if all modalities are present for the ID and add them at the same index
    if t2_path and t1ce_path and flair_path and mask_path:
        val_t2_list.append(t2_path[0])  
        val_t1ce_list.append(t1ce_path[0])
        val_flair_list.append(flair_path[0])
        val_mask_list.append(mask_path[0])

# t2_list now contains all T2 files
# print("T2 List:")
# print(val_t2_list)

index_to_check = 0  # Example index
if index_to_check < len(val_t2_list):
    print(f"Files at index {index_to_check}:")
    print(f"T2: {val_t2_list[index_to_check]}")
    print(f"T1ce: {val_t1ce_list[index_to_check]}")
    print(f"FLAIR: {val_flair_list[index_to_check]}")
    print(f"Mask: {val_mask_list[index_to_check]}")
else:
    print("Index is out of range.")

images_dir = '/exp1/experiment1Softmax/images'
masks_dir = '/exp1/experiment1Softmax/masks'

# Ensure the directories exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

images_dir = '/exp1/experiment1Softmax/val/images'
masks_dir = '/exp1/experiment1Softmax/val/masks'

# Ensure the directories exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

models = '/exp1/savedModels'
# masks_dir = '/exp1/new_model/experiment1Softmax/val/masks'

# Ensure the directories exist
os.makedirs(models, exist_ok=True)
# os.makedirs(masks_dir, exist_ok=True)

# for img in range(len(train_t2_list)):  # Using t2_list as all lists are of the same size
#     print("Now preparing image and masks number: ", img)

#     # Load and normalize T2 images
# #     print(train_t2_list[img])
#     temp_image_t2 = nib.load(train_t2_list[img]).get_fdata()
#     temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
# #     unique_values = np.unique(temp_image_t2)
# #     print(unique_values)

#     # Load and normalize T1ce images
# #     print(train_t1ce_list[img])
#     temp_image_t1ce = nib.load(train_t1ce_list[img]).get_fdata()
#     temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

#     # Load and normalize FLAIR images
# #     print(train_flair_list[img])
#     temp_image_flair = nib.load(train_flair_list[img]).get_fdata()
#     temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

#     # Load masks and adjust values
# #     print(train_mask_list[img])
#     temp_mask = nib.load(train_mask_list[img]).get_fdata()
#     temp_mask = temp_mask.astype(np.uint8)
    
#     temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
#     print("here")
#     unique_values = np.unique(temp_mask)
#     print(unique_values)

#     # Combine images along a new dimension
#     temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
#     #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
#     #cropping x, y, and z
#     temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
#     temp_mask = temp_mask[56:184, 56:184, 13:141]
    
#     val, counts = np.unique(temp_mask, return_counts=True)
    
#     if (1 - (counts[0]/counts.sum())) > 0.0001:  # At least 1% useful volume with labels that are not 0
#         print("Save Me")
#         temp_mask = to_categorical(temp_mask, num_classes=4)
#         np.save('/exp1/experiment1Softmax/images/image_'+str(img)+'.npy', temp_combined_images)
#         np.save('/exp1/experiment1Softmax/masks/mask_'+str(img)+'.npy', temp_mask)
#     else:
#         print("I am useless")
        
# for img in range(len(val_t2_list)):  # Using t2_list as all lists are of the same size
#     print("Now preparing image and masks number: ", img)

#     # Load and normalize T2 images
# #     print(train_t2_list[img])
#     temp_image_t2 = nib.load(val_t2_list[img]).get_fdata()
# #     temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
#     unique_values = np.unique(temp_image_t2)
# #     print(unique_values)

#     # Load and normalize T1ce images
#     temp_image_t1ce = nib.load(val_t1ce_list[img]).get_fdata()
# #     temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

#     # Load and normalize FLAIR images
#     temp_image_flair = nib.load(val_flair_list[img]).get_fdata()
# #     temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

#     # Load masks and adjust values
#     temp_mask = nib.load(val_mask_list[img]).get_fdata()
#     temp_mask = temp_mask.astype(np.uint8)
    
#     temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
#     print("here")
#     unique_values = np.unique(temp_mask)
#     print(unique_values)

#     # Combine images along a new dimension
#     temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
#     #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
#     #cropping x, y, and z
#     temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
#     temp_mask = temp_mask[56:184, 56:184, 13:141]
    
#     val, counts = np.unique(temp_mask, return_counts=True)
    
#     if (1 - (counts[0]/counts.sum())) > 0.0001:  # At least 1% useful volume with labels that are not 0
#         print("Save Me")
#         temp_mask = to_categorical(temp_mask, num_classes=4)
#         np.save('/exp1/experiment1Softmax/val/images/image_'+str(img)+'.npy', temp_combined_images)
#         np.save('/exp1/experiment1Softmax/val/masks/mask_'+str(img)+'.npy', temp_mask)
#     else:
#         print("I am useless")



def load_img(img_dir, img_list):
    images = []
    for image_name in img_list:    
        if image_name.endswith('.npy'):
            image_path = os.path.join(img_dir, image_name)
            image = np.load(image_path)
            images.append(image)
    return np.array(images)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    while True:  # Generator loop
        for batch_start in range(0, L, batch_size):
            limit = min(batch_start + batch_size, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # Yielding a tuple with two elements

from matplotlib import pyplot as plt
import random

train_img_dir = "/exp1/experiment1Softmax/images/"
train_mask_dir = "/exp1/experiment1Softmax/masks/"
train_img_list = sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))


batch_size = 1

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

import os
import numpy as np
#import imageloader
#from custom_datagen import imageLoader
#import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random

train_img_dir = "/exp1/experiment1Softmax/images/"
train_mask_dir = "/exp1/experiment1Softmax/masks/"

img_list = sorted(os.listdir(train_img_dir))
msk_list = sorted(os.listdir(train_mask_dir))

import pandas as pd
columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)

train_mask_dir = "/exp1/experiment1Softmax/masks/"
train_mask_list = sorted(glob.glob(os.path.join(train_mask_dir, '*.npy')))

for img_path in train_mask_list:
    temp_image = np.load(img_path)
    temp_image = np.argmax(temp_image, axis=0)  # Assuming the mask has shape (height, width, channels) and axis=3 is incorrect.
    val, counts = np.unique(temp_image, return_counts=True)
    
    # Ensure counts are matched to the correct columns based on `val`
    counts_dict = {str(v): counts[i] for i, v in enumerate(val) if str(v) in columns}
    
    # Fill missing columns with 0 counts
    for col in columns:
        if col not in counts_dict:
            counts_dict[col] = 0
    
    df = df.append(counts_dict, ignore_index=True)

# Define the column names for the DataFrame
columns = ['0', '1', '2', '3']
# Initialize an empty DataFrame with these columns
df = pd.DataFrame(columns=columns)

# Define the directory containing your mask files
val_mask_dir = '/exp1/experiment1Softmax/val/masks/'
# Sort the list of mask file paths
val_mask_list = sorted(glob.glob(os.path.join(val_mask_dir, '*.npy')))

# Iterate through each sorted mask file
for img_path in val_mask_list:
    # Load the mask using numpy
    temp_image = np.load(img_path)
    # Apply argmax to convert one-hot encoding to class labels, assuming the last axis is channels
    temp_image = np.argmax(temp_image, axis=3)
    # Calculate the counts of each unique value in the mask
    val, counts = np.unique(temp_image, return_counts=True)
    
    # Create a dictionary to map the column names to their corresponding counts
    counts_dict = {str(v): counts[i] for i, v in enumerate(val) if str(v) in columns}
    
    # Fill in zeros for any missing columns to ensure consistency across all rows
    for col in columns:
        if col not in counts_dict:
            counts_dict[col] = 0
    
    # Append the counts dictionary as a new row in the DataFrame
    df = df.append(counts_dict, ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4
#Class weights claculation: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
wt1 = round((total_labels/(n_classes*label_1)), 2)
wt2 = round((total_labels/(n_classes*label_2)), 2)
wt3 = round((total_labels/(n_classes*label_3)), 2)

#Define the image generators for training and validation

train_img_dir = "/exp1/experiment1Softmax/images/"
train_mask_dir = "/exp1/experiment1Softmax/masks/"

val_img_dir = "/exp1/experiment1Softmax/val/images/"
val_mask_dir = "/exp1/experiment1Softmax/val/masks/"

# Retrieve and sort the list of training images and masks
train_img_list = sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

# Retrieve and sort the list of validation images and masks
val_img_list = sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

# Define a function to extract the sort key (e.g., ID) from the file name
def extract_sort_key(filename):
    # This is an example; adjust the slicing as per your file naming convention
    return filename.split('_')[0]

# Sort the training image and mask lists
train_img_list = sorted(train_img_list, key=extract_sort_key)
train_mask_list = sorted(train_mask_list, key=extract_sort_key)

# Sort the validation image and mask lists
val_img_list = sorted(val_img_list, key=extract_sort_key)
val_mask_list = sorted(val_mask_list, key=extract_sort_key)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

# train_img_datagen.

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)
################################### DICE LOSSES USED##############################################################
# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss


 
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

################################### 3D DICE LOSSES REMIMPLEMENTATION##############################################################

# dice loss as defined above for 4 classes
# def dice_coef(y_true, y_pred, smooth=1e-6):
#     class_num = 4
#     for i in range(class_num):
#         y_true_f = K.flatten(y_true[:,:,:,i])
#         y_pred_f = K.flatten(y_pred[:,:,:,i])
#         intersection = K.sum(y_true_f * y_pred_f)
#         loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
#    #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
#         if i == 0:
#             total_loss = loss
#         else:
#             total_loss = total_loss + loss
#     total_loss = total_loss / class_num
# #    K.print_tensor(total_loss, message=' total dice coef: ')
#     return total_loss


 
# # define per class evaluation of dice coef
# # inspired by https://github.com/keras-team/keras/issues/9395
# def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
#     intersection = K.sum(K.abs(y_true[:,:,:,:,1] * y_pred[:,:,:,:,1]))
#     return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,:,1])) + epsilon)

# def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
#     intersection = K.sum(K.abs(y_true[:,:,:,:,2] * y_pred[:,:,:,:,2]))
#     return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,:,2])) + epsilon)

# def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
#     intersection = K.sum(K.abs(y_true[:,:,:,:,3] * y_pred[:,:,:,:,3]))
#     return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,:,3])) + epsilon)




# # Computing Precision 
# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def sensitivity(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     return true_positives / (possible_positives + K.epsilon())

# def specificity(y_true, y_pred):
#     true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
#     possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
#     return true_negatives / (possible_negatives + K.epsilon())

###################################################################################################################################

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling3D, BatchNormalization

from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform' 


################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_coefficient(y_true, y_pred, smooth=1e-6):
#     y_true_f = tf.reshape(y_true, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])
    
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
#     return (2. * intersection + smooth) / (union + smooth)

def dice_loss_specific_class(y_true, y_pred, class_index):
    return dice_loss(y_true[..., class_index], y_pred[..., class_index])

def combined_dice_loss(y_true, y_pred):
    num_classes = tf.shape(y_true)[-1]
    loss = 0.0
    for class_index in range(num_classes): 
        loss += dice_loss(y_true[..., class_index], y_pred[..., class_index])
    return loss / tf.cast(num_classes, tf.float32)

def hybrid_loss_1(alpha, beta, gamma, delta, y_true, y_pred):
    # Cross-entropy loss
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Recall loss for complete region
    rl_com_loss = complete_recall_loss(y_true, y_pred)
    
    # Recall loss for core region
    rl_core_loss = core_recall_loss(y_true, y_pred)
    
    # Recall loss for enhanced region
    rl_enh_loss = enhanced_recall_loss(y_true, y_pred)
    
    return alpha * ce_loss + beta * rl_com_loss + gamma * rl_core_loss + delta * rl_enh_loss

def hybrid_loss_2(alpha, beta, gamma, delta, y_true, y_pred):
    # Combined Dice loss
    dice_combined_loss = combined_dice_loss(y_true, y_pred)
    
    # Recall loss for complete region
    rl_com_loss = complete_recall_loss(y_true, y_pred)
    
    # Recall loss for core region
    rl_core_loss = core_recall_loss(y_true, y_pred)
    
    # Recall loss for enhanced region
    rl_enh_loss = enhanced_recall_loss(y_true, y_pred)
    
    return alpha * dice_combined_loss + beta * rl_com_loss + gamma * rl_core_loss + delta * rl_enh_loss

wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


 
model = simple_unet_model(128, 128, 128, 3, 4)
model.compile(loss=total_loss, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',sm.metrics.IOUScore(threshold=0.5), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )
print(model.input_shape)
print(model.output_shape)

import keras.backend as K
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

batch_size = 8

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size

csv_logger = CSVLogger('dice_loss.log', separator=',', append=False)


callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True),
        csv_logger
    ]

history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=35,
          verbose=1,
          callbacks = callbacks,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )




model.save("custom_dice_loss.h5")

import numpy as np
from keras.models import load_model
import segmentation_models_3D as sm

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size

model = load_model('custom_dice_loss.h5', compile=False)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=total_loss,  
    metrics=[
        sm.metrics.IOUScore(threshold=0.5),  
        dice_coef,
        precision,
        sensitivity,
        specificity,
        dice_coef_necrotic,
        dice_coef_edema,
        dice_coef_enhancing
    ]
)


history2 = model.fit(
    train_img_datagen,
    steps_per_epoch=steps_per_epoch,
    epochs=1,
    verbose=1,
    validation_data=val_img_datagen,
    validation_steps=val_steps_per_epoch,
)


my_model = load_model('custom_dice_loss.h5', compile=False)

from keras.metrics import MeanIoU

batch_size=8 
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)


test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


dice_coefficient_value = dice_coef(test_mask_batch, test_pred_batch)
precision_value = precision(test_mask_batch, test_pred_batch)
sensitivity_value = sensitivity(test_mask_batch, test_pred_batch)
specificity_value = specificity(test_mask_batch, test_pred_batch)
dice_coef_necrotic_value = dice_coef_necrotic(test_mask_batch, test_pred_batch)
dice_coef_edema_value = dice_coef_edema(test_mask_batch, test_pred_batch)
dice_coef_enhancing_value = dice_coef_enhancing(test_mask_batch, test_pred_batch)

print("Dice Coefficient =", dice_coefficient_value.numpy())
print("Precision =", precision_value.numpy())
print("Sensitivity =", sensitivity_value.numpy())
print("Specificity =", specificity_value.numpy())
print("Dice Coefficient Necrotic =", dice_coef_necrotic_value.numpy())
print("Dice Coefficient Edema =", dice_coef_edema_value.numpy())
print("Dice Coefficient Enhancing =", dice_coef_enhancing_value.numpy())

# img_num = 82

test_img = np.load("/exp1/experiment1Softmax/val/images/image_66.npy")

test_mask = np.load("/exp1/experiment1Softmax/val/masks/mask_66.npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


print(test_prediction_argmax.shape)
print(test_mask_argmax.shape)
print(np.unique(test_prediction_argmax))

#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:,:, n_slice])
plt.savefig("custom_dice_loss_prediction.png")
plt.close()