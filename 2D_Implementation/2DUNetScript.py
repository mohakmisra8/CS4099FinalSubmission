import os
# import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps   


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt

# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing



np.set_printoptions(precision=3, suppress=True)

# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include

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


import os
import glob

# Initialize lists for different types of files
train_t2_list = []
train_t1ce_list = []
train_flair_list = []
train_mask_list = []

# Iterate through all directories in TRAIN_DATASET_PATH
for directory in TRAIN_DATASET_PATH:
    ID = os.path.basename(directory)  # Extract the ID from the directory name

    # List files within the directory associated with the ID
    if ID in train_files:
        id_directory = os.path.join(data_directory, ID)
        
        # Glob for T2 files
        t2_files = glob.glob(os.path.join(id_directory, '*_T2.nii'))
        train_t2_list.extend(t2_files)

        # Glob for T1ce files
        t1ce_files = glob.glob(os.path.join(id_directory, '*_T1c.nii'))
        train_t1ce_list.extend(t1ce_files)

        # Glob for FLAIR files
        flair_files = glob.glob(os.path.join(id_directory, '*_FLAIR.nii'))
        train_flair_list.extend(flair_files)

        # Glob for mask files
        mask_files = glob.glob(os.path.join(id_directory, '*_brain_segmentation.nii'))
        train_mask_list.extend(mask_files)

# t2_list now contains all T2 files
# print("T2 List:")
# print(train_t2_list)

import os
import glob

# Initialize lists for different types of files
validation_t2_list = []
validation_t1ce_list = []
validation_flair_list = []
validation_mask_list = []

# Iterate through all directories in VAL_DATASET_PATH
for directory in VAL_DATASET_PATH:
    ID = os.path.basename(directory)  # Extract the ID from the directory name

    # List files within the directory associated with the ID
    if ID in val_files:
        id_directory = os.path.join(data_directory, ID)

        # Glob for T2 files
        t2_files = glob.glob(os.path.join(id_directory, '*_T2.nii'))
        validation_t2_list.extend(t2_files)

        # Glob for T1ce files
        t1ce_files = glob.glob(os.path.join(id_directory, '*_T1c.nii'))
        validation_t1ce_list.extend(t1ce_files)

        # Glob for FLAIR files
        flair_files = glob.glob(os.path.join(id_directory, '*_FLAIR.nii'))
        validation_flair_list.extend(flair_files)

        # Glob for mask files
        mask_files = glob.glob(os.path.join(id_directory, '*_brain_segmentation.nii'))
        validation_mask_list.extend(mask_files)

# t2_list now contains all T2 files
# print("T2 List for Validation:")
# print(validation_t2_list)

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

IMG_SIZE=128

from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Dropout, concatenate
from keras.models import Model

def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up], axis = 3)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)

input_layer = Input((IMG_SIZE, IMG_SIZE, 2))


model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )

train_and_val_directories = TRAIN_DATASET_PATH + VAL_DATASET_PATH
train_and_val_directories

# # Function to extract study IDs from directory paths
def pathListIntoIds(dirList):
    x = [dirList[i][dirList[i].rfind('/')+1:] for i in range(len(dirList))]
    return x

# Extract study IDs
train_and_test_ids = pathListIntoIds(train_and_val_directories)

# Split the study IDs into training, testing, and validation sets
train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)


import re  # Import the re module

def extractIDWithoutNifti(ID):
    # Remove "_nifti" and additional characters like "_FUXXXd"
    cleaned_id = re.sub(r'_nifti|_FU\d+d', '', ID)
    return cleaned_id

# Extract study IDs without "_nifti" and additional characters based on the original IDs
train_ids_without_nifti = [extractIDWithoutNifti(ID) for ID in train_ids]
val_ids_without_nifti = [extractIDWithoutNifti(ID) for ID in val_ids]
test_ids_without_nifti = [extractIDWithoutNifti(ID) for ID in test_ids]

# Verify if the IDs have the correct format
for study_id in train_ids_without_nifti + val_ids_without_nifti + test_ids_without_nifti:
    if not re.match(r'^[A-Z0-9-]+$', study_id):
        print(f"Invalid study ID: {study_id}")


test_ids_without_nifti

from skimage.transform import resize

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_directory, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.data_directory = data_directory  
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        # Generate data
        for c, i in enumerate(Batch_ids):
            # Extract study ID without '_nifti'
            nifti_without_id = extractIDWithoutNifti(i)
            # dir = data_directory + 

            case_path = os.path.join(self.data_directory, nifti_without_id+"_nifti")  # Use the NIfTI ID

            data_path = os.path.join(case_path, f'{nifti_without_id}_FLAIR.nii')  # Update the path with NIfTI ID
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{nifti_without_id}_T1c.nii')  # Update the path with NIfTI ID
            ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{nifti_without_id}_tumor_segmentation.nii')  # Update the path with NIfTI ID
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):

#                 print("Shape before resizing - FLAIR:", flair.shape)
#                 print("Shape before resizing - CE:", ce.shape)
                # Resize the flair and ce images
                X[j + VOLUME_SLICES * c, :, :, 0] = resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j + VOLUME_SLICES * c, :, :, 1] = resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

#                 print("Shape after resizing - FLAIR:", X[j + VOLUME_SLICES * c, :, :, 0].shape)
#                 print("Shape after resizing - CE:", X[j + VOLUME_SLICES * c, :, :, 1].shape)

                # Assign the segmentation mask directly
                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT]

        # Generate masks
        y[y == 4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
        return X / np.max(X), Y



        
# training_generator = DataGenerator('/data/UCSF-PDGM-v3/', train_ids)
# valid_generator = DataGenerator('/data/UCSF-PDGM-v3/', val_ids)
# test_generator = DataGenerator('/data/UCSF-PDGM-v3/', test_ids)

training_generator = DataGenerator(data_directory, train_ids)

# training_generator._DataGenerator__data_generation(train_ids)

valid_generator = DataGenerator(data_directory, val_ids)
test_generator = DataGenerator(data_directory, test_ids)

# Remove "UCSF-PDGM-0541_nifti" from all lists
train_test_ids = [id for id in train_test_ids if id != "UCSF-PDGM-0541_nifti"]
val_ids = [id for id in val_ids if id != "UCSF-PDGM-0541_nifti"]
train_ids = [id for id in train_ids if id != "UCSF-PDGM-0541_nifti"]
test_ids = [id for id in test_ids if id != "UCSF-PDGM-0541_nifti"]

is_in_train_test_ids = "UCSF-PDGM-0541_nifti" in train_test_ids
is_in_val_ids = "UCSF-PDGM-0541_nifti" in val_ids
is_in_train_ids = "UCSF-PDGM-0541_nifti" in train_ids
is_in_test_ids = "UCSF-PDGM-0541_nifti" in test_ids

print(is_in_train_test_ids)
print(is_in_val_ids)
print(is_in_train_ids)
print(is_in_test_ids)

print(test_ids)

csv_logger = CSVLogger('2D_training.log', separator=',', append=False)


callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
        csv_logger
    ]

K.clear_session()

history =  model.fit(training_generator,
                    epochs=35,
                    steps_per_epoch=len(train_ids),
                    callbacks= callbacks,
                    validation_data = valid_generator
                    )  
# # for epoch in range(35):
# #     for batch in training_generator:
# #         X_batch, Y_batch = batch
# #         predictions = model.predict(X_batch)  # Get predictions for the batch
# #         # Add debug output to check for negative predictions
# #         print("Batch Predictions: Min Value = {}, Max Value = {}".format(predictions.min(), predictions.max()))                    
model.save("2D_model.h5")

import os

def path_exists(path):
    """
    Check if a path exists.

    Args:
    path (str): The path to check.

    Returns:
    bool: True if the path exists, False if it does not.
    """
    return os.path.exists(path)

path = "/data/UCSF-PDGM-0406_nifti"
if path_exists(path):
    print(f"The path '{path}' exists.")
else:
    print(f"The path '{path}' does not exist.")