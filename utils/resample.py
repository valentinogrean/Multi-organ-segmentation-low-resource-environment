import tensorflow as tf
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, \
                                          dice_crossentropy, tversky_loss
from miscnn.evaluation.cross_validation import cross_validation
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                       EarlyStopping, CSVLogger

import os
import numpy as np

interface = NIFTI_interface(channels=1, classes=5)

# Create Data IO object to load and write samples in the file structure
data_io = Data_IO(interface, input_path="/home/vali/src/datasets/multi-organ-segthor", delete_batchDir=False)

# Access all available samples in our file structure
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True,
                             elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True,
                             gaussian_noise=True)

sf_normalize = Normalization(mode="grayscale")
sf_zscore = Normalization(mode="z-score")
sf_clipping = Clipping(min=-1000, max=1000)
sf_resample = Resampling((1.78, 1.78, 3.22))

# Assemble Subfunction classes into a list
#sf = [sf_clipping, sf_normalize, sf_resample, sf_zscore]
sf = [sf_clipping, sf_normalize, sf_resample]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage", patch_shape=(144, 144, 64))
# Adjust the patch overlap for predictions
pp.patchwise_overlap = (140, 140, 16)


# Initialize Keras Data Generator for generating batches
from miscnn.neural_network.data_generator import DataGenerator

dataGen = DataGenerator(sample_list, pp, training=False, validation=False, shuffle=False)

x = []
y = []
z = []
for batch in dataGen:
    print("Batch:", batch.shape)
    x.append(batch.shape[1])
    y.append(batch.shape[2])
    z.append(batch.shape[3])

print("Mean:")
print(np.min(x), np.min(y), np.min(z))
print(np.max(x), np.max(y), np.max(z))
print(np.mean(x), np.mean(y), np.mean(z))
print(np.median(x), np.median(y), np.median(z))

data_io.batch_cleanup()
