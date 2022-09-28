import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import miscnn

from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
interface = NIFTI_interface(pattern="Patient_[0-9]*", channels=1,
                            classes=5)
# Create the Data I/O object
data_path = "/home/vali/src/datasets/multi-organ-segthor-train"
data_io = miscnn.Data_IO(interface, data_path)

# Create and configure the Data Augmentation class
data_aug = miscnn.Data_Augmentation(cycles=2, scaling=True, rotations=True,
                                    elastic_deform=True, mirror=True,
                                    brightness=True, contrast=True,
                                    gamma=True, gaussian_noise=True)

# Create and configure the Preprocessor class
from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling

#subfunctions
sf_normalize = Normalization(mode="z-score")
sf_clipping = Clipping(min=-1000, max=1000)
sf_resample = Resampling((1.78, 1.78, 3.22))

subfunctions = [sf_clipping, sf_normalize, sf_resample]

pp = miscnn.Preprocessor(data_io, data_aug=data_aug, batch_size=2, 
                        subfunctions=subfunctions, prepare_subfunctions=True,
                        prepare_batches=False,
                        analysis="patchwise-crop", patch_shape=(144,144,64))
pp.patchwise_overlap = (72, 72, 32)

# Import standard U-Net architecture and Soft Dice
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft_loss, dice_crossentropy, tversky_loss, tversky_crossentropy

# Create a deep learning neural network model with a standard U-Net architecture
model = Neural_Network(preprocessor=pp, loss=tversky_crossentropy, metrics=[dice_soft_loss, dice_crossentropy, tversky_loss],
                       batch_queue_size=3, workers=3, learninig_rate=0.0001)

sample_list = data_io.get_indiceslist()
sample_list.sort()

from miscnn.evaluation.cross_validation import cross_validation
from miscnn.evaluation import split_validation
from tensorflow.keras.callbacks import ModelCheckpoint

cb_model_val_loss = ModelCheckpoint("/home/vali/src/NeuralNetworks/segthor-val-loss.hdf5",
                                   monitor="val_loss", verbose=1,
                                   save_best_only=True, mode="min")
cb_model_loss = ModelCheckpoint("/home/vali/src/NeuralNetworks/segthor-loss.hdf5",
                                   monitor="loss", verbose=1,
                                   save_best_only=True, mode="min")

# Define callbacks for early stopping
#from tensorflow.keras.callbacks import ReduceLROnPlateau
#cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, mode='min', min_delta=0.0001, cooldown=1, min_lr=0.00001)
#from tensorflow.keras.callbacks import EarlyStopping
#cb_es = EarlyStopping(monitor='loss', min_delta=0, patience=150, verbose=1, mode='min')

# normal learning
#model.reset_weights()
model.load("/home/vali/src/NeuralNetworks/segthor-loss.hdf5")

try:
  split_validation(sample_list, model, epochs=500, iterations=80, percentage=0.02, draw_figures=False, run_detailed_evaluation=False, 
    callbacks=[cb_model_val_loss, cb_model_loss])
except:
  print("An exception occurred")
