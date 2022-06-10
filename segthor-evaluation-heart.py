import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import miscnn

from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
interface = NIFTI_interface(pattern="Patient_[0-9]*", channels=1,
                            classes=2)
# Create the Data I/O object
data_path = "/home/vali/src/datasets/multi-organ-segthor"
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
sf_clipping = Clipping(min=-400, max=600)
sf_resample = Resampling((1.78, 1.78, 3.22))

subfunctions = [sf_clipping, sf_normalize, sf_resample]

pp = miscnn.Preprocessor(data_io, batch_size=2,
                        subfunctions=subfunctions, prepare_subfunctions=True,
                        prepare_batches=False,
                        analysis="patchwise-crop", patch_shape=(160,160,48))
pp.patchwise_overlap = (80, 80, 24)

# Import standard U-Net architecture and Soft Dice
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft, dice_crossentropy, tversky_loss, tversky_crossentropy, dice_coefficient_loss, dice_soft_loss

# Create a deep learning neural network model with a standard U-Net architecture
model = Neural_Network(preprocessor=pp, loss=tversky_loss, metrics=[dice_soft_loss, dice_crossentropy],
                       batch_queue_size=3, workers=3, learninig_rate=0.0001)

sample_list = data_io.get_indiceslist()
sample_list.sort()

from miscnn.evaluation.cross_validation import cross_validation
from miscnn.evaluation import split_validation
from tensorflow.keras.callbacks import ModelCheckpoint

# prediction
train = sample_list[0:40]
test = sample_list[40:70]

model.load("/home/vali/src/NeuralNetworks/segthor-heart-loss.hdf5")
model.predict(test)