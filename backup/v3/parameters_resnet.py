import os
import tensorflow as tf
from resnet_model import Resnet_model

# construct paths to data
data_dir = "/home/lopezurl/geophysic_inversion/create_dataset/output_data"
ice_velocity_dir = os.path.join(data_dir,'V_RGI-11_2021July01_tiles')
ice_occupation_dir = os.path.join(data_dir,'glacier_class')
ice_thickness_dir = os.path.join(data_dir,'IceThickness_tiles')
ice_thickness_uncertainty_minus_dir = os.path.join(data_dir,'IceThicknessUncertaintyMinus_tiles')
ice_thickness_uncertainty_plus_dir = os.path.join(data_dir,'IceThicknessUncertaintyPlus_tiles')
slope_dir = os.path.join(data_dir,'slope_swissAlti3d')
chechpoint_dir='/home/lopezurl/geophysic_inversion/nn/checkpoint/resnet'

args = {}

# model parameters
args["model"] = Resnet_model
args["batchsize"]=20
args["epochs"]=100
args["do_segmentation"]=False # if False central value should be extract from the groundtruth
args["split_fraction"]=3/5 # proportion used in training
args["learning_rate"]=1e-3
args["optimizer"]=tf.keras.optimizers.Adam
args["checkpoint_dir"]=chechpoint_dir # where to save the model

inputs = ["ice_velocity","slope"]
outputs = ["ice_occupation","ice_thickness"]

args["inputs"] = inputs
args["outputs"] = outputs

inputs_dir = {}
groundtruths_dir = {}
losses = {}
metrics = {}
loss_weights = {}

args["early_fusion"] = False

if "ice_velocity" in inputs:
    inputs_dir["ice_velocity"] = ice_velocity_dir

if "slope" in inputs:
    inputs_dir["slope"] = slope_dir

if "ice_occupation" in outputs:
    groundtruths_dir["ice_occupation"] = ice_occupation_dir
    losses["ice_occupation"] = tf.keras.losses.BinaryCrossentropy()
    loss_weights["ice_occupation"] = 1/5
    metrics["ice_occupation"] = [tf.keras.metrics.BinaryAccuracy()]

if "ice_thickness" in outputs:
    groundtruths_dir["ice_thickness"] = ice_thickness_dir
    losses["ice_thickness"] = tf.keras.losses.MeanSquaredError()
    loss_weights["ice_thickness"] = 4/5
    metrics["ice_thickness"] = [tf.keras.metrics.MeanSquaredError(name="mse")]

args["criterion"] = "mse"

args["inputs_dir"] = inputs_dir
args["groundtruths_dir"] = groundtruths_dir
args["losses"] = losses
args["loss_weights"] = loss_weights
args["metrics"] = metrics