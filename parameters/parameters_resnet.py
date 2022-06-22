from functions.importation import os, keras
from models.resnet_model import Resnet_model

args = {}

# construct paths to data
dataset = "./dataset/..."
chechpoint_dir="./checkpoints/resnet"

# model parameters
args["model"] = Resnet_model
args["batchsize"]=20
args["epochs"]=100
args["do_segmentation"]=False # if False central value should be extract from the groundtruth
args["split_fraction"]=3/5 # proportion used in training
args["learning_rate"]=1e-3
args["optimizer"]=keras.optimizers.Adam
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

if "ice_velocity" in inputs: pass

if "slope" in inputs: pass

if "ice_occupation" in outputs:
    losses["ice_occupation"] = keras.losses.BinaryCrossentropy()
    loss_weights["ice_occupation"] = 1/5
    metrics["ice_occupation"] = [keras.metrics.BinaryAccuracy()]

if "ice_thickness" in outputs:
    losses["ice_thickness"] = keras.losses.MeanSquaredError()
    loss_weights["ice_thickness"] = 4/5
    metrics["ice_thickness"] = [keras.metrics.MeanSquaredError(name="mse")]

args["criterion"] = "mse"

args["inputs_dir"] = inputs_dir
args["groundtruths_dir"] = groundtruths_dir
args["losses"] = losses
args["loss_weights"] = loss_weights
args["metrics"] = metrics