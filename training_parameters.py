from functions.importation import keras
from models.resnet_model import Resnet_model
from models.vgg_model import Vgg_Model

args = {}

# construct paths to data

# model parameters
args["dataset"] = "v1"
args["model"] = Resnet_model
args["early_fusion"] = False
args["batchsize"] = 20
args["epochs"] = 100
args["do_segmentation"] = False #  if False central value should be extract from the groundtruth
args["split_fraction"] = 3/5 #  proportion used in training
args["learning_rate"] = 1e-3
args["optimizer"] = keras.optimizers.Adam
args["criterion"] = "mse"

evaluation_parameters = {
    "ice_thickness": {
        "loss": keras.losses.MeanSquaredError(),
        "loss_weight": 4/5,
        "metrics": [keras.metrics.MeanSquaredError(name="mse")]
        },
    "ice_occupation": {
        "loss":keras.losses.BinaryCrossentropy(),
        "loss_weight": 1/5,
        "metrics":[keras.metrics.BinaryAccuracy()]
    }
}

inputs = ["ice_velocity", "slope"]
outputs = ["ice_occupation", "ice_thickness"]

args["inputs"] = inputs
args["outputs"] = outputs
args["losses"] = [evaluation_parameters[name]["loss"] for name in outputs]
args["loss_weights"] = [evaluation_parameters[name]["loss_weight"] for name in outputs]
args["metrics"] = [evaluation_parameters[name]["metrics"] for name in outputs]