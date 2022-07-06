from functions.importation import tensorflow as tf
from models.resnet_model import resnet_model
from models.vggnet_model import vggnet_model

tf.random.set_seed(0)

resnet_args = {
    "early_fusion": [],
    "inputs": [],
    "outputs": [],

    "blocks_filters": {
        "early_fusion": [[16, 16], [16, 16]],
        "ice_velocity": [[16, 16], [16, 16]],
        "slope": [[16, 16], [16, 16]],
        "main": [[32, 32]],
        "ice_thickness": [[32, 32], [32, 32]],
        "ice_occupation": [[32, 32], [32, 32]],
    },

    "denses_filters": {
        "ice_thickness": [128,1],
        "ice_occupation": [128,1],
    },

    "upsamples": {
        "early_fusion": 1,
        "ice_velocity": 1,
        "slope": 2,
        "main": 0,
        "ice_thickness": 1,
        "ice_occupation": 1,
    },

    "last_activations": {
        "ice_thickness": "relu",
        "ice_occupation": "sigmoid",
    },
}

vggnet_args = {

    "convolutions_filters": {
        "early_fusion": [16, 16],
        "ice_velocity": [16, 16],
        "slope": [16, 16],
        "main": [32, 32],
        "ice_thickness": [32, 32, 64, 64],
        "ice_occupation": [32, 32, 64, 64],
    },

    "strides": {
        "early_fusion": [1, 2],
        "ice_velocity": [1, 2],
        "slope": [1, 2],
        "main": [1, 1],
        "ice_thickness": [1, 2, 1, 2],
        "ice_occupation": [1, 2, 1, 2],
    },

    "denses_filters": {
        "ice_thickness": [128, 1],
        "ice_occupation": [128, 1],
    },

    "upsamples": {
        "early_fusion": 1,
        "ice_velocity": 1,
        "slope": 2,
        "main": 0,
        "ice_thickness": 1,
        "ice_occupation": 1,
    },

    "last_activations": {
        "ice_thickness": "relu",
        "ice_occupation": "sigmoid",
    },
}

models_instanciations = {
    "models_functions":{
        "vggnet": vggnet_model,
        "resnet": resnet_model,
    },
    "models_args":{
        "vggnet": vggnet_args,
        "resnet": resnet_args,
    }
}

inputs_names = ["ice_velocity", "slope"]
outputs_names = ["ice_occupation", "ice_thickness"]
early_fusion = False
model_name = "resnet"

dataset_name = "v1"
batch_size = 20
epochs = 100
split_fraction = 3/5
learning_rate = 1e-3
optimizer_name = "adam"
criterion = "val_loss"

losses_names = {
    "ice_thickness": "nan_mae",
    "ice_occupation": "binary_crossentropy"
}

losses_weights = {
    "ice_thickness": 9/10,
    "ice_occupation": 1/10,
}

metrics_names = {
    "ice_occupation": ["binary_accuracy", "binary_crossentropy"],
    "ice_thickness": ["nan_mse"]
}

args = {
    "batch_size": batch_size,
    "epochs": epochs,
    "split_fraction": split_fraction,
    "learning_rate": learning_rate,
    "optimizer_name": optimizer_name,
    "criterion": criterion,
    "dataset_name": dataset_name,
    "model_name": model_name,
    "inputs_names": inputs_names,
    "outputs_names": outputs_names,
    "early_fusion": early_fusion,
    "losses_names": losses_names,
    "losses_weights": losses_weights,
    "metrics_names": metrics_names,

    "tiles_sizes": {"ice_velocity": 64, "slope": 118},
    "padding_sizes": {"ice_velocity": 22, "slope": 19}
}