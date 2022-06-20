import os
import tensorflow as tf

data_dir = "/home/lorenzo/geophysic_inversion/process/create_dataset/output_data"
ice_velocity_dir = os.path.join(data_dir,'V_RGI-11_2021July01_tiles')
ice_occupation_dir = os.path.join(data_dir,'glacier_class')
ice_thickness = os.path.join(data_dir,'IceThickness_tiles')
ice_thickness_uncertainty_minus = os.path.join(data_dir,'IceThicknessUncertaintyMinus_tiles')
ice_thickness_uncertainty_plus = os.path.join(data_dir,'IceThicknessUncertaintyPlus_tiles')

chechpoint_dir='/home/lorenzo/geophysic_inversion/process/nn/checkpoint/vgg'

args = {}

# model parameters
args["batchsize"]=20
args["epochs"]=30
args["do_segmentation"]=False
args["split_fraction"]=3/5
args["learning_rate"]=1e-3
args["optimizer"]=tf.keras.optimizers.Adam
args["checkpoint_dir"]=chechpoint_dir
args["focus"]=[1/3,2/3]

# products paths
products_dir = {}
products_dir["ice_velocity"] = ice_velocity_dir
#products_dir["slopes"] = os.path.join(data_dir,'')
products_dir["ice_occupation"] = ice_occupation_dir
products_dir["ice_thickness"] = ice_thickness
products_dir["ice_thickness_uncertainty_minus"] = ice_thickness_uncertainty_minus
products_dir["ice_thickness_uncertainty_plus"] = ice_thickness_uncertainty_plus
args["products_dir"] = products_dir

# losses
losses = {}
losses["ice_occupation"] = tf.keras.losses.BinaryCrossentropy()
losses["ice_thickness"] = tf.keras.losses.MeanSquaredError()
args["losses"] = losses

# metrics
metrics = {}
metrics["ice_occupation"] = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.BinaryIoU()]
metrics["ice_thickness"] = [tf.keras.metrics.MeanSquaredError('mse')]
args["metrics"] = metrics