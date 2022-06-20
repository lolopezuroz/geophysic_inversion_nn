import os
import numpy as np
import matplotlib.pyplot as plt
from vgg_model import Vgg_Model
from osgeo import gdal, ogr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parameters import args

output_draw = "/home/lorenzo/geophysic_inversion/process/nn/draw"

images = {}
images["ice_velocity"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/V_RGI-11_2021July01_aligned.tif"
images["ice_occupation"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/glacier_class_aligned.tif"
images["ice_thickness"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/IceThickness_aligned.tif"

inputs = args["inputs"]
outputs = args["outputs"]

input_symbology = [{"cmap":"magma","vmin":0,"vmax":100,"title":"Vitesse d'écoulement (m/an)"}]
groundtruth_symbology = [{"cmap":"bwr","vmin":0,"vmax":1,"title":"Présence de glacier"},{"cmap":"Blues","vmin":0,"vmax":500,"title":"Épaisseur de glace (m)"}]
predicted_symbology = [{"cmap":"bwr","vmin":0,"vmax":1,"title":"Présence de glacier prédite"},{"cmap":"Blues","vmin":0,"vmax":500,"title":"Épaisseur de glace prédite (m)"}]

n_inputs = len(inputs)
n_outputs = len(outputs)

checkpoint_dir = args["checkpoint_dir"]
model = args["model"](inputs,outputs)
model.load_weights(checkpoint_dir).expect_partial()
half_size = 24

bounds = (2634398,1147741,2659184,1166414)
#### write some code to use a shapefile with n entities

gdal_warp_options = gdal.WarpOptions(outputBounds=bounds,dstNodata=0)

arrays = {}
for name, image_filepath in images.items():

    new_image_filepath = os.path.join(output_draw,name)
    
    raster = gdal.Open(image_filepath)
    raster = gdal.Warp(new_image_filepath,raster,options=gdal_warp_options)

    array = np.copy(raster.ReadAsArray())
    if name=="ice_occupation":
        array[array == 2] = 1/3
        array[array == 3] = 2/3
    arrays[name] = array

    del raster
    del array

input_array = np.array([arrays[name] for name in inputs])
input_array = np.moveaxis(input_array,0,-1)

output_array = np.array([arrays[name] for name in outputs])
output_array = np.moveaxis(output_array,0,-1)

predicted_array = np.full((list(np.shape(input_array)[:2])+[n_outputs]),np.nan)

x_size,y_size = np.shape(input_array)[:2]
for x in range(half_size,x_size-half_size):
    extracts = []
    for y in range(half_size,y_size-half_size):        
        extract = input_array[x-half_size:x+half_size,y-half_size:y+half_size,:]
        extracts.append(extract)
    extracts = np.array(extracts).reshape(len(extracts),half_size*2,half_size*2,n_inputs)

    predicted_batch = np.squeeze(model([extracts])).T

    predicted_array[x,half_size:-half_size,:] = predicted_batch

fig = plt.figure()

input_axes = [fig.add_subplot(n_inputs,3,i*3+1) for i in range(n_inputs)]
predicted_axes = [fig.add_subplot(n_outputs,3,i*3+2) for i in range(n_outputs)]
groundtruth_axes = [fig.add_subplot(n_outputs,3,i*3+3) for i in range(n_outputs)]

for array, axes, symbology in zip([input_array, predicted_array, output_array],
                                  [input_axes, predicted_axes, groundtruth_axes],
                                  [input_symbology, predicted_symbology, groundtruth_symbology]):
    for (i, ax), sym in zip(enumerate(axes),symbology):
        im = ax.imshow(array[:,:,i],vmin=sym["vmin"],vmax=sym["vmax"],cmap=sym["cmap"])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im,cax=cax)
        ax.title.set_text(sym["title"])
        ax.set_axis_off()
plt.show()