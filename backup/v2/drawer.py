import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parameters_resnet import args

def collect_input(array,x,y,factor,padding,half_size):

    x_start = x*factor
    x_end = x*factor+2*half_size+2*padding

    y_start = y*factor
    y_end = y*factor+2*half_size+2*padding

    extract = array[x_start:x_end,y_start:y_end] 
    extract = np.expand_dims(extract,-1) if len(np.shape(extract)) == 2 else extract

    return extract

output_draw = "/home/lorenzo/geophysic_inversion/process/nn/draw"

images = {}
images["ice_velocity"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/V_RGI-11_2021July01_aligned.tif"
images["ice_occupation"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/glacier_class_aligned.tif"
images["ice_thickness"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/IceThickness_aligned.tif"
images["slope"] = "/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/slope_swissAlti3d_aligned.tif"

inputs = args["inputs"]
outputs = args["outputs"]

input_symbology = [{"cmap":"magma","vmin":0,"vmax":100,"title":"Vitesse d'écoulement (m/an)"},{"cmap":"Reds","vmin":0,"vmax":90,"title":"Pentes °"}]
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

    print(name,np.shape(array))

    del raster
    del array

inputs = [arrays[name] for name in inputs]

output_array = np.array([arrays[name] for name in outputs])
output_array = np.moveaxis(output_array,0,-1)

predicted_array = np.full((list(np.shape(arrays['ice_velocity']))+[n_outputs]),np.nan)

x_size,y_size = np.shape(predicted_array)[:2]
for x in range(half_size+22,x_size-half_size-22):
    extracts = []
    for y in range(half_size+22,y_size-half_size-22):
        extract = [collect_input(input_array,x,y,f,p,hs) for input_array, f, p, hs in zip(inputs,[1,4],[22,19],[10,40])]
        extracts.append(extract)
    predicted_batch = np.squeeze(model(extracts)).T
    predicted_array[x,half_size:-half_size,:] = predicted_batch

fig = plt.figure()

input_axes = [fig.add_subplot(n_inputs,3,i*3+1) for i in range(n_inputs)]
predicted_axes = [fig.add_subplot(n_outputs,3,i*3+2) for i in range(n_outputs)]
groundtruth_axes = [fig.add_subplot(n_outputs,3,i*3+3) for i in range(n_outputs)]

for array, axes, symbology in zip([inputs, predicted_array, output_array],
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