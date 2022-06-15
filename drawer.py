import enum
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parameters_resnet import args
from tensorflow import convert_to_tensor

def exist_directory(path:str) -> str:
    """
    check if "path" represent an existent directory
    if not, it create the full path to it

    path:str

    return:str the path of tested directory
    """
    if not os.path.isdir(path):
        parent_directory = path[:path.rfind("/")]
        exist_directory(parent_directory)
        os.mkdir(path)
    return path

def collect_input(array,x,y,factor,padding,half_size):

    x_start = x*factor-padding
    x_end = x*factor+padding+half_size*2

    y_start = y*factor-padding
    y_end = y*factor+padding+half_size*2

    extract = array[x_start:x_end,y_start:y_end] 
    extract = np.expand_dims(extract,-1) if len(np.shape(extract)) == 2 else extract

    return extract

def extract_zone(images,bounds,output_draw):

    gdal_warp_options = gdal.WarpOptions(outputBounds=bounds,dstNodata=0)

    arrays = []
    for name, image_filepath in images.items():

        new_image_filepath = os.path.join(output_draw,name)
        
        raster = gdal.Open(image_filepath)
        raster = gdal.Warp(new_image_filepath,raster,options=gdal_warp_options)

        array = np.copy(raster.ReadAsArray())
        if name=="ice_occupation":
            array[array == 2] = 1/3
            array[array == 3] = 2/3
        
        arrays.append(array)

        del raster
        del array
    
    return arrays

def predict(inputs, model):
    
    n_inputs = len(inputs)

    half_sizes = [10,40]
    padding_sizes = [22,19]
    factors = [half_size//10 for half_size in half_sizes]

    predicted_array = []

    x_size,y_size = np.shape(inputs[0])
    x_start = y_start = padding_sizes[0]
    x_end = x_size - half_sizes[0]*2 - padding_sizes[0]
    y_end = y_size - half_sizes[0]*2 - padding_sizes[0]

    for x in range(x_start,x_end):
        row_extracts = []
        for input_array, f, p, hs in zip(inputs,factors,padding_sizes,half_sizes):
            input_extracts = []
            for y in range(y_start,y_end):
                extract = collect_input(input_array,x,y,f,p,hs)
                input_extracts.append(extract)
            row_extracts.append(convert_to_tensor(input_extracts))
        print(x)
        predicted_batch = np.squeeze(model(row_extracts)).T
        predicted_array.append(predicted_batch)

    predicted_array = np.array(predicted_array)
    return [predicted_array[:,:,i] for i in range(n_inputs)]

def plot(inputs_arrays, predictions_arrays, ground_truths_arrays, output_draw):

    input_symbology = []
    if "ice_velocity" in inputs:
        input_symbology.append(
        {
            "cmap":"magma",
            "vmin":0,
            "vmax":100,
            "title":"Vitesse d'écoulement (m/an)"
        })
    if "slope" in inputs:
        input_symbology.append(
        {
            "cmap":"Reds",
            "vmin":0,
            "vmax":90,
            "title":"Pentes °"
        })
    groundtruth_symbology = []
    if "ice_occupation" in outputs:
        groundtruth_symbology.append({
            "cmap":"bwr",
            "vmin":0.,
            "vmax":1.,
            "title":"Présence de glacier"
        })
    if "ice_thickness" in outputs:
        groundtruth_symbology.append({
            "cmap":"Blues",
            "vmin":0,
            "vmax":500,
            "title":"Épaisseur de glace (m)"})
    
    n_inputs = len(inputs)
    n_ground_truths = len(outputs)

    fig = plt.figure()

    input_axes = [fig.add_subplot(n_inputs, 3, i*3+1) for i in range(n_inputs)] # first column
    prediction_axes = [fig.add_subplot(n_ground_truths, 3, i*3+2) for i in range(n_ground_truths)] # second column
    groundtruth_axes = [fig.add_subplot(n_ground_truths, 3, i*3+3) for i in range(n_ground_truths)] # third column

    for arrays, axes, symbologies in zip([inputs_arrays, predictions_arrays, ground_truths_arrays],
                                         [input_axes, prediction_axes, groundtruth_axes],
                                         [input_symbology, groundtruth_symbology, groundtruth_symbology]):
        print(len(arrays),len(axes),len(symbologies))
        for ax, symbology, array in zip(axes, symbologies, arrays):

            print(np.shape(array))

            im = ax.imshow(array, vmin=symbology["vmin"], vmax=symbology["vmax"], cmap=symbology["cmap"])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im,cax=cax)
            ax.title.set_text(symbology["title"])
            ax.set_axis_off()

    plt.savefig(output_draw+"/fig.png")

images = {}
images["slope"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/slope_swissAlti3d_aligned.tif"
images["ice_velocity"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/V_RGI-11_2021July01_aligned.tif"
images["ice_occupation"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/glacier_class_aligned.tif"
images["ice_thickness"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/IceThickness_aligned.tif"

inputs = args["inputs"]
outputs = args["outputs"]

inputs_paths = {}
if "ice_velocity" in inputs: inputs_paths["ice_velocity"] = images["ice_velocity"]
if "slope" in inputs: inputs_paths["slope"] = images["slope"]

ground_truths_paths = {}
if "ice_occupation" in outputs: ground_truths_paths["ice_occupation"] = images["ice_occupation"]
if "ice_thickness" in outputs: ground_truths_paths["ice_thickness"] = images["ice_thickness"]

models_paths = [args["checkpoint_dir"]] # replace with list of files in directory

#bounds_list = [(2634398,1147741,2634398+20000,1147741+20000)]
bounds_list = [(2634398,1147741,2634398+10000,1147741+10000)]

for i, bounds in enumerate(bounds_list):

    output_draw = "/home/lopezurl/geophysic_inversion/nn/draw_"+str(i)+"_zone"
    exist_directory(output_draw)

    inputs_arrays = extract_zone(inputs_paths,bounds,output_draw)
    ground_truths_arrays = extract_zone(ground_truths_paths,bounds,output_draw)

    for j, model_path in enumerate(models_paths):

        early_fusion = args["early_fusion"]
        model = args["model"](early_fusion,inputs,outputs)
        model.load_weights(model_path).expect_partial() # not meant to be trained

        predictions_arrays = predict(inputs_arrays,model)

        plot(inputs_arrays,predictions_arrays,ground_truths_arrays,output_draw)