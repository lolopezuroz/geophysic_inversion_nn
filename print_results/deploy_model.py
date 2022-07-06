from functions.importation import os
from functions.usual_functions import exist_directory,\
                                      extract_zone,\
                                      extract_bounds,\
                                      unique_id,\
                                      copy_extent
from models.model_importation import models_instanciations
from functions.importation import numpy as np, convert_to_tensor


def collect_input(array, x, y, factor, padding, strip_size):

    x_start = x*factor-padding
    x_end = x*factor+padding+strip_size

    y_start = y*factor-padding
    y_end = y*factor+padding+strip_size

    extract = array[x_start:x_end, y_start:y_end]

    return extract


def predict(model, arrays, args):

    inputs_names = args["inputs_names"]
    outputs_names = args["outputs_names"]

    tiles_sizes = args["tiles_sizes"]
    padding_sizes = args["padding_sizes"]

    ref_size = min([tiles_sizes[input_name] for input_name in inputs_names])
    ref_product = list(tiles_sizes.keys())[list(tiles_sizes.values()).index(ref_size)]
    ref_padding = padding_sizes[ref_product]
    ref_strip_size = ref_size - ref_padding * 2

    # x and y coordinates of upper left corner extracted (x: line, y: column)
    x_size, y_size = np.shape(arrays[ref_product])
    
    x_start = ref_padding + 1
    y_start = ref_padding + 1

    x_end = x_size - ref_size + 1
    y_end = y_size - ref_size + 1

    prediction_shape = [x_size, y_size]
    predicted_arrays = {output_name:np.full(prediction_shape,np.nan)
                        for output_name in outputs_names}
    for x in range(x_start, x_end):
        print(x)
        input_extracts = {}
        for input_name in inputs_names:
            input_array = arrays[input_name]
            input_size = tiles_sizes[input_name]
            input_padding = padding_sizes[input_name]
            
            input_strip_size = input_size - input_padding * 2

            input_scale = input_strip_size // ref_strip_size
            
            input_extract = []
            for y in range(y_start, y_end):

                extract = collect_input(input_array,
                    x,
                    y,
                    input_scale,
                    input_padding,
                    input_strip_size
                )
                
                extract = np.expand_dims(extract, -1)
                input_extract.append(extract)
            input_extracts[input_name] = convert_to_tensor(input_extract)

        predicted_batch = model(input_extracts)
        for output_name in outputs_names:
            predicted_row = np.squeeze(predicted_batch[output_name])
            predicted_arrays[output_name][x, (y_size - len(predicted_row))//2 : -(y_size - len(predicted_row))//2] = predicted_row

    return predicted_arrays


# hard coded image sources
# would need a wonderfully designed database to extract/patch-together/sub-sample images based on the geometry intersection and inputs required
source_images = {
    "slope": "./data/sources/slope_swissAlti3d_aligned.tif",
    "ice_velocity": "./data/sources/V_RGI-11_2021July01_aligned.tif",
    "ice_occupation": "./data/sources/glacier_class_aligned.tif",
    "ice_thickness": "./data/sources/IceThickness_aligned.tif"
}

def deploy_model(
    args: dict,
    shapefile_path: str,
    save_location: str,
    only_best_model = True
) -> str:
    """
    save input, groundtruth and model prediction on defined regions

    args: dict
        inputs: list data names to be used as inputs
        outputs: list data names to be used as groundtruths
        checkpoint_dir: str directory path where models are saved
        early_fusion: bool is the model in early fusion mode
        model: tensorflow.Model neural network architecture to use
    save_location: str where to save model deployment
    shapefile_path: str shapefile path containing the regions of interest (will be turned to bounding boxes)
    only_best_model: bool print only the predictions from best models if true (will skip them if false)
    
    return: str directory path where is save the model deployment
    """
    
    # unique id to avoid overwriting old deployments (time referenced)
    deployment_id = unique_id()
    save_location = os.path.join(save_location, deployment_id)
    
    model_save_dir = args["save_dir"]
    
    # model parameters
    inputs_names = args["inputs_names"]
    inputs_shapes = args["tiles_sizes"]
    outputs_names = args["outputs_names"]
    early_fusion = args["early_fusion"]
    model_name = args["model_name"]
    model_args = models_instanciations["models_args"][model_name]
    model_function = models_instanciations["models_functions"][model_name]
    model = model_function(
        inputs_names = inputs_names,
        inputs_shapes = inputs_shapes,
        outputs_names = outputs_names,
        early_fusion = early_fusion,
        args = model_args
    )

    bounds_list = extract_bounds(shapefile_path)  # list of bounding boxes's coordinates that represent region of interest

    for i, bounds in enumerate(bounds_list):  # for each region of interest

        save_images = os.path.join(save_location, f"zone_{i+1}")  # where to save the images
        exist_directory(save_images)  # create directory if don't exist

        # extract region of interset from tif for figures and prediction
        
        arrays = {}
        for name in set(inputs_names + outputs_names):
            image_path = source_images[name]
            save_path = os.path.join(save_images, f'{name}_zone_{i+1}.tif')
            arrays[name] = extract_zone(image_path, bounds, save_path)  # save the inputs arrays for prediction

        for file_name in os.listdir(model_save_dir):  # for each model declination
            
            if file_name[-2:] != "h5":
                continue
            
            if (file_name[:4] == "best") == only_best_model:  # if the model is best skip it except if 
                pass
            else:
                continue

            save_images_i_zone = os.path.join(save_images, model_name)
            exist_directory(save_images_i_zone)  # create directory if don't exist

            # import model
            model_path = os.path.join(model_save_dir, file_name)
            model.load_weights(model_path)

            predictions_arrays = predict(model, arrays, args)

            # save prediction
            for output_name, output_array in predictions_arrays.items():
                save_path = os.path.join(
                    save_images_i_zone,
                    f'{output_name}_zone_{i+1}_model_{model_name}.tif'
                )
                ref_path = os.path.join(save_images, f'{output_name}_zone_{i+1}.tif')

                copy_extent(output_array, save_path, ref_path)
    
    return save_location