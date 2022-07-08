from numpy import ndarray
from functions.importation import os
from functions.usual_functions import exist_directory,\
                                      extract_zone,\
                                      extract_bounds,\
                                      unique_id
from training_parameters import models_instanciations
from functions.importation import numpy as np, convert_to_tensor, gdal


def imitate_extent(
    array: ndarray,
    resolution: float,
    raster_path: str,
    reference_raster_path: str
) -> None:
    """
    save given array to raster by imitating a reference raster extent
    the extent is slightly expanded so that the raster's pixels centers match the intersections of reference raster

    array: ndarray  the array to save as raster
    resolution: float  pixel size of array (in srid coordinates)
    raster_path: str  path to save raster
    reference_raster_path: str  path of reference raster

    return: None
    """
    raster = gdal.Open(reference_raster_path)
    driver = gdal.GetDriverByName('GTiff')

    geo_transform = list(raster.GetGeoTransform())
    
    geo_transform[1] = resolution
    geo_transform[5] = -resolution
    
    [rows,cols] = array.shape
    out_raster = driver.Create(
        raster_path,
        cols,
        rows,
        1,
        gdal.GDT_Float32
    )
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)  # replace array
    out_raster.SetGeoTransform(geo_transform)  # set extent
    out_raster.SetProjection(raster.GetProjection())  # set srid
    out_band.FlushCache()


def collect_input(array, x, y, factor, padding, size):

    x_rectified = x * factor
    y_rectified = y * factor

    x_start = x_rectified - padding
    y_start = y_rectified - padding
    
    x_end = x_start + size
    y_end = y_start + size

    extract = array[x_start:x_end, y_start:y_end]

    return extract


def predict(
    model,
    arrays: dict,
    args: dict
) -> dict:
    """
    perform prediction with given model over given arrays

    model: tf.keras.Model
    arrays: dict{str: ndarray}  arrays to serve as inputs
    args: dict{str}
        "inputs_names": list[str]
        "outputs_names": list[str]
        "tiles_sizes": dict{str: int}  # size of one side of a tile (in pixels units)
        "padding_sizes": dict{str: int}  # length of padding on one side (in pixels units)

    return: dict{str: ndarray}  results of the model
    """
    inputs_names = args["inputs_names"]
    outputs_names = args["outputs_names"]

    tiles_sizes = args["tiles_sizes"]
    padding_sizes = args["padding_sizes"]

    ref_product = "ice_velocity"  # which product use for pixel prediction iteration
    ref_size = tiles_sizes[ref_product]
    ref_padding = padding_sizes[ref_product]  # one side padding
    ref_strip_size = ref_size - ref_padding * 2  # full size of tile

    x_size, y_size = np.shape(arrays[ref_product])  # x and y coordinates of upper left corner extracted (x: line, y: column)

    margin = ref_size // 2 # how much to ignore each margins to have the prediction at the center of the array
    
    x_start = ref_padding
    y_start = ref_padding

    x_end = x_size - ref_strip_size - ref_padding + 1
    y_end = y_size - ref_strip_size - ref_padding + 1

    predicted_arrays_shapes = [x_size + 1, y_size + 1]  # since the prediction occur between corners, add 1 to shift array cells over intersections of ref array cells (+.5 at each side)
    predicted_arrays = {output_name:np.full(predicted_arrays_shapes, np.nan)
                        for output_name in outputs_names}
    for x in range(x_start, x_end):
        print(f"row {x - x_start} over {x_end - x_start}")
        input_extracts = {}
        for input_name in inputs_names:
            input_array = arrays[input_name]
            input_size = tiles_sizes[input_name]
            input_padding = padding_sizes[input_name]
            
            input_strip_size = input_size - input_padding * 2

            input_scale = input_strip_size // ref_strip_size
            
            input_extract = []
            for y in range(y_start, y_end):

                extract = collect_input(
                    input_array,
                    x,
                    y,
                    input_scale,
                    input_padding,
                    input_size
                )
                
                extract = np.expand_dims(extract, -1)
                input_extract.append(extract)
            input_extracts[input_name] = convert_to_tensor(input_extract)

        predicted_batch = model(input_extracts)
        for output_name in outputs_names:
            predicted_row = np.squeeze(predicted_batch[output_name])
            predicted_arrays[output_name][x + ref_strip_size//2, margin : -margin] = predicted_row

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
    
    model_save_dir = os.path.join(args["save_dir"],"models")
    
    # model parameters
    inputs_names = args["inputs_names"]
    inputs_shapes = args["tiles_sizes"]
    outputs_names = args["outputs_names"]
    early_fusion = args["early_fusion"]
    model_name = args["model_name"]
    model_args = args["model_args"]
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

                imitate_extent(output_array, 50, save_path, ref_path)
    
    return save_location