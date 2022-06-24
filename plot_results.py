from training_parameters import args
from print_results.deploy_model import deploy_model
from functions.usual_functions import plot, get_all_files
from print_results.graphic_parameters import parameters
from functions.importation import copy, gdal

shapefile_path = ""
save_location = ""

deployment_directory = deploy_model(args, shapefile_path, save_location) # perform prediction on given areas
files = get_all_files(deployment_directory) # get all files created from the function above
for file in files:
    name, extension = file.split("/")[-1].split(".")
    if extension != ".tif": continue # only perform plot for tif images
    array = gdal.Open(file).ReadAsArray() # array to plot

    # separate the file name into specific terms
    residual_name = copy.copy(name)
    residual_name, model_name = residual_name.split("model_") if "model_" in residual_name else residual_name, False
    residual_name, zone_id = residual_name.split("zone_") if "zone_" in residual_name else residual_name, False
    data_name = residual_name

    for key, value in parameters.items(): # iterate through graphical parameters availables
        if key in data_name: # the image represent 
            copy_param = copy.deepcopy(parameters)

            if "predicted" in data_name: 
                copy_param.title = f"Predicted {copy_param['title'].lower()}"
            if zone_id: 
                copy_param.title += f" zone {zone_id}"
            if model_name: 
                copy_param.title += f" model {model_name}"

            plot(array, copy_param)
            break