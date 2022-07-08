from print_results.deploy_model import deploy_model
from functions.usual_functions import plot, get_all_files, exist_directory
from functions.importation import copy, gdal, os, json
from print_results.graphic_parameters import parameters

shapefile_path = "./print_results/roi/selected_glaciers.shp"
save_location = "./deployments/2022_06_28_11_03_02"


f = open('./train_saves/model_2022_07_06_11_06_04/args.json', 'r', encoding='utf-8')
args = json.loads(f.read())
f.close()

deployment_directory = deploy_model(args, shapefile_path, save_location) # perform prediction on given areas
figures_directory = os.path.join(deployment_directory,"figures")
exist_directory(figures_directory)

files = get_all_files(deployment_directory) # get all files created from the function above

for file in files:

    if file[-3:] != "tif":
        continue

    name, extension = file.split("/")[-1].split(".")
    array = gdal.Open(file).ReadAsArray() # array to plot

    # separate the file name into specific terms
    residual_name = copy.copy(name)
    (residual_name, model_name) = residual_name.split("_model_") if "_model_" in residual_name else (residual_name, False)
    (residual_name, zone_id) = residual_name.split("_zone_") if "_zone_" in residual_name else (residual_name, False)
    data_name = residual_name

    print(data_name)

    for key, value in parameters.items(): # iterate through graphical parameters availables
        if key in data_name: # the image represent 
            value = copy.deepcopy(value)

            if "predicted" in data_name:
                value["title"] = f"Predicted {value['title'].lower()}"
            if zone_id: 
                value["title"] += f" zone {zone_id}"
                value["name"] += f"_zone_{zone_id}"
            if model_name: 
                value["title"] += f" model {model_name}"
                value["name"] += f"_model_{model_name}"

            plot(array, value, figures_directory)
            continue