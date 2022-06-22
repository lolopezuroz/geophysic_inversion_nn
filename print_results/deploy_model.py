from enum import unique
from numpy import save
from functions.importation import os

from functions.predict import predict
from functions.usual_functions import exist_directory,\
                                      extract_zone,\
                                      extract_bounds,\
                                      replace_zone,\
                                      unique_id

# hard coded image sources
# would need a wonderfully designed database to extract/patch-together/sub-sample images based on the geometry intersection and inputs required
source_images = {
    "slope": "/home/lopezurl/geophysic_inversion/create_dataset/input_data/slope_swissAlti3d_aligned.tif",
    "ice_velocity": "/home/lopezurl/geophysic_inversion/create_dataset/input_data/V_RGI-11_2021July01_aligned.tif",
    "ice_occupation": "/home/lopezurl/geophysic_inversion/create_dataset/input_data/glacier_class_aligned.tif",
    "ice_thickness": "/home/lopezurl/geophysic_inversion/create_dataset/input_data/IceThickness_aligned.tif"
}

def deploy_model(args:dict,shapefile_path:str,save_location:str,best_model=True) -> str:
    """
    save input, groundtruth and model prediction on defined regions

    args:dict
        inputs:list data names to be used as inputs
        outputs:list data names to be used as groundtruths
        checkpoint_dir:str directory path where models are saved
        early_fusion:bool is the model in early fusion mode
        model:tensorflow.Model neural network architecture to use
    save_location:str where to save model deployment
    shapefile_path:str shapefile path containing the regions of interest
                       (will be turned to bounding boxes)
    best_model:bool print only the predictions from best models if true
                    (will skip them if false)
    
    return:str directory path where is save the model deployment
    """
    
    # unique id to avoid overwriting old deployments (time referenced)
    deployment_id = unique_id()
    save_location = os.path.join(save_location,deployment_id)

    # data names
    inputs = args["inputs"]
    outputs = args["outputs"]

    # images paths
    inputs_paths = [source_images[name] for name in inputs]
    ground_truths_paths = [source_images[name] for name in outputs]
    
    models_paths = os.listdir(args["checkpoint_dir"]) # directories of model from each epoch and best results

    bounds_list = extract_bounds(shapefile_path) # list of bounding boxes's coordinates that represent region of interest

    for i, bounds in enumerate(bounds_list): # for each region of interest

        save_images = os.path.join(save_location, f"zone_{i+1}") # where to save the images
        exist_directory(save_images) # create directory if don't exist

        # extract region of interset from tif for figures and prediction
        inputs_arrays = []
        for name, image_path in zip(name,inputs_paths):
            save_path = os.path.join(save_images,f'{name}_zone_{i+1}.tif')
            inputs_arrays.append(extract_zone(image_path,bounds,save_path)) # save the inputs arrays for prediction
        for name, image_path in zip(outputs,ground_truths_paths):
            save_path = os.path.join(save_images,f'{name}_zone_{i+1}.tif')
            extract_zone(image_path,bounds,save_path)

        for model_path in models_paths: # for each model declination

            model_name = model_path.split("/")[-1] # extract model name from the directory where it is located
            
            # if the model is best skip it except if 
            if (model_name[:4] == "best") == best_model: pass
            else: continue

            save_images_i_zone = os.path.join(save_images,model_name)
            exist_directory(save_images_i_zone) # create directory if don't exist

            # import model
            early_fusion = args["early_fusion"]
            model = args["model"](early_fusion,inputs,outputs) # model object initialized
            model.load_weights(model_path).expect_partial() # load parameters (.expect_partial() because not meant to be trained)

            predictions_arrays = predict(inputs_arrays,model)

            # save prediction
            for array, name in zip(predictions_arrays,outputs):
                save_path = os.path.join(save_images_i_zone,f'{name}_zone_{i+1}_{model_name}.tif')
                replace_zone(inputs_paths[0],array,save_path) # georeference prediction using a duplicate of input raster
    
    return save_location