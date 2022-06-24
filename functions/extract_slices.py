from functions.importation import numpy as np,\
                                  os,\
                                  Image
from functions.usual_functions import extract_center, extract_paths

def extract_slices(dataset:str, inputs:list, groundtruths:list, do_segmentation:bool=False) -> dict:
    """
    a very convoluted function that create slices absolutely required for
    tf.data.Dataset.from_tensor_slices()

    inputs:list
    groundtruths:list
    do_segmentation:bool

    return:dict
    """

    datasets_directory = os.path.join("./data/datasets",dataset)

    inputs_dir = [os.path.join(datasets_directory,input) for input in inputs]
    groundtruths_dir = [os.path.join(datasets_directory,groundtruth) for groundtruth in groundtruths]

    directories = inputs_dir+groundtruths_dir

    # dataframe with available products
    paths = extract_paths(directories)

    slice_dataset = {}
    for product_name, product_directory in zip(inputs+groundtruths, directories):
        
        product_elements = []
        
        if not do_segmentation and not product_name in inputs:
            # if segmentation == False and item is not part of groundtruths
            
            # groundtruth = mean value at center
            for file in paths[product_name]:
                array = np.array(Image.open(file))
                center = extract_center(array)
                mean_value = np.mean(center)
                product_elements.append(mean_value)
            slice_dataset[product_name] = product_elements
            
        else:
            for file in paths[product_name]:
                array = np.nan_to_num(np.array(Image.open(file)))
                if len(np.shape(array)) < 3:
                    array = np.expand_dims(array, axis=-1) # add channel dimension if empty
                product_elements.append(array)
            slice_dataset[product_name] = product_elements

    return slice_dataset