from PIL import Image
import os
from pandas import DataFrame as Df
import numpy as np
import tensorflow as tf

"""
functions to prepare a dataset for a model to be trained on it by
providing a directory containing tiles

tiles are images with same row and column size (+ even size) that
represent a geographic area
"""

# is this alright ?
seed = 36
rng = tf.random.Generator.from_seed(seed, alg='philox')

def extract_center(array:np.ndarray) -> np.ndarray:
    """
    extract the 2x2xK matrix at the center of an array

    array must have row x columns x channels size, rows and columns
    size must be equals and even

    array:ndarray

    return:ndarray
    """
    x,y = array.shape[:2] # row x columns x channels
    assert y == x and x%2 == 0 # equal size for rows and columns + even size

    start = x//2-1 # row and column position from where to start the extract
    end = start + 2 # end of the extract
    extract = array[start:end,start:end]
    return extract

def extract_number(file:str) -> str:
    """
    extract tile id from file's name

    file:str

    return:str
    """
    return file.split('_')[-1].split('.')[0]

def extract_paths(items_dictionnary:dict) -> Df:
    """
    check tiles where inputs and groundtruths are available

    items_dictionnary:dict contain item name as key and directory as entry

    return:panda.Dataframe
    """
    data_dict={}
    for item_name, item_directory in items_dictionnary.items():
        data_dict[item_name] = {}
        for file in os.listdir(item_directory):
            if file.split('.')[-1] != "tif": continue
            number=extract_number(file)
            data_dict[item_name][number]=os.path.join(item_directory,file)
    
    df = Df.from_dict(data_dict).fillna(False)
    paths = df[df.all(axis=1)]

    return paths

def extract_slices(inputs:dict,groundtruths:dict,do_segmentation:bool=False):
    """
    a very convoluted function that create slices absolutely required for
    tf.data.Dataset.from_tensor_slices()

    inputs:dict name as key and directory path as entry
    groundtruths:dict //
    do_segmentation:bool

    return:dict
    """

    # fuse inputs and groundtruths dictionnaries
    items_dictionnary = dict(**inputs, **groundtruths)

    # dataframe with available products
    paths = extract_paths(items_dictionnary)

    slice_dataset = {}
    for item_name in items_dictionnary:
        product_elements = []
        if not do_segmentation and not item_name in inputs:
            # if segmentation == False and item is not part of groundtruths
            # function have to extract the central value
            if item_name == "ice_occupation":
                # special case with ice_occupation (NOTE:find a better solution)
                labels = [] # label = is a glacier there (0 no, 1 yes)
                certainties = [] # certainty = is groundtruth trustworthy (0. no, 1. yes)
                for file in paths[item_name]:
                    array = np.array(Image.open(file)) # extract array from tif
                    center = extract_center(array)
                    center = list(center.flatten().astype(int)) # turn matrix in list
                    max_value = max(set(center),key=center.count) # what is most representated value (random if equal but rarely the case)
                    if max_value   ==0: # no glacier for certain
                        label = 0
                        certainty = 1
                    elif max_value ==1: # glacier present for certain
                        label = 1
                        certainty = 1
                    elif max_value ==2: # no glacier 1/3 certain
                        label = 0
                        certainty = 1/3
                    elif max_value ==3: # glacier present 2/3 certain
                        label = 1
                        certainty = 2/3
                    labels.append(label)
                    certainties.append(certainty)
                slice_dataset["ice_occupation"] = labels
                slice_dataset["ice_occupation_certainty"] = certainties
            else:
                # groundtruth = mean value at center
                for file in paths[item_name]:
                    array = np.nan_to_num(np.array(Image.open(file)))
                    center = extract_center(array)
                    mean_value = np.mean(center)
                    product_elements.append(mean_value)
                slice_dataset[item_name] = product_elements
        else:
            for file in paths[item_name]:
                array = np.nan_to_num(np.array(Image.open(file)))
                if len(np.shape(array)) < 3:
                    array = np.expand_dims(array,axis=-1) # add channel dimension if empty
                product_elements.append(array)
            slice_dataset[item_name] = product_elements

    return slice_dataset

def load_dataset(args) -> tf.data.Dataset:
    """
    create the dataset from directories and according to parameters
    provided in args

    args:dictionnary
        do_segmentation:boolean is ground truth a scalar or an array
        inputs_dir:list
        groundtruths_dir:list
        batchsize:int
        split_fraction:float how much to use for training

    returns:tuple training dataset and test dataset
    """
    do_segmentation = args["do_segmentation"]
    inputs_dir = args["inputs_dir"]
    groundtruths_dir = args["groundtruths_dir"]
    batchsize = args["batchsize"]
    split_fraction = args["split_fraction"]

    slice_dataset = extract_slices(inputs_dir,groundtruths_dir,do_segmentation)
    
    def decode(sample) -> tuple:
        """
        function that yield dataset samples

        sample:dict product name as key and value (scalar or tensor) as entry

        return:tuple
        """
        inputs = inputs_dir.keys()
        outputs = groundtruths_dir.keys()

        yield_input = []
        yield_output = []

        seed = rng.make_seeds(2)[0] # generate a random seed each iteration
        for item_name, item in sample.items():
            if item_name in inputs or do_segmentation:
                # data augmentation for arrays (flipped and rotated)
                item_rot=tf.image.rot90(item,tf.random.uniform([],0,4,tf.int32))
                item_flipped=tf.image.stateless_random_flip_left_right(item_rot,seed)
                yield_input.append(item_flipped)
            elif item_name in outputs: yield_output.append(item)

        return tuple(yield_input + yield_output)

    dataset = tf.data.Dataset.from_tensor_slices(slice_dataset)
    dataset = dataset.map(decode)
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(batchsize)

    dataset_length = len(dataset)

    # split dataset
    train_dataset = dataset.take(int(dataset_length*split_fraction))
    test_dataset = dataset.skip(int(dataset_length*split_fraction))

    return train_dataset, test_dataset