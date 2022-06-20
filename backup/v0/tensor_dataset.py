from PIL import Image
import os
import pandas
import numpy as np
import tensorflow as tf

def extract_center(img,size):
    #size=1 for 2x2
    y,x = img.shape
    start = x//2-size
    return img[start:start+2*size,start:start+2*size]

def extract_number(file):
    return file.split('_')[-1].split('.')[0]

seed = 36
rng = tf.random.Generator.from_seed(seed, alg='philox')

def decode(sample):
    ice_velocity=sample['ice_velocity']
    ice_occupation=sample["ice_occupation"]
    ice_thickness=sample["ice_thickness"]

    seed = rng.make_seeds(2)[0]

    ice_velocity_rot=tf.image.rot90(ice_velocity,tf.random.uniform([],0,4,tf.int32))
    ice_velocity_flipped=tf.image.stateless_random_flip_left_right(ice_velocity_rot,seed)

    return (ice_velocity_flipped, ice_occupation, ice_thickness)

def load_dataset(args) -> tf.data.Dataset:

    groundtruths = ["ice_occupation",
                    "ice_thickness",
                    "ice_thickness_uncertainty_minus",
                    "ice_thickness_uncertainty_plus"]

    do_segmentation = args["do_segmentation"]
    products_dir = args["products_dir"]
    batchsize = args["batchsize"]
    split_fraction = args["split_fraction"]

    data_dict={}
    for product, product_dir in products_dir.items():
        data_dict[product] = {}
        for file in os.listdir(product_dir):
            if file.split('.')[-1] != "tif": continue
            number=extract_number(file)
            data_dict[product][number]=os.path.join(product_dir,file)
    
    df = pandas.DataFrame.from_dict(data_dict).fillna(False)
    paths = df[df.all(axis=1)]

    slice_dataset = {}
    for product, _ in products_dir.items():
        product_elements = []

        if not do_segmentation and (product in groundtruths):
            if product == "ice_occupation":
                labels = []
                certainties = []
                for file in paths[product]:
                    array = np.array(Image.open(file))
                    center = extract_center(array,1)
                    center = list(center.flatten().astype(int))
                    max_value = max(set(center),key=center.count)
                    if max_value   ==0:
                        label = [1,0]
                        certainty = 1
                    elif max_value ==1:
                        label = [0,1]
                        certainty = 1
                    elif max_value ==2:
                        label = [1,0]
                        certainty = 1/3
                    elif max_value ==3:
                        label = [0,1]
                        certainty = 2/3
                    labels.append(label)
                    certainties.append(certainty)
                slice_dataset["ice_occupation"] = labels
                slice_dataset["ice_occupation_certainty"] = certainties
            
            else:
                for file in paths[product]:
                    array = np.nan_to_num(np.array(Image.open(file)))
                    center = extract_center(array,1)
                    mean_value = np.mean(center)
                    product_elements.append(mean_value)
                slice_dataset[product] = product_elements

        else:
            for file in paths[product]:
                array = np.nan_to_num(np.array(Image.open(file)))
                product_elements.append(np.expand_dims(array,axis=2))
            slice_dataset[product] = product_elements
    
    dataset = tf.data.Dataset.from_tensor_slices(slice_dataset)
    dataset = dataset.map(decode)
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(batchsize)

    dataset_length = len(dataset)

    train_dataset = dataset.take(int(dataset_length*split_fraction))
    test_dataset = dataset.skip(int(dataset_length*split_fraction))

    return train_dataset, test_dataset