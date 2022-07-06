from functions.importation import tensorflow as tf, os, numpy as np, Image, DataFrame
from functions.usual_functions import extract_center

# is this alright ?
seed = 0
rng = tf.random.Generator.from_seed(seed, alg='philox')


def data_augmentation(x, seed):
    x_rot = tf.image.rot90(x, tf.random.uniform([], 0, 4, tf.int32))
    x_flip = tf.image.stateless_random_flip_left_right(x_rot, seed)
    return x_flip


def load_dataset(dataset_name, inputs, outputs) -> tf.data.Dataset:
    """
    create the dataset from directories and according to parameters
    provided in args

    args:dictionnary
        inputs:list
        outputs:list
        do_segmentation:boolean is ground truth a scalar or an array
        batchsize:int
        split_fraction:float how much to use for training

    returns:tuple training dataset and test dataset
    """
    
    def decoder(sample: dict) -> dict:
        """
        function that yield dataset samples

        sample:dict product name as key and value (scalar or tensor) as entry

        return:tuple
        """

        xs = {}
        ys = {}

        seed = rng.make_seeds(2)[0] # generate a random seed each iteration
        for name, value in sample.items():

            if name in inputs: xs[name] = data_augmentation(value,seed)
            else:
                if do_segmentation: 
                    ys[name] = data_augmentation(value,seed)
                else: 
                    ys[name] = value

        return (xs, ys)

    dataset_dir = os.path.join("./data/datasets",dataset_name)

    do_segmentation = False

    data_dict={}
    for item_name in inputs+outputs:
        data_dict[item_name] = {}
        item_dir = os.path.join(dataset_dir, item_name)
        for file in os.listdir(item_dir):
            if file.split('.')[-1] != "tif": continue
            number=file.split('_')[-1].split('.')[0] # extract tile number
            data_dict[item_name][number]=os.path.join(item_dir, file)
    
    df = DataFrame.from_dict(data_dict).fillna(False)
    df = df[df.all(axis=1)]

    slices = {name: [] for name in inputs + outputs}

    for _, files in df.iterrows():
        for product_name in (inputs+outputs):
            
            file = files[product_name]
            image = Image.open(file)
            array = np.array(image)
            
            if not do_segmentation and not product_name in inputs:
                center = extract_center(array)
                mean_value = np.mean(center)
                value = mean_value
            else:
                array = np.nan_to_num(array)
                if len(np.shape(array)) < 3:
                    array = np.expand_dims(array, axis = -1)  # add channel dimension if empty
                value = np.copy(array)
            
            slices[product_name].append(value)

    dataset = tf.data.Dataset.from_tensor_slices(slices)

    return dataset.map(decoder, num_parallel_calls = 10)