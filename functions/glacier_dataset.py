from functions.importation import tensorflow as tf, os
from functions.extract_slices import extract_slices

# is this alright ?
seed = 0
rng = tf.random.Generator.from_seed(seed, alg='philox')

def load_dataset(args) -> tf.data.Dataset:
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

    inputs, groundtruths = args["inputs"], args["outputs"]

    do_segmentation = args["do_segmentation"]
    batchsize = args["batchsize"]
    split_fraction = args["split_fraction"]

    slice_dataset = extract_slices(args["dataset"], inputs, groundtruths, do_segmentation)
    
    def decode(sample) -> tuple:
        """
        function that yield dataset samples

        sample:dict product name as key and value (scalar or tensor) as entry

        return:tuple
        """

        yield_input = []
        yield_output = []

        seed = rng.make_seeds(2)[0] # generate a random seed each iteration
        for item_name, item in sample.items():
            if item_name in inputs or do_segmentation:
                # data augmentation for arrays (flipped and rotated)
                item_rot=tf.image.rot90(item, tf.random.uniform([], 0, 4, tf.int32))
                item_flipped=tf.image.stateless_random_flip_left_right(item_rot, seed)
                yield_input.append(item_flipped)
            elif item_name in groundtruths: yield_output.append(item)

        return tuple(yield_input + yield_output)

    with tf.device('/CPU:0'): dataset = tf.data.Dataset.from_tensor_slices(slice_dataset)
    
    dataset_length = len(dataset)
    
    train_dataset = dataset.take(int(dataset_length*split_fraction))
    test_dataset = dataset.skip(int(dataset_length*split_fraction))
    
    train_dataset = train_dataset.map(decode)
    test_dataset = test_dataset.map(decode)
    
    train_dataset = train_dataset.shuffle(len(dataset))
    
    train_dataset = train_dataset.batch(batchsize)
    test_dataset = test_dataset.batch(batchsize)
    
    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)

    # split dataset

    return train_dataset, test_dataset