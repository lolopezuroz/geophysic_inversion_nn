from functions.train import train
from functions.glacier_dataset import load_dataset
from training_parameters import args

train_dataset, test_dataset = load_dataset(args)
train(args, train_dataset, test_dataset)