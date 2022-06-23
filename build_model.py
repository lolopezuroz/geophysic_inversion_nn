from functions.train import train
from functions.glacier_dataset import load_dataset

from parameters.parameters_resnet import args

train_dataset, test_dataset = load_dataset(args)
train(args, train_dataset, test_dataset)