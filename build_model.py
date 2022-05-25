from train import train
from parameters import args
from glacier_dataset import load_dataset

train_dataset, test_dataset = load_dataset(args)
train(args,train_dataset,test_dataset)