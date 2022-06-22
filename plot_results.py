from parameters.parameters_resnet import args
from print_results.deploy_model import deploy_model
from functions.usual_functions import plot

deploy_model(args)
plot()