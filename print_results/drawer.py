from functions.importation import datetime,\
                                  pyplot as plt,\
                                  make_axes_locatable

from functions.predict import predict
from functions.usual_functions import exist_directory, extract_zone

from parameters.parameters_resnet import args

def plot(array, symbology, output_draw):

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)

    im = ax.imshow(array, vmin=symbology["vmin"], vmax=symbology["vmax"], cmap=symbology["cmap"])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    ax.title.set_text(symbology["title"])
    ax.set_axis_off()

    plt.savefig(output_draw+symbology["name"]+".png")

images = {}
images["slope"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/slope_swissAlti3d_aligned.tif"
images["ice_velocity"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/V_RGI-11_2021July01_aligned.tif"
images["ice_occupation"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/glacier_class_aligned.tif"
images["ice_thickness"] = "/home/lopezurl/geophysic_inversion/create_dataset/input_data/IceThickness_aligned.tif"

inputs = args["inputs"]
outputs = args["outputs"]

inputs_paths = {}
if "ice_velocity" in inputs: inputs_paths["ice_velocity"] = images["ice_velocity"]
if "slope" in inputs: inputs_paths["slope"] = images["slope"]

ground_truths_paths = {}
if "ice_occupation" in outputs: ground_truths_paths["ice_occupation"] = images["ice_occupation"]
if "ice_thickness" in outputs: ground_truths_paths["ice_thickness"] = images["ice_thickness"]

models_paths = [args["checkpoint_dir"]] # replace with list of files in directory

#bounds_list = [(2634398,1147741,2634398+5000,1147741+5000)]
bounds_list = [(2634398,1147741,2634398+12000,1147741+12000)]

inputs_symbologies = []
if "ice_velocity" in inputs:
    inputs_symbologies.append(
    {
        "name":"ice_velocity",
        "cmap":"magma",
        "vmin":0,
        "vmax":100,
        "title":"Vitesse d'écoulement (m/an)"
    })
if "slope" in inputs:
    inputs_symbologies.append(
    {
        "name":"slope",
        "cmap":"Reds",
        "vmin":0,
        "vmax":90,
        "title":"Pentes °"
    })
groundtruths_symbologies = []
if "ice_occupation" in outputs:
    groundtruths_symbologies.append({
        "name":"ice_occupation",
        "cmap":"bwr",
        "vmin":0.,
        "vmax":1.,
        "title":"Présence de glacier"
    })
if "ice_thickness" in outputs:
    groundtruths_symbologies.append({
        "name":"ice_thickness",
        "cmap":"Blues",
        "vmin":0,
        "vmax":500,
        "title":"Épaisseur de glace (m)"})

for i, bounds in enumerate(bounds_list):

    draw_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    output_draw = "/home/lopezurl/geophysic_inversion/nn/figures/"+draw_id+"/zone_"+str(i+1)+"/"
    exist_directory(output_draw[:-1])

    inputs_arrays = extract_zone(inputs_paths,bounds,output_draw)
    ground_truths_arrays = extract_zone(ground_truths_paths,bounds,output_draw)

    for array, symbology in zip(inputs_arrays+ground_truths_arrays,
                                   inputs_symbologies+groundtruths_symbologies):
        plot(array, symbology, output_draw)

    for j, model_path in enumerate(models_paths):

        early_fusion = args["early_fusion"]
        model = args["model"](early_fusion,inputs,outputs)
        model.load_weights(model_path).expect_partial() # not meant to be trained

        predictions_arrays = predict(inputs_arrays,model)

        for array, symbology in zip(predictions_arrays, groundtruths_symbologies):
            plot(array, symbology, output_draw+"predicted_")