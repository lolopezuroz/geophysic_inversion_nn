from functions.importation import os, json, keras
from functions.usual_functions import exist_directory, unique_id
from functions.load_dataset import load_dataset
from models.compiler_functions import compiler_functions

def fitter(model, args:dict) -> None:
    """
    train, evaluate and record progress of given model on dataset

    args:dict{str: dict} see keys below
        "inputs_names": list[str]
        "outputs_names": list[str]
        "dataset_name": str
        "epochs": int
        "split_fraction": float  # percentage used for training (1-split_fraction for evaluation)
        "batch_size": int
        "criterion": str  # metric used to evaluate best model
        "save_dir": str
    
    no return
    """

    inputs_names = args["inputs_names"]
    outputs_names = args["outputs_names"]
    dataset_name = args["dataset_name"]
    epochs = args["epochs"]
    split_fraction = args["split_fraction"]
    batch_size = args["batch_size"]
    criterion = args["criterion"]
    save_dir = args["save_dir"]

    # load dataset (TODO: create a dataset rather than load one to better fit the situation)
    dataset = load_dataset(dataset_name, inputs_names, outputs_names)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)  # get batches ready on gpu before they are called
    validation_dataset = dataset.skip(int(len(dataset)*split_fraction))  # take 1-split_fraction of dataset for evaluation
    dataset = dataset.take(int(len(dataset)*split_fraction))  # take split_fraction of dataset for training

    logs_dir = os.path.join(save_dir, "logs")  # where logs are saved
    models_dir = os.path.join(save_dir, "models")  # where models weights are saved
    exist_directory(logs_dir)  # make the directory if it dont exist (see exist_directory documentation)
    exist_directory(models_dir)

    tensorboard_callback = keras.callbacks.TensorBoard(  # callback to save logs
    log_dir = logs_dir
    )

    callback_path = os.path.join(models_dir, "model_{epoch:02d}.h5")  # model weights path update with each epoch
    callback = keras.callbacks.ModelCheckpoint(  # callback to save model weights at each epoch
        filepath = callback_path,
        save_weights_only = True,
        verbose = 2
    )
    
    best_callback_path = os.path.join(models_dir, "best.h5")
    best_callback = keras.callbacks.ModelCheckpoint(  # callback to save model weights at best epoch
        filepath = best_callback_path,
        monitor = criterion,  # what metric decide if the model is good
        save_weights_only = True,
        save_best_only = True,
        verbose = 2
    )

    callbacks = [
        callback,
        best_callback,
        tensorboard_callback
    ]

    model.fit(
        x = dataset,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,  # shuffle dataset at each epoch
        workers = 2,
        use_multiprocessing = True,
        validation_data = validation_dataset,
        callbacks = callbacks,
        verbose = 2
    )

def compiler(model,
    args: dict
) -> None:
    """
    compile model before training

    the functions used in evaluation are provided by their name
    first step of the function is to extract these functions from it

    args:dict{str: dict} see keys below
        "losses": dict{str: str}
        "losses_weights": dict{str: float}
        "metrics": dict{str: list[str]}
        "learning_rate": float
        "optimizer": str
    
    no return
    """

    losses_names = args["losses_names"]
    losses_weights = args["losses_weights"]
    metrics_names = args["metrics_names"]
    learning_rate = args["learning_rate"]
    optimizer_name = args["optimizer_name"]
    
    # extract functions based on their names

    losses = losses_names
    for data_name in losses_names:
        losses[data_name] = compiler_functions["losses"][losses[data_name]]

    metrics = metrics_names
    for data_name in metrics_names:
        metrics[data_name] = [
            compiler_functions["metrics"][metric_name]
            for metric_name in metrics[data_name]
        ]

    optimizer = compiler_functions["optimizers"][optimizer_name]

    model.compile(
        loss = losses,
        optimizer = optimizer(learning_rate=learning_rate),
        loss_weights = losses_weights,
        metrics = metrics
    )

def train(
    args: dict,
    models_instanciations
) -> None: 
    """
    call, compile and train model
    
    WOULD NEED POSSIBILITY TO RESUME TRAINING OF GIVEN MODEL

    args
    """

    inputs_names = args["inputs_names"]
    inputs_shapes = args["tiles_sizes"]
    outputs_names = args["outputs_names"]
    early_fusion = args["early_fusion"]
    model_name = args["model_name"]
    model_args = models_instanciations["models_args"][model_name]  # arguments of the model
    model_function = models_instanciations["models_functions"][model_name]  # function that build the model

    # create a new model
    model = model_function(
        inputs_names = inputs_names,
        inputs_shapes = inputs_shapes,
        outputs_names = outputs_names,
        early_fusion = early_fusion,
        args = model_args
    )

    save_dir = os.path.join("./train_saves", f"model_{unique_id()}")  # generate unique directory to save train states and parameters
    exist_directory(save_dir)

    # save model graph
    keras.utils.plot_model(
        model,
        os.path.join(save_dir,f"{model.name}.png"),
        show_shapes = True
    )
    args["save_dir"] = save_dir
    args["model_args"] = model_args
    with open(os.path.join(save_dir,"args.json"), "w") as f:
        json.dump(args, f)  # save parameters

    compiler(model, args)  # compile model
    fitter(model, args)  # fit model