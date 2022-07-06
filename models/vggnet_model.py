from functions.importation import keras, Tensor

def vggnet_convolutions(
    x: Tensor,
    filters: list,
    strides: list
) -> Tensor:
    """
    the convolutional part of a vggnet

    x: Tensor
    filters: list[int] number of dimensions used in each convolution
    strides: list[int]

    return: Tensor
    """
    for i, (filter, stride) in enumerate(zip(filters,strides)):
        layer = keras.layers.Conv2D(filters = filter,
                                    kernel_size = 3,
                                    strides = stride,
                                    activation = "relu",
                                    kernel_regularizer = "l2")
        x = layer(x)
    return x

def vggnet_denses(
    x: Tensor,
    denses_filters: list,
    last_activation: str = "relu"
) -> Tensor:
    """
    the denses of vggnet employed before exiting the model

    x: Tensor
    dense_filters: list[int]
    last_activation: str what type of activation to use before exiting model (relu by default)

    return: Tensor
    """
    x = keras.layers.Flatten()(x)  # turn 3D tensor into 1D tensor
    for i, dense_filter in enumerate(denses_filters):
        if i+1 != len(denses_filters):
            activation = "relu"
            use_bias = True
        else:
            activation = last_activation  # to adapt to product type (regression or classification) last activation can depend
            use_bias = False  # no default value
        layer = keras.layers.Dense(units = dense_filter,
                                   activation = activation,
                                   use_bias = use_bias)
        x = layer(x)
    return x

def vggnet_model(early_fusion: bool,
                 inputs_names: list,
                 inputs_shapes: dict,
                 outputs_names: list,
                 args: dict) -> keras.Model:
    """
    build a vggnet model with multiple inputs and outputs

    early_fusion: bool if every inputs needs to be treated the same or sperataley (in their own branches)
    inputs_names: list[str] name of inputs to be used
    outputs_names: list[str] name of outputs to be used
    args:dict{str: dict} see keys below
        "block_filters": dict{str: list[list[int]]}
        "denses_filters": dict{str: list[int]}
        "upsamples": dict{str: int}
        "last_activations": dict{str: str}

    return: keras.Model
    """

    # build inputs template used by model
    inputs = {}
    for input_name in inputs_names:
        input_shape = inputs_shapes[input_name]
        inputs[input_name] = keras.Input(shape = (input_shape, input_shape, 1),
                                         name = input_name)

    # early fusion processing
    if early_fusion:
        y = keras.layers.Concatenate()(list(inputs.values()))
        y = vggnet_convolutions(
            y,
            args["convolutions_filters"]["early_fusion"],
            args["strides"]["early_fusion"]
        )

    # late fusion pre processings
    else:
        processed_inputs = []
        for input_name, input in inputs.items():
            processed_inputs.append(
                vggnet_convolutions(
                    input,
                    args["convolutions_filters"][input_name],
                    args["strides"][input_name]
                )
            )
        
        y = keras.layers.Add()(processed_inputs)  # /!\ use of addition is questionable
        
    # main branch processing
    y = vggnet_convolutions(
        y,
        args["convolutions_filters"]["main"],
        args["strides"]["main"]
    )
    
    # outputs post processings
    outputs = {}
    for output_name in outputs_names:
        outputs[output_name] = vggnet_convolutions(
            y,
            args["blocks_filters"][output_name],
            args["upsamples"][output_name]
        )
        outputs[output_name] = vggnet_denses(
            outputs[output_name],
            args["denses_filters"][output_name],
            args["last_activations"][output_name]
        )

    model = keras.Model(
        inputs = inputs,
        outputs = outputs,
        name = "vggnet"
    )

    return model