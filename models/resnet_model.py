from warnings import filters
from functions.importation import keras, Tensor

#ortho_regularizer = keras.regularizers.OrthogonalRegularizer(factor=0.01)

def identity_block(x: Tensor, filters: list) -> Tensor:
    """
    the convolutional part of a resnet block

    x: Tensor
    filters: list[int] number of dimensions used in each convolution

    return: Tensor
    """
    for i, filter in enumerate(filters):
        activation = "relu" if i+1 != len(filters) else None  # no activation on last convolution (will occur later)
        layer = keras.layers.Conv2D(filters = filter,
                                    kernel_size = 3,
                                    strides = (1 , 1),  # cant use stride for pooling because wont accomodate with residuals
                                    activation = activation,
                                    kernel_regularizer = "l2")
        x = layer(x)
    return x

def shortcut(x:Tensor, out_dim:int, cropping:int) -> Tensor:
    """
    the adaptation of residuals part of a resnet block

    x: Tensor
    out_dim: int the dimension which residuals needs to atteign
    cropping: int number of edges to drop due to convolution (should be equal to number of convolutions in the resnet block)

    return: Tensor
    """
    layer = keras.layers.Conv2D(filters = out_dim,
                                kernel_size = 1,
                                strides = (1 , 1),
                                activation = None,  # activation will occur after addition of convolutional product and residuals
                                kernel_regularizer = "l2")
    cropping = keras.layers.Cropping2D(cropping = cropping)
    
    x = layer(x)
    x = cropping(x)
    return x

def resnet_block(x:Tensor, filters:list) -> Tensor:
    """
    major component of a resnet

    x: Tensor
    filters: list[int]

    return: Tensor
    """
    x_a = identity_block(x, filters)  # convolutional part
    x_b = shortcut(x, filters[-1], len(filters))  # adaptation of residuals
    x = keras.layers.Add()([x_a, x_b])  # fuse product of convolutions and residuals
    x = keras.layers.ReLU()(x)  # activation that was skipped for both

    return x

def resnet_blocks(x: Tensor,
                  blocks_filters: list,
                  n_upsample: int
                  ) -> Tensor:
    """
    a serie of resnet blocks

    x: Tensor
    blocks_filters: list[list[int]] list of the list of filters for each block

    return: Tensor
    """
    for i, filters in enumerate(blocks_filters):
        x = resnet_block(x, filters)
        if i < n_upsample:  # if upsample needs to be applied
            x = keras.layers.MaxPool2D()(x)
    return x

def resnet_denses(x: Tensor,
                  denses_filters: list,
                  last_activation: str = "relu"
                  ) -> Tensor:
    """
    the denses of resnet employed before exiting the model

    x: Tensor
    dense_filters: list[int]
    last_activation: str what type of activation to use before exiting model (relu by default)

    return: Tensor
    """
    x = keras.layers.Flatten()(x)  # turn 3D tensor into 1D tensor
    for i, dense_filter in enumerate(denses_filters):
        activation = "relu" if i+1 != len(denses_filters) else last_activation  # to adapt to product type (regression or classification) last activation can depend
        layer = keras.layers.Dense(units = dense_filter,
                                   activation = activation,
                                   use_bias = False)
        x = layer(x)
    return x

def resnet_model(early_fusion: bool,
                 inputs_names: list,
                 inputs_shapes: dict,
                 outputs_names: list,
                 args: dict) -> keras.Model:
    """
    build a resnet model with multiple inputs and outputs

    early_fusion: bool if every inputs needs to be treated the same or sperataley (in their own branches)
    inputs_names: list[str] name of inputs to be used
    outputs_names: list[str] name of outputs to be used
    args:dict{str: dict} see keys below
        "early_fusion": bool
        "inputs": list[str]
        "outputs": list[str]
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
        y = resnet_blocks(y,
                          args["blocks_filters"]["early_fusion"],
                          args["upsamples"]["late_fusion"])

    # late fusion pre processings
    else:
        processed_inputs = []
        for input_name, input in inputs.items():
            processed_inputs.append(resnet_blocks(input,
                                                  args["blocks_filters"][input_name],
                                                  args["upsamples"][input_name]))
        
        y = keras.layers.Add()(processed_inputs)  # /!\ use of addition is questionable
        
    # main branch processing
    y = resnet_blocks(y,
                      args["blocks_filters"]["main"],
                      args["upsamples"]["main"])
    
    # outputs post processings
    outputs = {}
    for output_name in outputs_names:
        outputs[output_name] = resnet_blocks(
            y,
            args["blocks_filters"][output_name],
            args["upsamples"][output_name]
        )
        outputs[output_name] = resnet_denses(
            outputs[output_name],
            args["denses_filters"][output_name],
            args["last_activations"][output_name]
        )

    model = keras.Model(inputs = inputs,
                        outputs = outputs,
                        name = "resnet")

    return model