from subprocess import call
from keras import layers, Model, Sequential
from tensorflow import Tensor

class Resnet_model(Model):
    """
    a vgg model very orientated towards the prediction of glacier thickness
    
    operations applied to optional entries and outputs can be cutted out
    from the model

    entries :
        -ice velocities (main information)
        -slopes (optional)
        -temperatures (optional)

    outputs :
        -ice thickness (main prediction)
        -ice occupation (optional)
    """
    class Resnet_block(Model):

        ### static methods

        def identity_block(filters:list) -> Sequential:
            convolutions = [layers.Conv2D(filters=i_filters,
                                        kernel_size=3,
                                        strides=(1, 1),
                                        activation="relu" if i+1<len(filters) else None,
                                        kernel_regularizer='l2')
                            for i, i_filters in enumerate(filters)]
            return Sequential(convolutions)

        def shortcut(filters) -> layers.Conv2D:
            convolution = layers.Conv2D(filters=filters,
                                        kernel_size=1,
                                        strides=(1, 1),
                                        activation=None,
                                        kernel_regularizer='l2')
            return convolution

        def __init__(self, filters, upsample=False, *args, **kwargs) -> None:
            super(Resnet_model.Resnet_block, self).__init__(*args, **kwargs)
            n = len(filters) # number of convolutions
            self.upsample = upsample
            self.identity_block_ = Resnet_model.Resnet_block.identity_block(filters)
            self.shortcut_ = Sequential([layers.Cropping2D(cropping=((n, n), (n, n))),
            Resnet_model.Resnet_block.shortcut(filters[-1])]) # the channel size adapt to that of the block's output

        def call(self, input, training=None, mask=None) -> Tensor:
            x = input
            y = [self.identity_block_(x), self.shortcut_(x)]
            y = Resnet_model.add(y)
            y = Resnet_model.rl(y)
            y = Resnet_model.mp(y) if self.upsample else y
            return y

    class Resnet_empty(Model):
        def call(self, input, training=None, mask=None) -> Tensor:
            return input

    ### static methods
    
    add = layers.Add()
    rl = layers.ReLU()
    mp = layers.MaxPool2D()
    cat = layers.Concatenate()
    
    def resnet_denses(units, activations) -> Sequential:
        denses = [layers.Dense(units, activation=activation, use_bias=False) for units, activation in zip(units, activations)]
        return Sequential([layers.Flatten()]+denses)

    def input_process(filters,do_pooling):
        blocks = [Resnet_model.Resnet_block(filters, up) for filters, up in zip(filters, do_pooling)]
        return Sequential(blocks)

    def output_process(filters,do_pooling,units,activations):
        blocks = [Resnet_model.Resnet_block(filters, up) for filters, up in zip(filters, do_pooling)]
        denses = [Resnet_model.resnet_denses(units, activations)]
        return Sequential(blocks+denses)

    ### input parameters


    def input_ice_velocity():
        filters = [[16, 16], [16, 16]]
        do_pooling = [True, False]
        processes = Resnet_model.input_process(filters,
                                               do_pooling)
        return processes

    def input_slope():
        filters = [[16, 16], [16, 16]]
        do_pooling = [True, True]
        processes = Resnet_model.input_process(filters,
                                               do_pooling)
        return processes

    ### main parameters

    def early_process():
        filters = [[16, 16], [16, 16]]
        do_pooling = [True, False]
        return Resnet_model.input_process(filters, do_pooling)
    
    def main_process():
        filters = [[32, 32]]
        do_pooling = [False]
        return Resnet_model.input_process(filters, do_pooling)

    ### output parameters

    def output_ice_thickness():
        filters = [[32, 32], [32, 32]]
        do_pooling = [True, False]
        units = [128, 1]
        activations = ["relu", "relu"]
        processes = Resnet_model.output_process(filters,
                                                do_pooling,
                                                units,
                                                activations)
        return processes

    def output_ice_occupation():
        filters = [[32, 32], [32, 32]]
        do_pooling = [True, False]
        units = [128, 1]
        activations = ["relu", "sigmoid"]
        processes = Resnet_model.output_process(filters,
                                                do_pooling,
                                                units,
                                                activations)
        return processes

    def __init__(self, early_fusion, selected_inputs, selected_outputs, *args, **kwargs) -> None:
        super(Resnet_model, self).__init__(*args, **kwargs)
        
        self.input_processes = []
        self.main_processes = []
        self.output_processes = []
        
        if early_fusion:
            # note : using early fusion requires concatenating the data (to same resolution and padding)
            self.input_processes = [Resnet_model.Resnet_empty() for _ in selected_inputs]
            self.fusion = Resnet_model.cat
            self.main_processes += [Resnet_model.early_process()]
        else:
            self.input_processes.append(Resnet_model.input_ice_velocity()) if "ice_velocity" in selected_inputs else None
            self.input_processes.append(Resnet_model.input_slope()) if "slope" in selected_inputs else None
            self.fusion = Resnet_model.add
        
        self.main_processes += [Resnet_model.main_process()]
        self.main_processes = Sequential(Resnet_model.main_process())

        self.output_processes.append(Resnet_model.output_ice_occupation()) if "ice_occupation" in selected_outputs else None
        self.output_processes.append(Resnet_model.output_ice_thickness()) if "ice_thickness" in selected_outputs else None
        
    def call(self, inputs, training=None, **kwargs) -> list:
        processed_inputs = [input_process(input) for input, input_process in zip(inputs, self.input_processes)]
        x = self.fusion(processed_inputs) if len(processed_inputs) != 1 else processed_inputs[0]
        x = self.main_processes(x)
        outputs = [output_process(x) for output_process in self.output_processes]
        return outputs