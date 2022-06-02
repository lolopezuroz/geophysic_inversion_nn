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
                                        strides=(1,1),
                                        activation="relu" if i+1<len(filters) else None,
                                        kernel_regularizer='l2')
                            for i, i_filters in enumerate(filters)]
            return Sequential(convolutions)

        def shortcut(filters) -> layers.Conv2D:
            convolution = layers.Conv2D(filters=filters,
                                        kernel_size=1,
                                        strides=(1,1),
                                        activation=None,
                                        kernel_regularizer='l2')
            return convolution

        def __init__(self, filters, upsample=False, *args, **kwargs) -> None:
            super(Resnet_model.Resnet_block,self).__init__(*args, **kwargs)
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

    ### static methods
    
    add = layers.Add()
    rl = layers.ReLU()
    mp = layers.MaxPool2D()
    
    def resnet_denses(units,activations) -> Sequential:
        denses = [layers.Dense(units, activation=activation, use_bias=False) for units, activation in zip(units,activations)]
        return Sequential([layers.Flatten()]+denses)

    def __init__(self, selected_inputs, selected_outputs, *args, **kwargs) -> None:
        super(Resnet_model, self).__init__(*args, **kwargs)

        ice_velocity_processes = None
        if "ice_velocity" in selected_inputs:
            ice_velocity_filters = [[16,16],[16,16]]
            blocks = [Resnet_model.Resnet_block(filters, up) for filters, up in zip(ice_velocity_filters,[True,False])]
            ice_velocity_processes = Sequential(blocks)
        
        slope_processes = None
        if "slope" in selected_inputs:
            slope_filters = [[16,16],[16,16]]
            blocks = [Resnet_model.Resnet_block(filters, up) for filters, up in zip(slope_filters,[True,True])]
            slope_processes = Sequential(blocks)
        
        ice_thickness_processes = None
        if "ice_thickness" in selected_outputs:
            ice_thickness_filters = [[32,32],[32,32]]
            blocks = [Resnet_model.Resnet_block(filters, up) for filters, up in zip(ice_thickness_filters,[True,False])]
            
            ice_thickness_units = [128,1]
            ice_thickness_activations = ["relu","relu"]
            denses = [Resnet_model.resnet_denses(ice_thickness_units, ice_thickness_activations)]
            
            ice_thickness_processes = Sequential(blocks+denses)
            
        ice_occupation_processes = None
        if "ice_occupation" in selected_outputs:
            ice_occupation_filters = [[32,32],[32,32]]
            blocks = [Resnet_model.Resnet_block(filters,up) for filters, up in zip(ice_occupation_filters,[True,False])]
            
            ice_occupation_units = [128,1]
            ice_occupation_activations = ["relu","sigmoid"]
            denses = [Resnet_model.resnet_denses(ice_occupation_units, ice_occupation_activations)]
            
            ice_occupation_processes = Sequential(blocks+denses)
        
        self.input_processes = []
        self.input_processes.append(ice_velocity_processes) if ice_velocity_processes else None
        self.input_processes.append(slope_processes) if slope_processes else None
        
        main_filters = [[32,32]]
        main_blocks = [Resnet_model.Resnet_block(filters) for filters in main_filters]
        self.main_processes = Sequential(main_blocks)

        self.output_processes = []
        self.output_processes.append(ice_occupation_processes) if ice_occupation_processes else None
        self.output_processes.append(ice_thickness_processes) if ice_thickness_processes else None
        
    def call(self,inputs,training=None, **kwargs) -> list:
        processed_inputs = [input_process(input) for input, input_process in zip(inputs, self.input_processes)]
        x = Resnet_model.add(processed_inputs) if len(processed_inputs) != 1 else processed_inputs[0]
        x = self.main_processes(x)
        outputs = [output_process(x) for output_process in self.output_processes]
        return outputs