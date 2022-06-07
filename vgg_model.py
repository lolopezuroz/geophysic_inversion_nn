from keras import layers, Model, Sequential


class Vgg_Model(Model):
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

    ### static methods

    def vgg_denses(units,activations):
        denses = [layers.Dense(units, activation=activation) for units, activation in zip(units,activations)]
        return Sequential([layers.Flatten()]+denses)

    def vgg_convolutions(filters,strides):
        convolutions = [layers.Conv2D(filters=filters,
                                      kernel_size=3,
                                      strides=strides,
                                      activation="relu",
                                      kernel_regularizer='l2')
                        for filters, strides in zip(filters,strides)]
        return Sequential(convolutions)
    
    def __init__(self, early_fusion, selected_inputs, selected_outputs, *args, **kwargs):
        super(Vgg_Model, self).__init__(*args, **kwargs)

        self.cat = layers.Concatenate()

        ice_velocity_processes = None
        if "ice_velocity" in selected_inputs:

            ice_velocity_filters = [16,16]
            ice_velocity_strides = [[1,1],[2,2]]

            ice_velocity_processes = Vgg_Model.vgg_convolutions(ice_velocity_filters,ice_velocity_strides)
        
        slope_processes = None
        if "slope" in selected_inputs:

            slope_filters = [16,16]
            slope_strides = [[1,1],[2,2]]

            slope_processes = Vgg_Model.vgg_convolutions(slope_filters,slope_strides)
        
        ice_thickness_processes = None
        if "ice_thickness" in selected_outputs:
            
            ice_thickness_filters = [32,32,64,64]
            ice_thickness_strides = [[1,1],[2,2],[1,1],[2,2]]
            
            ice_thickness_units = [128,1]
            ice_thickness_activation = ["relu","relu"]
            
            ice_thickness_processes = Sequential([Vgg_Model.vgg_convolutions(ice_thickness_filters, ice_thickness_strides),
                                                  Vgg_Model.vgg_denses(ice_thickness_units, ice_thickness_activation)])
            
        ice_occupation_processes = None
        if "ice_occupation" in selected_outputs:
            ice_occupation_filters = [32,32,64,64]
            ice_occupation_strides = [[1,1],[2,2],[1,1],[2,2]]
            
            ice_occupation_units = [128,1]
            ice_occupation_activation = ["relu","sigmoid"]

            ice_occupation_processes = Sequential([Vgg_Model.vgg_convolutions(ice_occupation_filters, ice_occupation_strides),
                                                   Vgg_Model.vgg_denses(ice_occupation_units, ice_occupation_activation)])

        self.input_processes = []
        if ice_velocity_processes:
            self.input_processes.append(ice_velocity_processes)
        if slope_processes:
            self.input_processes.append(slope_processes)
        
        self.main_processes = Sequential([])

        self.output_processes = []
        if ice_occupation_processes:
            self.output_processes.append(ice_occupation_processes)
        if ice_thickness_processes:
            self.output_processes.append(ice_thickness_processes)
        
    def call(self,inputs,training=None, **kwargs):

        # process each input
        processed_inputs = [input_process(input) for input, input_process in zip(inputs, self.input_processes)]

        # data fusion
        x = processed_inputs[0]
        for processed_input in processed_inputs[1:]: x = self.cat(x, processed_input)
        
        # process shared by all data
        x = self.main_processes(x)

        # process each output
        outputs = [output_process(x) for output_process in self.output_processes]
        
        return outputs