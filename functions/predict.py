from functions.importation import numpy as np, convert_to_tensor
from functions.usual_functions import collect_input

def predict(inputs, model):
    
    n_inputs = len(inputs)

    half_sizes = [10,40]
    padding_sizes = [22,19]
    factors = [half_size//10 for half_size in half_sizes]

    predicted_array = []

    x_size,y_size = np.shape(inputs[0])
    x_start = y_start = padding_sizes[0]
    x_end = x_size - half_sizes[0]*2 - padding_sizes[0]
    y_end = y_size - half_sizes[0]*2 - padding_sizes[0]

    for x in range(x_start,x_end):
        row_extracts = []
        for input_array, f, p, hs in zip(inputs,factors,padding_sizes,half_sizes):
            input_extracts = []
            for y in range(y_start,y_end):
                extract = collect_input(input_array,x,y,f,p,hs)
                input_extracts.append(extract)
            row_extracts.append(convert_to_tensor(input_extracts))
        print(x)
        predicted_batch = np.squeeze(model(row_extracts)).T
        predicted_array.append(predicted_batch)

    predicted_array = np.array(predicted_array)

    predicted_array = np.pad(predicted_array,
                             [[padding_sizes[0],padding_sizes[0]],
                              [padding_sizes[0],padding_sizes[0]],
                              [0,0]],
                             constant_values=(np.nan))

    return [predicted_array[:,:,i] for i in range(n_inputs)]