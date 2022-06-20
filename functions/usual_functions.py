from functions.importation import os, gdal, numpy as np

def collect_input(array,x,y,factor,padding,half_size):

    x_start = x*factor-padding
    x_end = x*factor+padding+half_size*2

    y_start = y*factor-padding
    y_end = y*factor+padding+half_size*2

    extract = array[x_start:x_end,y_start:y_end] 
    extract = np.expand_dims(extract,-1) if len(np.shape(extract)) == 2 else extract

    return extract

def exist_directory(path:str) -> str:
    """
    check if "path" represent an existent directory
    if not, it create the full path to it

    path:str

    return:str the path of tested directory
    """
    if not os.path.isdir(path):
        parent_directory = path[:path.rfind("/")]
        exist_directory(parent_directory)
        os.mkdir(path)
    return path

def extract_zone(images,bounds,output_draw):

    gdal_warp_options = gdal.WarpOptions(outputBounds=bounds,dstNodata=0)

    arrays = []
    for name, image_filepath in images.items():

        new_image_filepath = os.path.join(output_draw,name)
        
        raster = gdal.Open(image_filepath)
        raster = gdal.Warp(new_image_filepath,raster,options=gdal_warp_options)

        array = np.copy(raster.ReadAsArray())
        if name=="ice_occupation":
            array[array == 2] = 1/3
            array[array == 3] = 2/3
        
        arrays.append(array)

        del raster
        del array
    
    return arrays