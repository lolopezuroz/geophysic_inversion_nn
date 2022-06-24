from functions.importation import os,\
                                  datetime,\
                                  gdal,\
                                  ogr,\
                                  numpy as np,\
                                  pyplot as plt,\
                                  DataFrame,\
                                  make_axes_locatable


def extract_paths(items_dictionnary:list) -> DataFrame:
    """
    check tiles where inputs and groundtruths are available

    items_dictionnary:dict contain item name as key and directory as entry

    return:panda.Dataframe
    """
    data_dict={}
    for item_directory in items_dictionnary:
        item_name = item_directory.split("/")[-1]
        data_dict[item_name] = {}
        for file in os.listdir(item_directory):
            if file.split('.')[-1] != "tif": continue
            number=extract_number(file)
            data_dict[item_name][number]=os.path.join(item_directory, file)
    
    df = DataFrame.from_dict(data_dict).fillna(False)
    paths = df[df.all(axis=1)]

    return paths


def extract_number(file:str) -> str:
    """
    extract tile id from file's name

    file:str

    return:str
    """
    return file.split('_')[-1].split('.')[0]


def extract_center(array:np.ndarray) -> np.ndarray:
    """
    extract the 2x2xK matrix at the center of an array

    array must have row x columns x channels size, rows and columns
    size must be equals and even

    array:ndarray

    return:ndarray
    """
    x, y = array.shape[:2] # row x columns x channels
    assert y == x and x%2 == 0 # equal size for rows and columns + even size

    start = x//2-1 # row and column position from where to start the extract
    end = start + 2 # end of the extract
    extract = array[start:end, start:end]
    return extract


def unique_id():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def collect_input(array, x, y, factor, padding, half_size):

    x_start = x*factor-padding
    x_end = x*factor+padding+half_size*2

    y_start = y*factor-padding
    y_end = y*factor+padding+half_size*2

    extract = array[x_start:x_end, y_start:y_end] 
    extract = np.expand_dims(extract, -1) if len(np.shape(extract)) == 2 else extract

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


def extract_zone(image_filepath, bounds, save_location):
    
    gdal_warp_options = gdal.WarpOptions(outputBounds=bounds, dstNodata=0)
    raster = gdal.Open(image_filepath)
    raster = gdal.Warp(save_location, raster, options=gdal_warp_options)

    array = np.copy(raster.ReadAsArray())

    raster.FlushCache()
    
    return array


def replace_zone(image_filepath, array, save_location):

    copy_raster = gdal.Open(image_filepath)
    new_raster = copy_raster.GetDriver().CreateCopy(save_location, copy_raster)

    new_raster.GetRasterBand(1).WriteArray(array)
    new_raster.FlushCache()


def extract_bounds(shapefile_path):
    
    shapefile = shapefile_path
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()

    extents = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        extent = geom.GetEnvelope()
        extents.append(extent)


def plot(array, parameters, save_location):

    cmap = parameters.cmap
    vmin, vmax = parameters.vmin, parameters.vmax
    title = parameters.title
    name = parameters.name
    extension = parameters.extension
    save_location = os.path.join(save_location,f"{name}.{extension}")

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    im = ax.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.title.set_text(title)
    ax.set_axis_off()

    plt.savefig(save_location)


def get_all_files(directory):
    files = []
    for object in os.listdir(directory):
        object = os.path.join(directory, object)
        if os.path.isdir(object): files += get_all_files(directory)
        elif os.path.isfile(object): files.append(object)
    return files