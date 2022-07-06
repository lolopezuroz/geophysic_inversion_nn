from functions.importation import os,\
                                  datetime,\
                                  gdal,\
                                  ogr,\
                                  numpy as np,\
                                  pyplot as plt,\
                                  make_axes_locatable


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

def copy_extent(array, raster_path, reference_raster_path):
    
    ds = gdal.Open(reference_raster_path)
    driver = gdal.GetDriverByName('GTiff')

    geo_transform = list(ds.GetGeoTransform())
    geo_transform[1] = 50
    geo_transform[5] = -50
    
    [rows,cols] = array.shape
    out_ds=driver.Create(raster_path,cols,rows,1,gdal.GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(ds.GetProjection())
    out_band.FlushCache()

def extract_bounds(shapefile_path):
    
    shapefile = shapefile_path
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()

    extents = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        extent = geom.GetEnvelope()
        extent = [extent[0], extent[2], extent[1], extent[3]] # for whatever reason ogr and gdal don't have the same coordinate order
        extents.append(list(extent))

    return extents


def plot(array, parameters, save_location):

    cmap = parameters["cmap"]
    vmin = parameters["vmin"]
    vmax = parameters["vmax"]
    title = parameters["title"]
    name = parameters["name"]
    extension = parameters["extension"]
    save_location = os.path.join(save_location,f"{name}.{extension}")

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        array,
        vmin = vmin,
        vmax = vmax,
        cmap = cmap
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.title.set_text(title)
    ax.set_axis_off()

    plt.savefig(save_location)
    plt.close()


def get_all_files(directory):
    files = []
    for object in os.listdir(directory):
        object = os.path.join(directory, object)
        if os.path.isdir(object):
            files += get_all_files(object)
        elif os.path.isfile(object):
            files.append(object)
    return files