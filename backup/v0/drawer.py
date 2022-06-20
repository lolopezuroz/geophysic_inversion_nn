from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from vgg_model import InversionModel

checkpoint_dir = '/home/lorenzo/geophysic_inversion/process/nn/checkpoint/vgg'

model = InversionModel()
model.load_weights(checkpoint_dir)

array = np.array(Image.open('/home/lorenzo/geophysic_inversion/process/create_dataset/input_data/V_RGI-11_2021July01_aligned.tif'))
array = array[2600:2900,3100:3400]

def crop_center(img,crop):
    y,x = img.shape
    start = x//2-crop
    return img[start:start+2*crop,start:start+2*crop]

check_image = np.full_like(array,np.nan)
thickness_image = np.full_like(array,np.nan)

seuil_image = np.zeros_like(array)
x_size,y_size = array.shape
for x in range(24,x_size-24):
    extracts = []
    for y in range(24,y_size-24):
        extract = np.nan_to_num(array[x-24:x+24,y-24:y+24])
        center = crop_center(extract,1)
        if (center != 0).any() : seuil_image[x,y] = 1
        extracts.append(extract)
    extracts = np.array(extracts).reshape(len(extracts),48,48,1)
    occupation, thickness = model(extracts)
    #values = np.argmax(values,axis=1)
    occupation = occupation[:,1]
    check_image[x,24:-24] = occupation
    thickness_image[x,24:-24] = thickness[:,0]

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(235)
ax5 = fig.add_subplot(236)

ax1.imshow(np.nan_to_num(array[24:-24,24:-24]),cmap="plasma",vmin=0,vmax=50)
ax2.imshow(check_image[24:-24,24:-24],cmap="seismic",vmin=0,vmax=1)
ax3.imshow(seuil_image[24:-24,24:-24],cmap="seismic",vmin=0,vmax=1)
ax4.imshow(thickness_image[24:-24,24:-24],cmap="Reds",vmin=0)
#ax5.imshow(seuil_image[24:-24,24:-24],cmap="Reds",vmin=0)

ax1.title.set_text("Velocities m/yr")
ax2.title.set_text("Glacier probability predicted")
ax3.title.set_text("Ground truth")
ax4.title.set_text("Glacier thicknesses predicted")
ax5.title.set_text("Glacier thicknesses")

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
ax5.set_axis_off()

plt.show()