from PIL import Image
import numpy
from scipy import io, ndimage
import matplotlib.pyplot as plt


image_names = [ "cardinal1.jpg", "cardinal2.jpg",   \
                "leopard1.jpg", "leopard2.jpg",     \
                "panda1.jpg", "panda2.jpg"]


images = [Image.open(name).resize((100,100)).convert(mode="L") for name in image_names]

# filters consists of 48 filters fo size 49x49
filters = numpy.array(io.loadmat("filters.mat")["F"]).T

for j in range(len(filters)):
    fig, axs = plt.subplots(2,4)
    axs[0,0].imshow(filters[j])
    for i in range(len(images)):
        data = ndimage.convolve(images[i], filters[j])
        r = int((i+2)/4)
        c = int((i+2)%4)
        axs[r, c].imshow(data)

    fig.savefig("./filtered/filt" + str(j) + ".png")
    plt.close()
