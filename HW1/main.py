from PIL import Image
import numpy
from scipy import io, ndimage
import matplotlib.pyplot as plt



# PART A
def partA():
    image_names = [ "cardinal1.jpg", "cardinal2.jpg",   \
                    "leopard1.jpg", "leopard2.jpg",     \
                    "panda1.jpg", "panda2.jpg"]


    images = [Image.open(name).resize((100,100)).convert(mode="L") for name in image_names]

    # filters consists of 48 filters fo size 49x49
    filters = numpy.array(io.loadmat("filters.mat")["F"]).T
    responses = []
    for j in range(len(filters)):
        fig, axs = plt.subplots(2,4)
        axs[0,0].imshow(filters[j])
        axs[0,1].axis('off')
        filt_response = []
        for i in range(len(images)):
            data = ndimage.convolve(images[i], filters[j])
            filt_response.append(data)
            r = int((i+2)/4)
            c = int((i+2)%4)
            axs[r, c].imshow(data)
            axs[r,c].title.set_text(image_names[i])
        responses.append(filt_response)
        fig.savefig("./filtered/filt" + str(j) + ".png")
        plt.close()

    return images, filters, responses

# PART B

def computeTextureReprs(image, F):
    num_rows, num_cols = image.size
    responses = []
    for filter in F:
        res = ndimage.convolve(image, filter).flatten()
        responses.append(res)
    texture_repr_concat = numpy.concatenate(responses)
    texture_repr_mean = numpy.mean(responses, axis=1)

    # print(texture_repr_mean.shape)
    # print(texture_repr_concat.shape)
    return texture_repr_concat, texture_repr_mean

    return None

def partB():
    image_names = [ "cardinal1.jpg", "cardinal2.jpg",   \
                    "leopard1.jpg", "leopard2.jpg",     \
                    "panda1.jpg", "panda2.jpg"]
    #read images, convert to 100x100 greyscale using PIL
    images = [Image.open(name).resize((100,100)).convert(mode="L") for name in image_names]
    filters = numpy.array(io.loadmat("filters.mat")["F"]).T

    for image in images:
        computeTextureReprs(image, filters)
        quit()


# Part C
def partC():
    image_names = [ "baby_happy.jpg", "baby_weird.jpg"]
    #read images, convert to 100x100 greyscale using PIL
    im1 = Image.open(image_names[0]).resize((512,512)).convert(mode="L")
    im2 = Image.open(image_names[1]).resize((512,512)).convert(mode="L")

    im1_blur = ndimage.gaussian_filter(im1, 7)
    im2_blur = ndimage.gaussian_filter(im2, 8)
    im2_detail = im2_blur - im2
    # axs = plt.imshow(im1_blur, cmap="gray", vmin=0, vmax=255)
    # plt.show()
    # axs = plt.imshow(im2_detail, cmap="gray", vmin=0, vmax=255)
    # plt.show()

    hybrid = im1_blur + im2_detail
    axs = plt.imshow(hybrid, cmap="gray", vmin=0, vmax=255)
    plt.savefig("./hybrid.png")
