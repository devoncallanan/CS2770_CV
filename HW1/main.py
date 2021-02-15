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
    image = Image.open(image).resize((100,100)).convert(mode="L")
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
    filters = numpy.array(io.loadmat("filters.mat")["F"]).T

    for image in image_names:
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

# Pard D

def extract_keypoints(image):
    # greyscale
    image_orig = numpy.array(Image.open(image).resize((100,100)).convert(mode="L"), dtype='int64')
    # image_orig = numpy.array(Image.open(image).resize((200,200)).convert(mode="L"))
    image = image_orig# -128
    rows, cols = image.shape
    k = .05
    win_size = 5
    grad_filter = numpy.array([-1, -1, 0, 1, 1])
    # Ix_filter = numpy.array([   [-1, 0, 1],
    #                             [-1, 0, 1],
    #                             [-1, 0, 1]])
    # Iy_filter = numpy.array([   [-1, -1, -1],
    #                             [0, 0, 0],
    #                             [1, 1, 1]])
    # Ix = ndimage.convolve(image, Ix_filter)
    # Iy = ndimage.convolve(image, Iy_filter)
    Ix = ndimage.convolve1d(image, grad_filter, axis=0)
    Iy = ndimage.convolve1d(image, grad_filter, axis=1)
    # print(Ix)
    # print(Iy)

    # quit()

    R = numpy.zeros((rows,cols))

    # iterate throught image to compute r score at each pixel
    for i in range(rows):
        for j in range(cols):
            # if a pixel is less than 2 pixels from the top/left or 2 pixels
            # from the bottom/right of the image, set its R score to 0
            if i < 2 or j < 2:
                R[i][j] = 0
                continue
            if i >= rows-2 or j >= cols-2:
                R[i][j] = 0
                continue
            # compute matrix for each pixel in window
            M = numpy.zeros((2,2))
            for ii in range(win_size):
                for jj in range(win_size):
                    ix = Ix[i-2 + ii][j - 2 + jj]
                    iy = Iy[i-2 + ii][j - 2 + jj]
                    M += numpy.array(   [[ix*ix, ix*iy],
                                        [ix*iy, iy*iy]])
            # print(ix)
            # print(iy)
            # print(M)
            # print(numpy.linalg.det(M))
            # print(k*(numpy.trace(M)**2))
            R[i][j] = numpy.linalg.det(M) - k*(numpy.trace(M)**2)

    avg_rscore = R.mean()
    # print(avg_rscore)
    top_scores = []

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(Ix, cmap="gray")
    axs[0,1].imshow(Iy, cmap="gray")
    axs[0,2].imshow(R, cmap="gray")

    # find top 1% of key points
    for x in range(rows):
        for y in range(cols):
            score = R[x][y]
            if score > avg_rscore*5:
                # non-max suppresion
                max = True
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if R[(x+i)%rows][(y+j)%cols] > score:
                            max = False
                if max:
                    top_scores.append((y,x, score))

    axs[1,1].imshow(image_orig, cmap="gray", vmin=0, vmax=255)
    for x,y,score in top_scores:
        rad=numpy.log(score/(5*avg_rscore))
        axs[1,1].plot(x,y, 'o', ms=rad*5, mec='g', mfc='none', mew=1)
    plt.show()



    return None

def partD():
    extract_keypoints("panda1.jpg")
    return None

partD()





































































# end
