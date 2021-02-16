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
    Ix = ndimage.convolve1d(image, grad_filter, axis=0)
    Iy = ndimage.convolve1d(image, grad_filter, axis=1)

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

            R[i][j] = numpy.abs(numpy.linalg.det(M) - k*(numpy.trace(M)**2))

    avg_rscore = R.mean()
    top_scores = []

    # find top 1% of key points
    for y in range(cols):
        for x in range(rows):
            score = R[x][y]
            if score > avg_rscore*4:
                # non-max suppresion
                max = True
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if R[(x+i)%rows][(y+j)%cols] > score:
                            max = False
                if max:
                    top_scores.append((x,y, score))

    # fig, axs = plt.subplots(2,3)
    # axs[0,0].imshow(Ix, cmap="gray")
    # axs[0,1].imshow(Iy, cmap="gray")
    # axs[0,2].imshow(R, cmap="gray")
    # axs[1,1].imshow(image_orig, cmap="gray", vmin=0, vmax=255)
    # for x,y,score in top_scores:
    #     rad=numpy.log(score/(5*avg_rscore))
    #     axs[1,1].plot(x,y, 'o', ms=rad*5, mec='g', mfc='none', mew=1)
    # plt.show()

    y = [y for x,y,score in top_scores]
    x = [x for x,y,score in top_scores]
    scores = [score for x,y,score in top_scores]


    return x,y,scores,Ix,Iy

def partD():
    extract_keypoints("house.jpg")
    return None

# partD()
# Part E

def compute_features(x,y,scores, Ix, Iy):
    features = zip(x,y,scores)

    rows, cols = Ix.shape

    descriptors = []
    for x, y, scores in features:
        desc = numpy.zeros(8)
        if x < 5 or y < 5:
            continue
        if x >= cols - 5 or y >= rows - 5:
            continue

        for ii in range(11):
            for jj in range(11):
                try:
                    ix = Ix[x +5 -ii][y+5-jj]
                    iy = Iy[x +5 -ii][y+5-jj]
                except:
                    print(x)
                    print(y)
                m = numpy.sqrt(ix**2 + iy**2)
                # print(m)
                angle = numpy.arctan2(iy, ix)
                # quantize
                # print(angle)
                # print(numpy.floor(numpy.interp(angle, [-numpy.pi, numpy.pi], [0,7])))
                bin = int(numpy.floor(numpy.interp(angle, [0, numpy.pi], [0,7])))
                desc[bin] += m
        # norm, clip and norm
        desc = desc/desc.sum()
        # sorry this is very unreadable but effectively clips in one line
        desc = numpy.array(list(map(lambda el: el if el <= .2 else .2, list(desc))))
        desc = desc/desc.sum()
        descriptors.append(desc)

    return descriptors

def partE():
    x,y,scores,Ix,Iy = extract_keypoints("leopard1.jpg")
    compute_features(x,y,scores,Ix,Iy)

# partE()

# Part F
def computeBOWRepr(features, means):
    bow = numpy.zeros(len(means))
    # print(features)
    for feature in features:
        min_dist = numpy.inf
        closest = 0
        for i in range(len(means)):
            dist = numpy.linalg.norm(feature - means[i])
            if dist < min_dist:
                min_dist = dist
                closest = i
        bow[closest] += 1

    # print(bow)
    # normalize bow
    bow = bow/bow.sum()
    return bow

def partF():
    means = numpy.array(io.loadmat("means_k50.mat")["means"])



    x,y,scores,Ix,Iy = extract_keypoints("cardinal1.jpg")
    features = compute_features(x,y,scores,Ix,Iy)
    bow_card1 = computeBOWRepr(features, means)

    x,y,scores,Ix,Iy = extract_keypoints("cardinal2.jpg")
    features = compute_features(x,y,scores,Ix,Iy)
    bow_card2 = computeBOWRepr(features, means)

    dist_card = numpy.linalg.norm(bow_card1-bow_card2)

    x,y,scores,Ix,Iy = extract_keypoints("panda1.jpg")
    features = compute_features(x,y,scores,Ix,Iy)
    bow_pand1 = computeBOWRepr(features, means)

    x,y,scores,Ix,Iy = extract_keypoints("panda2.jpg")
    features = compute_features(x,y,scores,Ix,Iy)
    bow_pand2 = computeBOWRepr(features, means)

    dist_pand = numpy.linalg.norm(bow_pand1-bow_pand2)

    x,y,scores,Ix,Iy = extract_keypoints("leopard1.jpg")
    features = compute_features(x,y,scores,Ix,Iy)
    bow_leo1 = computeBOWRepr(features, means)

    x,y,scores,Ix,Iy = extract_keypoints("leopard2.jpg")
    features = compute_features(x,y,scores,Ix,Iy)
    bow_leo2 = computeBOWRepr(features, means)

    dist_leo = numpy.linalg.norm(bow_leo1-bow_leo2)


    dist_leocard = numpy.linalg.norm(bow_leo1-bow_card1)
    dist_leopand = numpy.linalg.norm(bow_leo1-bow_pand1)
    dist_pandcard = numpy.linalg.norm(bow_pand1-bow_card1)

    print(dist_card)
    print(dist_pand)
    print(dist_leo)
    print("accross")
    print(dist_leocard)
    print(dist_leopand)
    print(dist_pandcard)

    # print(means)

# partF()

# Part G

def partG():
    image_names = [ ["cardinal1.jpg", "cardinal2.jpg"],   \
                    ["leopard1.jpg", "leopard2.jpg"],     \
                    ["panda1.jpg", "panda2.jpg"]]

    means = numpy.array(io.loadmat("means_k10.mat")["means"])
    filters = numpy.array(io.loadmat("filters.mat")["F"]).T


    within_bow = []
    between_bow = []
    within_concat = []
    between_concat = []
    within_mean = []
    between_mean = []

    bow_reprs = []
    tex_concat_reprs = []
    tex_mean_reprs = []

    for label in image_names:
        cl = []
        con = []
        mea = []
        for image in label:
            x,y,scores,Ix,Iy = extract_keypoints(image)
            features = compute_features(x,y,scores,Ix,Iy)
            bow = computeBOWRepr(features, means)
            concat, mean = computeTextureReprs(image, filters)
            cl.append(bow)
            con.append(concat)
            mea.append(mean)
        bow_reprs.append(cl)
        tex_concat_reprs.append(con)
        tex_mean_reprs.append(mea)

    print("bows found")
    # print(bow_reprs)
    for i in range(len(bow_reprs)):
        label = bow_reprs[i]
        for j in range(len(label)):
            bow = label[j]
            concat = tex_concat_reprs[i][j]
            mean = tex_mean_reprs[i][j]
            # compare to all other bows for in/out class distances
            for ii in range(len(bow_reprs)):
                for jj in range(len(bow_reprs[i])):
                    other_bow = bow_reprs[ii][jj]
                    other_concat = tex_concat_reprs[ii][jj]
                    other_mean = tex_mean_reprs[ii][jj]
                    if i == ii and  j != jj:
                        within_bow.append(numpy.linalg.norm(other_bow-bow))
                        within_concat.append(numpy.linalg.norm(other_concat-concat))
                        within_mean.append(numpy.linalg.norm(other_mean-mean))
                    elif j != jj:
                        between_bow.append(numpy.linalg.norm(other_bow-bow))
                        between_concat.append(numpy.linalg.norm(other_concat-concat))
                        between_mean.append(numpy.linalg.norm(other_mean-mean))


    print(numpy.array(within_bow).mean())
    print(numpy.array(between_bow).mean())
    print(numpy.array(within_concat).mean())
    print(numpy.array(between_concat).mean())
    print(numpy.array(within_mean).mean())
    print(numpy.array(between_mean).mean())

partG()

































































# end
