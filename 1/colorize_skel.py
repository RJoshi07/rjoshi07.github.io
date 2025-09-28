# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import cv2


# # function which takes 2 images + adds padding 
# def return_score(img_1, img_2, h_dis, w_dis):


#     h, w = img_1.shape
#     w_start = max(0, w_dis)
#     w_end = min(w, w+w_dis)
#     h_start = max(0, h_dis)
#     h_end = min(h, h+h_dis)

#     img_1_cropped = img_1[h_start:h_end, w_start:w_end].copy()
#     img_2_cropped = img_2[h_start-h_dis:h_end-h_dis, w_start-w_dis:w_end-w_dis].copy()

#     return np.sqrt(np.sum((img_1_cropped - img_2_cropped) ** 2))

#     # img_1_norm = np.linalg.norm(img_1_cropped)
#     # img_2_norm = np.linalg.norm(img_2_cropped)

#     # img_1_normalized = (img_1_cropped / img_1_norm).flatten()
#     # img_2_normalized = (img_2_cropped / img_2_norm).flatten()
#     # return np.dot(img_1_normalized, img_2_normalized)



# # name of the input file
# imname = 'church.tif'

# # read in the image
# im = skio.imread(imname)

# # convert to double (might want to do this later on to save memory)    
# im = sk.img_as_float(im)
    
# # compute the height of each part (just 1/3 of total)
# height = np.floor(im.shape[0] / 3.0).astype(np.int64)

# # separate color channels
# b = im[:height]
# g = im[height: 2*height]
# r = im[2*height: 3*height]

# # crop images
# h, w = b.shape
# crop_h = int(0.05 * h)
# crop_w = int(0.05 * w)
# b = b[crop_h: -crop_h, crop_w: -crop_w]
# g = g[crop_h: -crop_h, crop_w: -crop_w]
# r = r[crop_h: -crop_h, crop_w: -crop_w]

# stacked = np.dstack([r, g, b])
# stacked = (stacked * 255).clip(0, 255).astype(np.uint8)

# # save the image
# fname = './out_fname_no_pad.jpg'
# skio.imsave(fname, stacked)

# # display the image
# # skio.imshow(stacked)
# # skio.show()

# smallest_score = 100000000000000000
# # find b, r displacement
# for i in range(-15, 16):
#     for j in range(-15, 16):
#         score = return_score(b, r, i, j)
#         if score < smallest_score:
#             smallest_score = score
#             best_w = i
#             best_h = j
# b_r_dis = (best_h, best_w)

# smallest_score = 100000000000000000
# # find b, g displacement
# for i in range(-15, 16):
#     for j in range(-15, 16):
#         score = return_score(b, g, i, j)
#         if score < smallest_score:
#             smallest_score = score
#             best_w = i
#             best_h = j
# b_g_dis = (best_h, best_w)



# g_h = b_g_dis[0]
# g_w = b_g_dis[1]
# r_h = b_r_dis[0]
# r_w = b_r_dis[1]
# r = np.roll(r, b_r_dis[1], axis = 0)
# r = np.roll(r, b_r_dis[0], axis = 1)
# g = np.roll(g, b_g_dis[1], axis = 0)
# g = np.roll(g, b_g_dis[0], axis = 1)

# new_stacked = np.dstack([r, g, b])
# new_stacked = np.clip(new_stacked, 0, 1)
# new_stacked = (new_stacked * 255).astype(np.uint8)


# # save the image
# fname = './church_final.jpg'
# skio.imsave(fname, new_stacked)
try:
    from PIL import Image
except Exception:
    Image = None


def resize_image(img, factor):
    # 0) Validate factor
    if factor is None or float(factor) <= 0:
        raise ValueError(f"'factor' must be > 0, got {factor}")

    # 1) Coerce to ndarray from common types
    if Image is not None and isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # 2) Sanity checks
    if img is None:
        raise ValueError("Input image is None")
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got ndim={img.ndim} and shape={getattr(img,'shape',None)}")

    # 3) Normalize dtype to OpenCV-friendly types
    if img.dtype == np.bool_:
        img = (img.astype(np.uint8) * 255)
    elif img.dtype == object:
        img = np.array(img.tolist(), dtype=np.uint8)
    elif img.dtype not in (np.uint8, np.int16, np.int32, np.float32, np.float64):
        img = img.astype(np.float32, copy=False)

    # 4) Force C-contiguous (some cv2 builds choke on strided views)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)

    # 5) Compute explicit output size instead of fx/fy
    h, w = img.shape[:2]
    new_w = max(1, int(round(w / float(factor))))
    new_h = max(1, int(round(h / float(factor))))

    # Optional debug (keep while diagnosing)
    # print("IN:", type(img), img.dtype, img.shape, img.flags['C_CONTIGUOUS'])
    # print("OUT size:", (new_w, new_h))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)



# name of the input file
imname = 'harvesters.tif'

# read in the image
im = skio.imread(imname)
print("Image shape:", im.shape)


# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int64)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]
# crop images
h, w = b.shape
crop_h = int(0.05 * h)
crop_w = int(0.05 * w)
b = b[crop_h: -crop_h, crop_w: -crop_w]
g = g[crop_h: -crop_h, crop_w: -crop_w]
r = r[crop_h: -crop_h, crop_w: -crop_w]


b_g_dis = (0, 0)
b_r_dif = (0, 0)


for factor in (16, 8, 4):
    new_b = resize_image(b, factor)
    new_g = resize_image(g, factor)

    smallest_score = float("inf")
    # find b, g displacement
    for i in range(-15, 16):
        for j in range(-15, 16):
            score = return_score(new_b, new_g, i, j)
            if score < smallest_score:
                smallest_score = score
                best_w = i
                best_h = j
    b_g_dis = b_g_dis + (best_h * factor, best_w * factor)
print(b_g_dis)

