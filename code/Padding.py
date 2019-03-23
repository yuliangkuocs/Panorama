# image manipulation module
import numpy as np


normal_shape = None
normal_img = None


def addPadding(stitched_img, raw_img):
    if type(stitched_img) != np.ndarray or type(raw_img) != np.ndarray:
        print('image type must be \'ndarray\', but found {0} and {1}'.format(type(stitched_img), type(raw_img)))

    h1, w1, _ = stitched_img.shape
    h2, w2, _ = raw_img.shape

    global normal_shape, normal_img
    normal_shape = [stitched_img.shape, raw_img.shape]
    normal_img = raw_img

    pad_stitched = np.zeros(shape=(2 * h1 + 2 * h2, 2 * w1 + 2 * w2, 3), dtype=np.uint8)
    pad_raw = np.full(shape=(2 * h1 + 2 * h2, 2 * w1 + 2 * w2, 3), fill_value=255, dtype=np.uint8)

    pad_stitched[h2 + int(h1 / 2): h2 + int(h1 / 2) + h1, w2 + int(w1 / 2): w2 + int(w1 / 2) + w1, :] = stitched_img
    pad_raw[h1 + int(h2 / 2): h1 + int(h2 / 2) + h2, w1 + int(w2 / 2): w1 + int(w2 / 2) + w2, :] = raw_img

    return pad_stitched, pad_raw


def paddingNormalize(raw_image):
    # Convert white padding of the image to black padding
    global normal_shape, normal_img
    h1, w1, _ = normal_shape[0]
    h2, w2, _ = normal_shape[1]

    normalize = np.zeros(shape=raw_image.shape, dtype=np.uint8)
    normalize[h1 + int(h2 / 2): h1 + int(h2 / 2) + h2, w1 + int(w2 / 2): w1 + int(w2 / 2) + w2, :] = normal_img

    return normalize


def cutPadding(image, imgs=None, masks=None):
    # Cut black padding of the image to make it as small as possible
    if masks:
        for mask in masks:
            if mask.shape != image.shape[:2]:
                print('[ERROR] mask must have the same shape as image, mask shape =', mask.shape, 'image shape =', image.shape)
                return

    if imgs:
        for img in imgs:
            if img.shape != img.shape:
                print('[ERROR] img must have the same shape as image')
                return

    while np.sum(image[0, :, :]) == 0:
        image = image[1:, :, :]
        if masks:
            for i, mask in enumerate(masks):
                masks[i] = mask[1:, :]
        if imgs:
            for i, img in enumerate(imgs):
                imgs[i] = img[1:, :, :]

    while np.sum(image[:, 0, :]) == 0:
        image = image[:, 1:, :]
        if masks:
            for i, mask in enumerate(masks):
                masks[i] = mask[:, 1:]

        if imgs:
            for i, img in enumerate(imgs):
                imgs[i] = img[:, 1:, :]

    while np.sum(image[image.shape[0] - 1, :, :]) == 0:
        image = image[:image.shape[0] - 1, :, :]
        if masks:
            for i, mask in enumerate(masks):
                masks[i] = mask[:mask.shape[0] - 1, :]

        if imgs:
            for i, img in enumerate(imgs):
                imgs[i] = img[:img.shape[0] - 1, :, :]

    while np.sum(image[:, image.shape[1] - 1, :]) == 0:
        image = image[:, :image.shape[1] - 1, :]
        if masks:
            for i, mask in enumerate(masks):
                masks[i] = mask[:, :mask.shape[1] - 1]

        if imgs:
            for i, img in enumerate(imgs):
                imgs[i] = img[:, :img.shape[1] - 1, :]

    return image, imgs, masks
