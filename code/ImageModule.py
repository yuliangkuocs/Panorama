# image manipulation module
import numpy as np
import cv2


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


def getBlendRegion(mask):
    region_left = np.argmax(mask, axis=1)
    region_top = np.argmax(mask, axis=0)
    region_right = mask.shape[1] - np.argmax(np.flip(mask, axis=1), axis=1) - 1
    region_down = mask.shape[0] - np.argmax(np.flip(mask, axis=0), axis=0) - 1

    return [region_left, region_right, region_top, region_down]


def findCoverState(direct, mask1, mask2, at, lowbound, upbound):
    # To know how two images cover each other
    # 0: Not the both, 1: image1, 2: image2

    if direct:  # horizon
        if lowbound == 0 or (not mask1[at, lowbound - 1] and not mask2[at, lowbound - 1]):
            left_img = 0
        elif mask1[at, lowbound - 1]:
            left_img = 1
        else:
            left_img = 2

        if upbound == mask1.shape[1] - 1 or (not mask1[at, upbound + 1] and not mask2[at, upbound + 1]):
            right_img = 0
        elif mask1[at, upbound + 1]:
            right_img = 1
        else:
            right_img = 2

        return [left_img, right_img]

    else: # vertical
        if lowbound == 0 or (not mask1[lowbound - 1, at] and not mask2[lowbound - 1, at]):
            top_img = 0
        elif mask1[lowbound - 1, at]:
            top_img = 1
        else:
            top_img = 2

        if upbound == mask1.shape[0] - 1 or (not mask1[upbound + 1, at] and not mask2[upbound + 1, at]):
            down_img = 0
        elif mask1[upbound + 1, at]:
            down_img = 1
        else:
            down_img = 2

        return [top_img, down_img]


def blending(img1, img2, mask, mask1, mask2):
    if img1.shape != img2.shape:
        print('The shape of img1 must be the same as the shape of img2')
        return

    if mask.shape != mask1.shape != mask2.shape:
        print('Masks\' shapes are not the same')
        return

    image = img1.copy()
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask2[row, col]:
                image[row, col] = img2[row, col]

    print('cut padding...')
    image, imgs, masks = cutPadding(image, [img1, img2], [mask, mask1, mask2])

    region = getBlendRegion(masks[0])

    rowCoverState = [findCoverState(True, masks[1], masks[2], row, region[0][row], region[1][row]) for row in range(image.shape[0])]
    colCoverState = [findCoverState(False, masks[1], masks[2], col, region[2][col], region[3][col]) for col in range(image.shape[1])]

    print('start blending...')
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if masks[0][row, col]:
                left, right, top, down = region[0][row], region[1][row], region[2][col], region[3][col]

                alpha = (col - left) / (right - left)
                beta = (row - top) / (down - top)

                if rowCoverState[row][0] == 0 and rowCoverState[row][1] == 0:
                    alpha = 0.5
                elif (rowCoverState[row][0] == 0 and rowCoverState[row][1] == 2) or \
                     (rowCoverState[row][0] == 1 and rowCoverState[row][1] == 0) or \
                     (rowCoverState[row][0] == 1 and rowCoverState[row][1] == 2):
                    alpha = 1 - alpha
                elif rowCoverState[row][0] == 1 and rowCoverState[row][1] == 1:
                    mid = int((right - left) / 2)
                    alpha = (col - left) / (mid - left) if col <= mid else (col - mid) / (right - mid)
                    alpha = 1 - alpha
                elif rowCoverState[row][0] == 2 and rowCoverState[row][1] == 2:
                    mid = int((right - left) / 2)
                    try:
                        alpha = (col - left) / (mid - left) if col <= mid else (col - mid) / (right - mid)
                    except ZeroDivisionError:
                        print('[ERROR] Divide by zero')
                        alpha = 1

                if colCoverState[col][0] == 0 and colCoverState[col][1] == 0:
                    beta = 0.5
                elif (colCoverState[col][0] == 0 and colCoverState[col][1] == 2) or \
                     (colCoverState[col][0] == 1 and colCoverState[col][1] == 0) or \
                     (colCoverState[col][0] == 1 and colCoverState[col][1] == 2):
                    beta = 1 - beta
                elif colCoverState[col][0] == 1 and colCoverState[col][1] == 1:
                    mid = int((down - top) / 2)
                    try:
                        beta = (row - top) / (mid - top) if row <= mid else (row - mid) / (down - mid)
                    except ZeroDivisionError:
                        print('[ERROR] Divide by zero')
                        beta = 0
                elif colCoverState[col][0] == 2 and colCoverState[col][1] == 2:
                    mid = int((down - top) / 2)
                    try:
                        beta = (row - top) / (mid - top) if row <= mid else (row - mid) / (down - mid)
                    except ZeroDivisionError:
                        print('[ERROR] Divide by zero')
                        beta = 1
                image[row, col] = 0.5 * (alpha * imgs[0][row, col] + (1 - alpha) * imgs[1][row, col]) + 0.5 * (beta * imgs[0][row, col] + (1 - beta) * imgs[1][row, col])

    return image


def getMask(img1, img2):
    if img1.shape != img2.shape:
        print('The shape of img1 must be the same as the shape of img2')
        return

    all_true = np.full(shape=img1.shape, fill_value=1)
    mask1 = np.logical_and(np.any(img1, axis=2), np.any(all_true, axis=2))
    mask2 = np.logical_and(np.any(img2, axis=2), np.any(all_true, axis=2))

    return np.logical_and(mask1, mask2), mask1, mask2


def showImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
