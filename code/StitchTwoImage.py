import cv2
import time
import numpy as np
from ImageModel import saveImage, ImageModel
from Blending import multibandBlending


def getFeature(image):
    start = time.time()

    print('finding features...')
    MAX_FEATURES = int(image.shape[0] * image.shape[1] / 10000)
    MAX_FEATURES = 500 if MAX_FEATURES < 500 else MAX_FEATURES

    print('max features:', MAX_FEATURES)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    end = time.time()

    print('--', end - start, 's')

    return keypoints, descriptors


def featureMatching(des1, des2):
    print('start matching descriptors...')
    start = time.time()

    GOOD_MATCH_PERCENT = 0.15

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort match descriptors in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    print('--', time.time() - start, 's')

    return matches


def getHomographyMatrix(kp1, kp2, matches):
    print('calculating homography matrix...')
    start = time.time()

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, s = cv2.findHomography(points1, points2, cv2.RANSAC)

    inliers_radio = float(np.sum(s)) / float(len(s))

    isHomographyGood = inliers_radio >= 0.2

    print('homography matrix inliers radio:', inliers_radio)

    print('--', time.time() - start, 's')

    return h, isHomographyGood


def getMask(img):
    all_true = np.full(shape=img.shape, fill_value=1)
    mask = np.logical_and(np.any(img, axis=2), np.any(all_true, axis=2))
    mask = np.asarray(mask, dtype=np.uint8)

    return mask


def cutPadding(images, mask=None):
    # Cut black padding of the image to make it as small as possible
    start = time.time()

    if mask is None:
        mask = images[0].copy()
    else:
        mask = np.expand_dims(mask, axis=2)

    for i in range(mask.shape[0]):
        if np.sum(mask[i, :, :]):
            break
        for i in range(len(images)):
            images[i] = images[i][1:, :, :]

    for i in range(mask.shape[1]):
        if np.sum(mask[:, i, :]):
            break
        for i in range(len(images)):
            images[i] = images[i][:, 1:, :]

    for i in range(mask.shape[0]):
        if np.sum(mask[mask.shape[0] - 1 - i, :, :]):
            break
        for i in range(len(images)):
            images[i] = images[i][:images[i].shape[0] - 1, :, :]

    for i in range(mask.shape[1]):
        if np.sum(mask[:, mask.shape[1] - 1 - i, :]):
            break
        for i in range(len(images)):
            images[i] = images[i][:, :images[i].shape[1] - 1, :]

    print('--', time.time() - start, 's')

    return images


def isAlreadyStitch(mask1, mask2):
    mask_xor = cv2.bitwise_xor(mask1, mask2)

    return np.sum(mask_xor) < 0.01 * mask_xor.shape[0] * mask_xor.shape[1]


def drawMatchImage(name, image):
    saveImage(name, image, ImageModel.SAVE_MATCH)


def stitchTwoImage(image_model1, image_model2):
    stitch_img, raw_img = image_model1.image, image_model2.image

    img1_gray = cv2.cvtColor(stitch_img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # Feature matching
    kp1, des1 = getFeature(img1_gray)
    kp2, des2 = getFeature(img2_gray)

    matches = featureMatching(des1, des2)

    img_match = cv2.drawMatches(stitch_img, kp1, raw_img, kp2, matches, None)
    drawMatchImage('match ' + image_model1.name + ' + ' + image_model2.name, img_match)

    # Calculate homography matrix
    h, isHomographyGood = getHomographyMatrix(kp1, kp2, matches)

    if not isHomographyGood:
        print('[WARNING] Bad homography matrix, discard this data', image_model2.name)
        return image_model1, False

    # Warping
    print('Warping...')
    start = time.time()

    new_size = (raw_img.shape[1] + int(stitch_img.shape[1] * 1.5), raw_img.shape[0] + int(stitch_img.shape[0] * 1.5))

    # put the image to the central position of the result image
    t = np.identity(3, np.float)
    t[0, 2] = new_size[0] / 4
    t[1, 2] = new_size[1] / 4

    pad_raw = cv2.warpPerspective(raw_img, t, new_size, borderMode=cv2.BORDER_REFLECT)
    raw_mask = cv2.warpPerspective(raw_img, t, new_size)

    pad_warp = cv2.warpPerspective(stitch_img, t.dot(h), new_size, borderMode=cv2.BORDER_REFLECT)
    warp_mask = cv2.warpPerspective(stitch_img, t.dot(h), new_size)

    stitch_mask = np.logical_or(getMask(raw_mask), getMask(warp_mask))
    stitch_mask = np.asarray(stitch_mask, dtype=np.uint8)

    saveImage('homo image {0} -> {1}'.format(image_model1.name, image_model2.name), warp_mask, ImageModel.SAVE_HOMO)

    print('--', time.time() - start, 's')

    # Cut Padding
    print('Cut padding...')
    cut_pad_images = [pad_raw, pad_warp, raw_mask, warp_mask]
    [cut_pad_raw, cut_pad_warp, cut_pad_raw_mask, cut_pad_warp_mask] = cutPadding(cut_pad_images, stitch_mask)

    if isAlreadyStitch(cut_pad_raw_mask, cut_pad_warp_mask):
        print('Data {0} already stitched, so don\'t need to keep stitching.'.format(image_model2.name))
        return image_model1, True

    final_mask = cv2.bitwise_or(getMask(cut_pad_raw_mask), getMask(cut_pad_warp_mask))
    final_mask = np.asarray(final_mask, dtype=np.uint8)

    # Blending
    print('blending images...')
    blend_mask = (np.sum(cut_pad_raw_mask, axis=2) != 0).astype(np.float)

    blend_img = multibandBlending(cut_pad_raw, cut_pad_warp, blend_mask)

    blend_img = cv2.bitwise_and(blend_img, blend_img, mask=final_mask)

    image_model_stitch = ImageModel(image_model1.name + ' ' + image_model2.name, blend_img)

    return image_model_stitch, True
