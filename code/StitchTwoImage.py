import cv2
import numpy as np
import ImageModel
import Padding
from AlphaBlending import alphaBlending


def getFeature(image):
    print('finding features...')
    MAX_FEATURES = 500

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def featureMatching(des1, des2):
    print('start matching descriptors...')
    GOOD_MATCH_PERCENT = 0.15

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    return matches


def getHomographyMatrix(kp1, kp2, matches):
    print('calculating homography matrix...')
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print('Homography Matrix:\n', h)

    return h


def drawMatchImage(name, image):
    ImageModel.saveImage(name, image, ImageModel.SAVE_MATCH)


def stitchTwoImage(image_model1, image_model2):
    stitch_img, raw_img = Padding.addPadding(image_model1.image, image_model2.image)

    img1_gray = cv2.cvtColor(stitch_img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    kp1, des1 = getFeature(img1_gray)
    kp2, des2 = getFeature(img2_gray)

    matches = featureMatching(des1, des2)

    img_match = cv2.drawMatches(stitch_img, kp1, raw_img, kp2, matches, None)
    drawMatchImage('match ' + image_model1.name + ' + ' + image_model2.name, img_match)

    h = getHomographyMatrix(kp1, kp2, matches)

    warp_img = cv2.warpPerspective(stitch_img, h, (raw_img.shape[1], raw_img.shape[0]))

    ImageModel.saveImage('homo image {0} -> {1}'.format(image_model1.name, image_model2.name),
                         warp_img,
                         ImageModel.SAVE_HOMO)

    raw_img = Padding.paddingNormalize(raw_img)

    print('getting masks of images...')
    mask_cover, mask1, mask2 = Padding.getMask(warp_img, raw_img)

    print('blending images...')
    blend_img = alphaBlending(warp_img, raw_img, mask_cover, mask1, mask2)

    image_model_stitch = ImageModel.ImageModel(image_model1.name + ' ' + image_model2.name, blend_img)

    return image_model_stitch
