import cv2
import numpy as np
import FileIO
import ImageModule


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


def stitchImage(file1, file2):
    stitch_img, raw_img = ImageModule.addPadding(file1.image, file2.image)

    img1_gray = cv2.cvtColor(stitch_img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    kp1, des1 = getFeature(img1_gray)
    kp2, des2 = getFeature(img2_gray)

    matches = featureMatching(des1, des2)

    img_match = cv2.drawMatches(stitch_img, kp1, raw_img, kp2, matches, None)
    file_match = FileIO.File('match ' + file1.name + ' + ' + file2.name, img_match)
    file_match.saveImage(FileIO.SAVE_MATCH)

    h = getHomographyMatrix(kp1, kp2, matches)

    h_pad, w_pad, _ = raw_img.shape

    warp_img = cv2.warpPerspective(stitch_img, h, (w_pad, h_pad))

    raw_img = ImageModule.paddingNormalize(raw_img)

    print('getting masks of images...')
    mask_cover, mask1, mask2 = ImageModule.getMask(warp_img, raw_img)

    file_tmp = FileIO.File('homo image {0}'.format(file1.name), warp_img)
    file_tmp.saveImage(FileIO.SAVE_HOMO)

    blend_img = ImageModule.blending(warp_img, raw_img, mask_cover, mask1, mask2)

    file_stitch = FileIO.File(file1.name + ' ' + file2.name, blend_img)

    return file_stitch
