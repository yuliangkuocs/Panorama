import cv2
import numpy as np
from Padding import cutPadding


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    image = A + B
    image, imgs, masks = cutPadding(image, [A, B], [m])
    # assume mask is float32 [0,1]
    m = masks[0]
    A, B = imgs[0], imgs[1]
    m = np.asarray(m, dtype=np.float32)
    mask = np.zeros(shape=(m.shape[0], m.shape[1], 3))
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            mask[r, c] = [1, 1, 1] if m[r, c] == 1 else [0, 0, 0]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = mask.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [gpA[num_levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        upA = cv2.resize(cv2.pyrUp(gpA[i]), (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
        upB = cv2.resize(cv2.pyrUp(gpB[i]), (gpB[i - 1].shape[1], gpB[i - 1].shape[0]))
        LA = np.subtract(gpA[i - 1], upA)
        LB = np.subtract(gpB[i - 1], upB)
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        up_ls = cv2.resize(cv2.pyrUp(ls_), dsize=(LS[i].shape[1], LS[i].shape[0]))
        up_ls = np.asarray(up_ls, dtype=np.float32)
        LS[i] = np.asarray(LS[i], np.float32)
        ls_ = cv2.add(up_ls, LS[i])

    return ls_


def alphaBlending(img1, img2, mask_cover, mask1, mask2):
    if img1.shape != img2.shape:
        print('The shape of img1 must be the same as the shape of img2')
        return

    if mask_cover.shape != mask1.shape != mask2.shape:
        print('Masks\' shapes are not the same')
        return

    image = img1.copy()
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask2[row, col]:
                image[row, col] = img2[row, col]

    image, imgs, masks = cutPadding(image, [img1, img2], [mask_cover, mask1, mask2])

    region = getBlendRegion(masks[0])

    rowCoverState = [findCoverState(True, masks[1], masks[2], row, region[0][row], region[1][row]) for row in
                     range(image.shape[0])]
    colCoverState = [findCoverState(False, masks[1], masks[2], col, region[2][col], region[3][col]) for col in
                     range(image.shape[1])]

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if masks[0][row, col]:
                alpha, beta = caculateAlphaBeta(region, row, col, rowCoverState, colCoverState)
                image[row, col] = 0.5 * (alpha * imgs[0][row, col] + (1 - alpha) * imgs[1][row, col]) + 0.5 * (
                beta * imgs[0][row, col] + (1 - beta) * imgs[1][row, col])

    return image


def caculateAlphaBeta(region, row, col, rowCoverState, colCoverState):
    left, right, top, down = region[0][row], region[1][row], region[2][col], region[3][col]

    alpha = (col - left) / (right - left) * 1.2 if (col - left) / (right - left) > 0.5 else (col - left) / (
        right - left) * 0.8
    beta = (row - top) / (down - top) * 1.2 if (row - top) / (down - top) > 0.5 else (row - top) / (down - top) * 0.8

    alpha = 1 if alpha > 1 else alpha
    beta = 1 if beta > 1 else beta

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

    return alpha, beta


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

    else:  # vertical
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
