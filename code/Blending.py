import cv2
import time
import numpy as np


def multibandBlending(A, B, mask):
    start = time.time()
    BAND_NUM = 6

    # generate Mask pyramid
    pyM = [mask]
    for i in range(BAND_NUM - 1):
        M = cv2.pyrDown(pyM[i])
        pyM.append(M)

    # generate Gaussian pyramid for A and B
    gpA = [np.asarray(A, np.float64)]
    gpB = [np.asarray(B, np.float64)]

    for i in range(1, BAND_NUM):
        a = cv2.pyrDown(gpA[i - 1])
        b = cv2.pyrDown(gpB[i - 1])
        gpA.append(a)
        gpB.append(b)

    # generate Laplacian Pyramid for A and B
    lpA = [gpA[BAND_NUM - 1]]
    lpB = [gpB[BAND_NUM - 1]]

    for i in range(BAND_NUM - 1, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        res = cv2.resize(GE, (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
        L = cv2.subtract(gpA[i - 1], res)
        lpA.append(L)

    for i in range(BAND_NUM - 1, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        res = cv2.resize(GE, (gpB[i - 1].shape[1], gpB[i - 1].shape[0]))
        L = cv2.subtract(gpB[i - 1], res)
        lpB.append(L)

    # Now add A and B in each level
    LS = []
    for i, (la, lb) in enumerate(zip(lpA, lpB)):
        ls = np.zeros(la.shape)
        for c in range(la.shape[2]):
            mask = pyM[BAND_NUM - i - 1]
            ls[:, :, c] = la[:, :, c] * mask + lb[:, :, c] * (1 - mask)

        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, BAND_NUM):
        ls_ = cv2.pyrUp(ls_)
        res = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(res, LS[i])

    # Normalize image and convert the type back to uint8
    ls_ = cv2.convertScaleAbs(ls_)

    print('--', time.time() - start, 's')

    return ls_
