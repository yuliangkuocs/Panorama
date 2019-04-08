import time
from ImageModel import ImageModel, saveImage, loadImage
from StitchTwoImage import stitchTwoImage


image_models = loadImage(shrink_times=0)
now_stitch = None
continue_warp_not_good = 0

start = time.time()

i = 0
while i < len(image_models)-1:
    print('\ni =', i)

    if continue_warp_not_good == 2:
        image_model1 = image_models[i - 1] if not now_stitch else now_stitch
        continue_warp_not_good = 0
    else:
        image_model1 = image_models[i] if not now_stitch else now_stitch
    image_model2 = image_models[i + 1]

    print('Start stitching {0} and {1}...'.format(image_model1.name, image_model2.name))

    stitch, isWarpGood = stitchTwoImage(image_model1, image_model2)

    if isWarpGood:
        now_stitch = stitch
        continue_warp_not_good = 0

    else:
        continue_warp_not_good += 1

        if continue_warp_not_good == 2:
            print('[Warning] continue warping bad, discard data \'{0}\''.format(image_models[i - 1].name))

            saveImage(stitch.name, stitch.image, ImageModel.SAVE_RESULT)
            now_stitch = None

            i = i - 1 if i - 1 >= 0 else 0

    i += 1

saveImage(stitch.name, stitch.image, ImageModel.SAVE_RESULT)

print('Total stitch {0} images --'.format(len(image_models)), time.time() - start, 's')

