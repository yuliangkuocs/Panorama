import ImageModel
from StitchTwoImage import stitchTwoImage


image_models = ImageModel.loadImage(shrink_times=1)
now_stitch, pre_stitch = None, None
continue_warp_not_good = 0

i = 0
while i < len(image_models)-1:
    print('\ni =', i)
    if now_stitch:
        print('now:', now_stitch.name)
    else:
        print('now: none')
    if pre_stitch:
        print('pre:', pre_stitch.name)
    else:
        print('pre: none')
    print()
    if continue_warp_not_good == 2:
        image_model1 = image_models[i - 1] if not now_stitch else now_stitch
        continue_warp_not_good = 0
    else:
        image_model1 = image_models[i] if not now_stitch else now_stitch
    image_model2 = image_models[i + 1]

    print('Start stitching {0} and {1}...'.format(image_model1.name, image_model2.name))

    stitch, isWarpGood = stitchTwoImage(image_model1, image_model2)

    if isWarpGood:
        ImageModel.saveImage(stitch.name, stitch.image, ImageModel.SAVE_RESULT)

        pre_stitch = now_stitch
        now_stitch = stitch
        continue_warp_not_good = 0

    else:
        continue_warp_not_good += 1

        if continue_warp_not_good == 2:
            print('[Warning] continue warping bad, discard data \'{0}\''.format(i))
            now_stitch = pre_stitch
            pre_stitch = None
            i = i - 2 if i - 2 >= 0 else 0

    i += 1
