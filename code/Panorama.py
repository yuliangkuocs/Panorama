import ImageModel
from StitchTwoImage import stitchTwoImage


image_models = ImageModel.loadImage(shrink_times=1)
image_model_stitch = None

for i in range(0, len(image_models)-1):
    if i == 0:
        image_model1 = image_models[0]
    else:
        image_model1 = image_model_stitch

    image_model2 = image_models[i + 1]

    print('Start stitching {0} and {1}...'.format(image_model1.name, image_model2.name))

    image_model_stitch = stitchTwoImage(image_model1, image_model2)

    ImageModel.saveImage(image_model_stitch.name, image_model_stitch.image, ImageModel.SAVE_RESULT)
