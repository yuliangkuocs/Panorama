import os
import cv2
from glob import glob

DATA_SHAPE = None
IMAGE_FORMAT = ['.jpg', '.JPG', '.png', '.PNG', '.tif', '.TIF', '.bmp', '.BMP', '.jpeg', '.JPEG']


class ImageModel:
    def __init__(self, name, image):
        self.name = removeFormatInName(name)
        self.image = image

    SAVE_RESULT = DEL_RESULT = 0
    SAVE_MATCH = DEL_MATCH = 1
    SAVE_TEST = DEL_TEST = 2
    SAVE_HOMO = DEL_HOMO = 3


def removeFormatInName(name):
    global IMAGE_FORMAT

    for img_format in IMAGE_FORMAT:
        name = name.replace(img_format, '')

    return name


def appendFormatToName(name):
    print('append \'.JPG\' format to the image \'{0}\''.format(name))
    name += '.JPG'

    return name


def saveImage(name, image, save_type):
    global DATA_SHAPE
    if image.shape == DATA_SHAPE:
        return

    name = appendFormatToName(name)

    if save_type == ImageModel.SAVE_RESULT:
        save_dir = 'result'
    elif save_type == ImageModel.SAVE_MATCH:
        save_dir = 'log/match'
    elif save_type == ImageModel.SAVE_TEST:
        save_dir = 'log/test'
    elif save_type == ImageModel.SAVE_HOMO:
        save_dir = 'log/homo'
    else:
        print('save type not accepted\nmust be {0}~{1}'.format(ImageModel.SAVE_RESULT, ImageModel.SAVE_HOMO))
        return

    path = os.path.join(os.path.abspath(__file__ + '/../../{0}/'.format(save_dir)), name)
    cv2.imwrite(path, image)
    print('image \'{0}\' already saved at {1}, '.format(name, save_dir))


def loadImage(shrink_times):
    print('loading images from dir \'images\'...')
    images_path = sorted(glob(os.path.dirname(__file__) + '/../images/*'))
    images = [cv2.imread(imagePath) for imagePath in images_path]

    print('shrink {0} times using Gaussian pyramid down method'.format(shrink_times))
    for i in range(shrink_times):
        images = [cv2.pyrDown(img) for img in images]

    global DATA_SHAPE
    DATA_SHAPE = images[0].shape

    imageModels = []

    for i in range(len(images_path)):
        s = os.path.abspath(__file__ + '/../') + '/../images/'

        original_name = images_path[i].replace(s, '')
        name = original_name.replace('DJI_', '')
        # print('convert the data name \'{0}\' to \'{1}\''.format(original_name, name))

        imageModels.append(ImageModel(name, images[i]))

    return imageModels


def removeImage(name, del_type):
    name = appendFormatToName(name)

    if del_type == ImageModel.DEL_RESULT:
        del_dir = 'result'
    elif del_type == ImageModel.DEL_MATCH:
        del_dir = 'log/match'
    elif del_type == ImageModel.DEL_TEST:
        del_dir = 'log/test'
    elif del_type == ImageModel.DEL_HOMO:
        del_dir = 'log/homo'
    else:
        print('delete type did not accept\nmust be {0}~{1}'.format(ImageModel.DEL_RESULT, ImageModel.DEL_HOMO))
        return

    path = os.path.join(os.path.abspath(__file__ + '/../../{0}/'.format(del_dir)), name)
    os.remove(path)
    print('image \'{0}\' at {1} already deleted.'.format(name, del_dir))