# data model module
# To manipulate file operation like load image, save image...
# ImageModel member:
#   name: the name of the image in 'str' type, without image format like '.JPG', '.png', ...
#   image: the image in 'ndarray' type
import os
import cv2
from glob import glob

SAVE_RESULT = 0
SAVE_MATCH = 1
SAVE_TEST = 2
SAVE_HOMO = 3

IMAGE_FORMAT = ['.jpg', '.JPG', '.png', '.PNG', '.tif', '.TIF', '.bmp', '.BMP', '.jpeg', '.JPEG']


class ImageModel:
    def __init__(self, name, image):
        self.name = removeFormatInName(name)
        self.image = image


def removeFormatInName(name):
    for img_format in IMAGE_FORMAT:
        name = name.replace(img_format, '')

    return name


def appendFormatToName(name):
    print('append \'.JPG\' format to the image \'{0}\''.format(name))
    name += '.JPG'

    return name


def saveImage(name, image, save_type):
    name = appendFormatToName(name)

    if save_type == SAVE_RESULT:
        save_dir = 'result'
    elif save_type == SAVE_MATCH:
        save_dir = 'log/match'
    elif save_type == SAVE_TEST:
        save_dir = 'log/test'
    elif save_type == SAVE_HOMO:
        save_dir = 'log/homo'
    else:
        print('save type not accepted\nmust be {0}~{1}'.format(SAVE_RESULT, SAVE_HOMO))
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

    imageModels = []

    for i in range(len(images_path)):
        s = os.path.abspath(__file__ + '/../') + '/../images/'

        original_name = images_path[i].replace(s, '')
        name = str(i+1)
        print('convert the data name \'{0}\' to \'{1}\''.format(original_name, name))

        imageModels.append(ImageModel(name, images[i]))

    return imageModels
