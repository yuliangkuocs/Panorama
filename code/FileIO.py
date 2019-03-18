# file IO module
# To manipulate file operation like load image, save image...
import os
import cv2
from glob import glob

SAVE_RESULT = 0
SAVE_MATCH = 1
SAVE_TEST = 2
SAVE_HOMO = 3


class File:
    def __init__(self, name, image):
        self.name = name
        self.image = image
        self.name = self.name.replace('.JPG', '')

    def __checkName(self):
        if type(self.name) != str:
            print('file name must be \'str\', but found {0}'.format(type(self.name)))
            return False

        elif self.name.find('.') == -1:
            print('change image \'{0}\' format into \'.JPG\''.format(self.name))
            self.name += '.JPG'

        return True

    def saveImage(self, save_type):
        if self.__checkName():
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

            try:
                path = os.path.join(os.path.abspath(__file__ + '/../../{0}/'.format(save_dir)), self.name)
                cv2.imwrite(path, self.image)
                print('image \'{0}\' already saved at {1}, '.format(self.name, save_dir))
                self.name = self.name.replace('.JPG', '')
            except Exception as e:
                print('save image fail:', e)


def loadImage(shrink_times):
    print('loading images from dir \'images\'...')
    try:
        images_path = sorted(glob(os.path.dirname(__file__) + '/../images/*.JPG'))
        images = [cv2.imread(imagePath) for imagePath in images_path]
    except Exception as e:
        print('load images fail:', e)
        return

    print('shrink {0} times using Gaussian pyramid down method...'.format(shrink_times))
    try:
        for i in range(shrink_times):
            images = [cv2.pyrDown(img) for img in images]
    except Exception as e:
        print('shrink images fail:', e)
        return

    files = []

    for i in range(len(images_path)):
        s = os.path.abspath(__file__ + '/../') + '/../images/'
        name = images_path[i].replace(s, '').replace('.JPG', '').replace('DJI_00', '')
        files.append(File(name, images[i]))

    return files
