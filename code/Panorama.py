import FileIO
from StitchImage import stitchImage


files = FileIO.loadImage(shrink_times=1)
file_stitch = None

for i in range(0, len(files)-1):
    if i == 0:
        file1 = files[0]
    else:
        file1 = file_stitch

    file2 = files[i + 1]

    print('Start stitching {0} and {1}...'.format(file1.name, file2.name))

    file_stitch = stitchImage(file1, file2)

    file_stitch.saveImage(FileIO.SAVE_TEST)
