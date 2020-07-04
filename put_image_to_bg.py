import os
import cv2
import numpy as np
import numpy as np
os.chdir('.')
INPUT_PATH = 'pics/banana_staff/'
OUTPUT_PATH = 'res/'


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def put_image_on_background(image, bg):
    cropped_bg = get_random_crop(bg, image.shape[0], image.shape[1])

    Conv_hsv_Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY)
    mask_inverted = cv2.bitwise_not(mask)
    fg = cv2.bitwise_or(image, image, mask=mask)
    bk = cv2.bitwise_or(cropped_bg, cropped_bg, mask=mask_inverted)
    res = cv2.bitwise_or(fg, bk)
    return res

image = cv2.imread('pics/img.jpg')
bg = cv2.imread('pics/banana-bg.PNG')
files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(INPUT_PATH)] for val in sublist]
# files = [f for f in listdir(INPUT_PATH) if isfile(join(INPUT_PATH, f))]
for file_name in files:
    input_file = file_name
    image = cv2.imread(input_file)
    res = put_image_on_background(image, bg)
    outpath = os.path.join(OUTPUT_PATH, file_name)
    dirname = os.path.dirname(outpath)
    print(outpath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cv2.imwrite(outpath, res)

#
#
# res = put_image_on_background(image, bg)
# cv2.imshow('res', res); cv2.waitKey(); cv2.destroyAllWindows()
