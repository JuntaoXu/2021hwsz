import random
import math
import cv2
import numpy as np
from skimage import exposure
from skimage.util import random_noise


# crop
def crop_img_bboxes(img, bboxes):
    # get width and height
    w, h = img.shape[:2]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    # largest distance to all edges with all boxed included
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    # randomly expand smallest area able to crop
    crop_x_min = int(x_min - random.uniform(0, d_to_left))
    crop_y_min = int(y_min - random.uniform(0, d_to_top))
    crop_x_max = int(x_max + random.uniform(0, d_to_right))
    crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

    # avoid out of range
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)

    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # crop image
    crop_bboxes = list()
    for bbox in bboxes:
        crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min,
                            bbox[2] - crop_x_min, bbox[3] - crop_y_min])

    return crop_img, crop_bboxes


# pan
def shift_pic_bboxes(img, bboxes):
    # get width and height
    w, h = img.shape[:2]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(x_max, bbox[3])

    # largest distance to all edges with all boxed included
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    # line 1, if x > 0, right shift, else left shift
    # line 2, if y > 0, up shift, else down shift
    x = random.uniform(-(d_to_left / 3), d_to_right / 3)
    y = random.uniform(-(d_to_top / 3), d_to_bottom / 3)
    M = np.float32([[1, 0, x], [0, 1, y]])

    # Affine transformation of the img
    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # shifting the bounding box
    shift_bboxes = list()
    for bbox in bboxes:
        shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

    return shift_img, shift_bboxes


# alter exposure
def alterLight(img):
    flag = random.uniform(0.5, 1.5)  # lighter when flag < 1, darker when flag > 1
    return exposure.adjust_gamma(img, flag)


# add_noise
def addNoise(img):
    # output pixel between 0 and 1, thus need to be multiplied by 255
    return random_noise(img) * 255


# rotate img and bboxes
def rotate_img_bboxes(img, bboxes, degrees):
    # get width and height of the img
    h, w = img.shape[:2]
    # src = img
    # srcTri = np.array([[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]]).astype(np.float32)
    # dstTri = np.array([[0, src.shape[1] * 0.33], [src.shape[1] * 0.85, src.shape[0] * 0.25],
    #                    [src.shape[1] * 0.15, src.shape[0] * 0.7]]).astype(np.float32)
    # warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    # warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
    # rotate the img by degrees
    rotated = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1)
    img = cv2.warpAffine(img, rotated, (w, h), borderValue=(255,255,255))

    # convert degree into radian, positive radian leads to anticlockwise Affine transformation
    rotate_degrees = -1 * math.pi * (degrees/180)
    rotated_bboxes = []
    for bbox in bboxes:
        xmid = (bbox[2]+bbox[0])/2/w
        ymid = (bbox[3]+bbox[1])/2/h
        bw = (bbox[2]-bbox[0])/w
        bh = (bbox[3]-bbox[1])/h
        bbox = [xmid, ymid, bw, bh]
        # change the coordiante center origin to center of the image
        bbox[0] = bbox[0] - 0.5
        bbox[1] = bbox[1] - 0.5
        # get xmin, xmax, ymin, ymax to construct diagonal points A(xmin, ymin) & D(xmax, ymax)
        xmin, ymin = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2
        xmax, ymax = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        '''
        Affine transformation, anticlockwise with positive rotate_degrees
        x' = cos(degree) * x - sin(degree) * y
        y' = sin(degree) * x + cos(degree) * y
        '''
        # A' & D‘
        Ax = math.cos(rotate_degrees) * xmin - math.sin(rotate_degrees) * ymin + 0.5
        Ay = math.sin(rotate_degrees) * xmin + math.cos(rotate_degrees) * ymin + 0.5
        Dx = math.cos(rotate_degrees) * xmax - math.sin(rotate_degrees) * ymax + 0.5
        Dy = math.sin(rotate_degrees) * xmax + math.cos(rotate_degrees) * ymax + 0.5

        bxmin = Ax*w
        bymin = Ay*h
        bxmax = Dx*w
        bymax = Dy*h

        # get xmid', ymid', w, h
        # rotated_bboxes.append([(Ax + Dx)/2, (Ay + Dy)/2, abs(Ax - Dx), abs(Ay - Dy)])
        rotated_bboxes.append([bxmin, bymin, bxmax, bymax])

    return img, rotated_bboxes



def rotate(img, bboxes, angle):
    h, w = img.shape[:2]
    xmid, ymid = w / 2, h / 2

    rotation_img = cv2.getRotationMatrix2D((xmid, ymid), angle, 1)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_img[0, 0])
    abs_sin = abs(rotation_img[0, 1])

    # find bound_w and bound_h
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # alter to new image center
    rotation_img[0, 2] += bound_w / 2 - xmid
    rotation_img[1, 2] += bound_h / 2 - ymid

    rotated_img = cv2.warpAffine(img, rotation_img, (bound_w, bound_h), cv2.IMREAD_UNCHANGED, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    radian = -1 * math.pi * (angle / 180)
    rotated_bboxes = []
    for bbox in bboxes:
        xmin_to_origin = bbox[0] - xmid
        ymin_to_origin = bbox[1] - ymid
        xmax_to_origin = bbox[2] - xmid
        ymax_to_origin = bbox[3] - ymid

        '''
        Affine transformation, anticlockwise with positive rotate_degrees
        x' = cos(radian) * x - sin(radian) * y
        y' = sin(radian) * x + cos(radian) * y
        '''

        sin = math.sin(radian)
        cos = math.cos(radian)
        r_xmin = cos * xmin_to_origin - sin * ymin_to_origin + (bound_w / 2)
        r_ymin = sin * xmin_to_origin + cos * ymin_to_origin + (bound_h / 2)
        r_xmax = cos * xmax_to_origin - sin * ymax_to_origin + (bound_w / 2)
        r_ymax = sin * xmax_to_origin + cos * ymax_to_origin + (bound_h / 2)

        rotated_bboxes.append([int(r_xmin), int(r_ymin), int(r_xmax), int(r_ymax)])

    return rotated_img, rotated_bboxes


def draw(img, bboxes):
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# flip
def flip_pic_bboxes(img, bboxes):
    # print('ori: ', bboxes)
    import copy
    flip_img = copy.deepcopy(img)
    if random.random() < 0.5:
        horizon = True
    else:
        horizon = False
    h, w = img.shape[:2]
    if horizon:  # horizontal flip
        flip_img = cv2.flip(flip_img, 1)
    else:   # vertical flip
        flip_img = cv2.flip(flip_img, 0)
    # modify bboxes
    flip_bboxes = list()
    for bbox in bboxes:
        # print(bbox)
        if horizon:
            xmin, ymin = w-bbox[2],bbox[1]
            xmax, ymax = w-bbox[0],bbox[3]
            # print('newbbox: ', [xmin,ymin, xmax, ymax])
            flip_bboxes.append([xmin,ymin, xmax, ymax])
        else:
            xmin, ymin = bbox[0], h-bbox[3]
            xmax, ymax = bbox[2], h-bbox[1]
            # print('newbbox: ', [xmin,ymin, xmax, ymax])
            flip_bboxes.append([xmin,ymin, xmax, ymax])
    # print('flipped: ',flip_bboxes)
    return flip_img, flip_bboxes

def random_crop_boxes(img, bboxes, min_scale=-0.1, max_scale=0.1):
    '''
    randomly crop or expand boxes
    :param img:
    :param bboxes:
    :return:
    '''
    # get width and height
    h, w = img.shape[:2]
    cropped_boxes = list()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        if (xmax - xmin)*(ymax-ymin) < 800: min_scale, max_scale = 0.1, 0.3
        scale = random.uniform(min_scale, max_scale)
        w_bbox = xmax - xmin
        h_bbox = ymax - ymin
        xmin = max(xmin - w_bbox*scale, 0)
        ymin = max(ymin - h_bbox*scale, 0)
        xmax = min(xmax + w_bbox*scale, w)
        ymax = min(ymax + h_bbox*scale, h)
        bbox = [xmin,ymin,xmax,ymax]
        cropped_boxes.append(bbox)
    return img, cropped_boxes


if __name__ == '__main__':
    angle = 30
    bboxes = [[1201, 621, 1233, 691], [1099, 455, 1122, 540], [1084, 789, 1071, 762]]
    img = cv2.imread("D:/2021hwsz/Images/2020-01-11_21_36_02_642.jpg")
    draw(img, bboxes)

    img = cv2.imread("D:/2021hwsz/Images/2020-01-11_21_36_02_642.jpg")
    rotated_img, rotated_bboxes = rotate(img, bboxes, angle)
    draw(rotated_img, rotated_bboxes)


