import cv2
import math


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


if __name__ == '__main__':
    while True:
        angle = 30
        bboxes = [[1201, 621, 1233, 691], [1099, 455, 1122, 540], [1084, 789, 1071, 762]]
        img = cv2.imread("D:/2021hwsz/Images/2020-01-11_21_36_02_642.jpg")
        draw(img, bboxes)

        img = cv2.imread("D:/2021hwsz/Images/2020-01-11_21_36_02_642.jpg")
        rotated_img, rotated_bboxes = rotate(img, bboxes, angle)
        draw(rotated_img, rotated_bboxes)
