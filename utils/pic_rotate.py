import cv2
import math


def pic_rotate(img, bboxes, angle):
    h, w = img.shape[:2]
    xmid = w / 2
    ymid = h / 2

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
        xmin, ymin, xmax, ymax = bbox

        xmin -= xmid
        ymin -= ymid
        xmax -= xmid
        xmax -= ymid

        '''
        Affine transformation, anticlockwise with positive rotate_degrees
        x' = cos(degree) * x - sin(degree) * y
        y' = sin(degree) * x + cos(degree) * y
        '''

        sin = math.sin(radian)
        cos = math.cos(radian)

        rotated_xmin = cos * xmin - sin * ymin + (bound_w / 2)
        rotated_ymin = sin * xmin + cos * ymin + (bound_h / 2)
        rotated_xmax = cos * xmax - sin * ymax + (bound_w / 2)
        rotated_ymax = sin * xmax + cos * ymax + (bound_h / 2)

        rotated_bboxes.append([int(rotated_xmin), int(rotated_ymin), int(rotated_xmax), int(rotated_ymax)])

    return rotated_img, rotated_bboxes


if __name__ == '__main__':
    # test
    angle = 30
    bboxes = [[1201, 621, 1233, 691]]
    # bboxes = [[1201, 621, 1233, 691], [1099, 455, 1122, 540], [1084, 789, 1071, 762]]
    img = cv2.imread("D:/2021hwsz/Images/2020-01-11_21_36_02_642.jpg")
    for bbox in bboxes:
        print(bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.imread("D:/2021hwsz/Images/2020-01-11_21_36_02_642.jpg")
    rotated_img, rotated_bboxes = pic_rotate(img, bboxes, angle)
    for rotated_bbox in rotated_bboxes:
        cv2.rectangle(rotated_img, (rotated_bbox[0], rotated_bbox[1]), (rotated_bbox[2], rotated_bbox[3]), (0, 0, 255), 2)
    cv2.imshow("rotated_img", rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()