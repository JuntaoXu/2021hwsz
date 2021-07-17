import cv2
import math
import os
import json
import numpy as np
from tqdm import tqdm


def point_affine_transformation(x_origin, y_origin, w, h, radian):
    '''
    Affine transformation, anticlockwise with positive rotate_degrees
    x' = cos(radian) * x - sin(radian) * y
    y' = sin(radian) * x + cos(radian) * y
    '''

    sin = math.sin(radian)
    cos = math.cos(radian)
    x_rotated = int(cos * x_origin - sin * y_origin + (w / 2))
    y_rotated = int(sin * x_origin + cos * y_origin + (h / 2))
    return [x_rotated, y_rotated]


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

    rotated_img = cv2.warpAffine(img, rotation_img, (bound_w, bound_h),
                                 cv2.IMREAD_UNCHANGED, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    radian = -1 * math.pi * (angle / 180)
    rotated_point_groups = []
    for bbox in bboxes:
        xmin_origin = bbox[0] - xmid
        ymin_origin = bbox[1] - ymid
        xmax_origin = bbox[2] - xmid
        ymax_origin = bbox[3] - ymid

        A = point_affine_transformation(xmin_origin, ymin_origin, bound_w, bound_h, radian)
        B = point_affine_transformation(xmin_origin, ymax_origin, bound_w, bound_h, radian)
        C = point_affine_transformation(xmax_origin, ymax_origin, bound_w, bound_h, radian)
        D = point_affine_transformation(xmax_origin, ymin_origin, bound_w, bound_h, radian)

        rotated_point_groups.append([A, B, C, D])

    return rotated_img, rotated_point_groups


def bound_rotated_bbox(rotated_point_groups):
    rotated_bboxes = []
    for group in rotated_point_groups:
        xmin = min(group[0][0], group[1][0], group[2][0], group[3][0])
        xmax = max(group[0][0], group[1][0], group[2][0], group[3][0])
        ymin = min(group[0][1], group[1][1], group[2][1], group[3][1])
        ymax = max(group[0][1], group[1][1], group[2][1], group[3][1])
        rotated_bboxes.append([xmin, ymin, xmax, ymax])

    return rotated_bboxes


def rotated(img, bboxes, angle):
    rotated_img, rotated_point_groups = rotate(img, bboxes, angle)
    rotated_bboxes = bound_rotated_bbox(rotated_point_groups)
    return rotated_img, rotated_bboxes


def draw_points(img, point_groups):
    for group in point_groups:
        A, B, C, D = group
        pts = np.array([A, B, C, D], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)


def draw_rectangle(img, bboxes):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)


def read_img_annotation(pic_dir, annotation_dir):
    img = cv2.imread(pic_dir)
    with open(annotation_dir, 'r') as f:
        annotation = json.load(f)
        bboxes = []
        for bbox in annotation["shapes"]:
            bbox_coordinate = (bbox["points"])
            xmin = int(min(bbox_coordinate[0][0], bbox_coordinate[1][0]))
            ymin = int(min(bbox_coordinate[0][1], bbox_coordinate[1][1]))
            xmax = int(max(bbox_coordinate[0][0], bbox_coordinate[1][0]))
            ymax = int(max(bbox_coordinate[0][1], bbox_coordinate[1][1]))
            bboxes.append([xmin, ymin, xmax, ymax])

    return img, bboxes


if __name__ == '__main__':
    angle = 10
    img_origin_dir = "D:/2021hwsz/data/test/augmentation_test/Images/"
    annotation_origin_dir = "D:/2021hwsz/data/test/augmentation_test/Annotations/"
    img_save_dir = "D:/2021hwsz/data/test/augmentation_test/Rotated_Images/"

    for file in tqdm(os.listdir(img_origin_dir)):
        filename = file.split(".")[0]
        pic_dir = img_origin_dir + file
        annotation_dir = annotation_origin_dir + "/" + filename + ".json"
        img, annotations = read_img_annotation(pic_dir, annotation_dir)

        img, rotated_points = rotate(img, annotations, angle)
        draw_points(img, rotated_points)

        rotated_bboxes = bound_rotated_bbox(rotated_points)
        draw_rectangle(img, rotated_bboxes)

        # cv2.imshow('result', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(img_save_dir + file, img)