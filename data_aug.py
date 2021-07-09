import cv2
import json
import glob
import os
import tqdm
import numpy as np
from data_augmentation import *
import datetime

HARD_CLASS = ['cavity_defect', 'chuizhidu']
COMMON_CLASS = ['huahen', 'mosun', 'jianju', 'basi']


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

def read_imgs(imgpath, savepath):
    imgs = glob.glob(os.path.join(imgpath,'*g'))
    for img in tqdm.tqdm(imgs):
        json_file = img.replace('jpg', 'json').replace('Images', 'Annotations')
        f = open(json_file)
        data = json.load(f)
        f.close()
        ori_shapes = data['shapes']
        bboxes = list()
        loop_count = list()
        hard = dict()
        common = dict()
        for s in ori_shapes:
            # print(s)
            pts = np.asarray(s['points'])
            # print(type(pts))
            bbox = [
                np.min(pts[:, 0]),
                np.min(pts[:, 1]),
                np.max(pts[:, 0]),
                np.max(pts[:, 1])
            ]
            if bbox is None: print('no bbx:', img)
            bboxes.append(bbox)
            label = s['label']
            if label in HARD_CLASS:
                if label in hard:
                    hard[label].append(bbox)
                else: hard[label] = [bbox]
                # print('addlabel:', hard[label])
                loop = 8
            elif label in COMMON_CLASS:
                # print("common is", common)
                # print(bbox)
                if label in common:
                    common[label].append(bbox)
                else:
                    common[label] = [bbox]

                loop = 4
            else:
                loop = 2
            loop_count.append(loop)
        # set a min data augmentation number
        base_loop = min(loop_count)
        loop_count = [i-base_loop for i in loop_count]

        I = cv2.imread(img)

        for i in range(base_loop):
            # print(bboxes)
            newI, newbboxes = aug(I, bboxes)
            imgname = os.path.split(img)[-1].split('.')[0] + '_' + str(i)
            cv2.imwrite(os.path.join(savepath,'Images',imgname + '.jpg'), newI)
            shapes = data['shapes']
            for idx, s in enumerate(shapes):
                newbbox = newbboxes[idx]
                newbbox = [[newbbox[0], newbbox[1]], [newbbox[2], newbbox[3]]]
                s['points'] = newbbox
            data['shapes'] = shapes
            print('shapes: ', data['shapes'])

            # Writing to sample.json
            with open(os.path.join(savepath,'Annotations',imgname + '.json'), 'w') as outfile:
                # print(type(data))
                json.dump(data, outfile, indent=4, default=myconverter)
        if max(loop_count) < 1: continue
        # print(loop_count)
        # do extra loop for hard classes and change the corresponding json file
        if len(hard) >= 1:
            # print(hard)
            shapes = ori_shapes
            bboxes = list()
            for h in hard:
                # print(hard[h])
                bboxes.extend(hard[h])
            for i in range(8-base_loop):
                newI, newbboxes = aug(I, bboxes)
                imgname = os.path.split(img)[-1].split('.')[0] + '_' + 'hard' + '_' + str(i)
                cv2.imwrite(os.path.join(savepath,'Images',imgname + '.jpg'), newI)
                shapes = [i for i in shapes if i['label'] in hard.keys()]
                if len(shapes) < 1: continue
                # print(shapes)
                for idx, s in enumerate(shapes):
                    newbbox = newbboxes[idx]
                    newbbox = [[newbbox[0], newbbox[1]], [newbbox[2], newbbox[3]]]
                    s['points'] = newbbox
                data['shapes'] = shapes
                print('shapes: ', data['shapes'])
                with open(os.path.join(savepath, 'Annotations', imgname + '.json'), "w") as outfile:
                    json.dump(data, outfile, indent=4, default=myconverter)

        if len(common) >= 1 and common is not None:
            # print(common)
            shapes = ori_shapes
            bboxes = list()
            for h in common:
                if common[h] is None:
                    # print(h, img)
                    continue
                bboxes.extend(common[h])
            for i in range(8-base_loop):
                # print(bboxes)
                newI, newbboxes = aug(I, bboxes)
                imgname = os.path.split(img)[-1].split('.')[0] + '_' + 'common' + '_' + str(i)
                cv2.imwrite(os.path.join(savepath,'Images',imgname + '.jpg'), newI)
                shapes = [i for i in shapes if i['label'] in common.keys()]
                for idx, s in enumerate(shapes):
                    newbbox = newbboxes[idx]
                    newbbox = [[newbbox[0], newbbox[1]], [newbbox[2], newbbox[3]]]
                    s['points'] = newbbox
                data['shapes'] = shapes
                print('shapes: ', data['shapes'])

                # Writing to sample.json
                with open(os.path.join(savepath, 'Annotations', imgname + '.json'), "w") as outfile:
                    json.dump(data, outfile, indent=4, default=myconverter)


def aug(img, bboxes, degree=10):
    '''

    Args:
        img: image matrix
        bboxes: list of boxes

    Returns: the augmented image, and new bboxes with respect to the new image

    '''
    print('ori: ', bboxes)
    random_transform = [rotate_img_bboxes, flip_pic_bboxes, random_crop_boxes, alterLight]
    num_transform = random.randint(1,3)
    randomlist = random.sample(range(0, 4), num_transform)
    for i in randomlist:
        if i == 0:
            rotate_degrees = random.randint(-degree, degree)
            img, bboxes = random_transform[i](img, bboxes, rotate_degrees)
            print('rotated: ', bboxes)
        elif i == 3:
            img = random_transform[i](img)

        else:
            # print(bboxes)
            img, bboxes = random_transform[i](img, bboxes)
            print('flipped or cropped: ',bboxes)
    print('return: ', bboxes)
    return img, bboxes

if __name__ == '__main__':
    imgpath = 'Images'
    savepath = '/Users/staceywang/Downloads/aug0707'
    if not os.path.exists(savepath):
        os.makedirs(os.path.join(savepath,'Images'))
        os.makedirs(os.path.join(savepath, 'Annotations'))
    read_imgs(imgpath,savepath)