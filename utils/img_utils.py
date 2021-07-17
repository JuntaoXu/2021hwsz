import os
import cv2
import json
from tqdm import tqdm

pic_origin_dir = "D:/2021hwsz/data/base_unzip/Images/"
annotation_origin_dir = "D:/2021hwsz/data/base_unzip/Annotations/"
saving_dir = ""

defect_bbox_area = 10

defect_types = {'right_angle_edge_defect': 0, 'connection_edge_defect': 0, 'burr_defect': 0, 'cavity_defect': 0,
                'huahen': 0, 'mosun': 0, 'yanse': 0, 'jianju': 0, 'basi': 0, 'chuizhidu': 0}


def read_img(read_dir):
    return cv2.imread(read_dir)


def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_img(saving_dir, img, file):
    cv2.imwrite(saving_dir + file, img)


def traverse_img_annotation(pic_origin_dir, annotation_dir, saving_dir,
                            defect_types, defect_bbox_area, defect_annotation=0):

    for file in tqdm(os.listdir(pic_origin_dir)):
        filename = file.split(".")[0]

        try:
            os.path.exists(annotation_dir + "/" + filename + ".json")
        except:
            print("corresponding annotation not found for " + file)

        img = read_img(pic_origin_dir + file)

        with open(annotation_dir + filename + ".json", 'r') as f:
            annotation = json.load(f)
            for bbox in annotation["shapes"]:
                defect_type = bbox["label"]
                defect_types[defect_type] += 1
                bbox_coordinate = (bbox["points"])

                try:
                    xmin = int(min(bbox_coordinate[0][0], bbox_coordinate[1][0]))
                    ymin = int(min(bbox_coordinate[0][1], bbox_coordinate[1][1]))
                    xmax = int(max(bbox_coordinate[0][0], bbox_coordinate[1][0]))
                    ymax = int(max(bbox_coordinate[0][1], bbox_coordinate[1][1]))

                    if (xmax - xmin) * (ymax - ymin) < defect_bbox_area:
                        defect_bbox_area += 1
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img, defect_type, (xmin - 350, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                        # show_img(img)
                    if (xmax - xmin) * (ymax - ymin) < 150:
                        defect_bbox_area += 1
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img, defect_type, (xmin - 350, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                        defect_nums_area[defect_types.index(defect_type)] += 1
                        cv2.imshow("img", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    else:
                        continue
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.putText(img, defect_type, (xmin - 350, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

                except:
                    # print(filename)
                    defect_annotation += 1

        # save_img(saving_dir, img, file)
        # cv2.imwrite(saving_dir + file, img)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(defect_types)
    print("number of false annotations: ", defect_annotation)
    print("number of defect_bbox_area: ", defect_bbox_area)


traverse_img_annotation(pic_origin_dir, annotation_origin_dir, saving_dir, defect_types, defect_bbox_area)
