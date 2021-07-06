import os
import cv2
import json
from tqdm import tqdm

pic_origin_dir = "D:/2021hwsz/Images/"
annotation_dir = "D:/2021hwsz/Annotations/"
saving_dir = "D:/2021hwsz/Images_with_bboxes/"

defect_types = ['right_angle_edge_defect', 'connection_edge_defect', 'burr_defect', 'cavity_defect', 'huahen', 'mosun', 'yanse', 'jianju', 'basi', 'chuizhidu']
defect_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
defect_nums_area = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

defect_annotation = 0
defect_bbox_area = 0


def add_bboxes(pic_origin_dir, annotation_dir, saving_dir, defect_types, defect_nums, defect_nums_area, defect_annotation, defect_bbox_area):
    for file in tqdm(os.listdir(pic_origin_dir)):
        filename = file.split(".")[0]

        try:
            os.path.exists(annotation_dir + "/" + filename + ".json")
        except:
            print("corresponding annotation not found for " + file)
            break

        img = cv2.imread(pic_origin_dir + file)

        with open(annotation_dir + filename + ".json", 'r') as f:
            annotation = json.load(f)
            for bbox in annotation["shapes"]:
                defect_type = bbox["label"]
                defect_nums[defect_types.index(defect_type)] += 1

                bbox_coordinate = (bbox["points"])

                try:
                    xmin = int(min(bbox_coordinate[0][0], bbox_coordinate[1][0]))
                    ymin = int(min(bbox_coordinate[0][1], bbox_coordinate[1][1]))
                    xmax = int(max(bbox_coordinate[0][0], bbox_coordinate[1][0]))
                    ymax = int(max(bbox_coordinate[0][1], bbox_coordinate[1][1]))

                    if (xmax - xmin) * (ymax - ymin) < 300:
                        defect_bbox_area += 1
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img, defect_type, (xmin - 350, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                        defect_nums_area[defect_types.index(defect_type)] += 1
                        # cv2.imshow("img", img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                    else:
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.putText(img, defect_type, (xmin - 350, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

                except:
                    defect_annotation += 1

        # cv2.imwrite(saving_dir + file, img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(defect_types)
    print(defect_nums)
    print(defect_nums_area)
    print("number of false annotations: ", defect_annotation)
    print("number of defect_bbox_area: ", defect_bbox_area)


add_bboxes(pic_origin_dir, annotation_dir, saving_dir, defect_types, defect_nums, defect_nums_area, defect_annotation, defect_bbox_area)
