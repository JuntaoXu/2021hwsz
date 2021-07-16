import shutil
import os

img_dir = "D:/2021hwsz/data/test/augmentation_test/Images/"
annotation_origin_dir = "D:/2021hwsz/data/base_unzip/Annotations/"
annotation_save_dir = "D:/2021hwsz/data/test/augmentation_test/Annotations/"


def move_selected(img_dir, annotation_origin_dir, annotation_save_dir):
    for file in os.listdir(img_dir):
        filename = file.split(".")[0]
        annotation_origin = annotation_origin_dir + "/" + filename + ".json"
        annotation_save = annotation_save_dir + "/" + filename + ".json"
        shutil.copy(annotation_origin, annotation_save)


move_selected(img_dir, annotation_origin_dir, annotation_save_dir)