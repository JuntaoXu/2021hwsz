import os
import json
import argparse

# install requirement
# cp pretrained model to working directory
os.system('cd /home/work/user-job-dir/code && pip install -r requirements.txt')
os.system('cd /home/work/user-job-dir/code && '
          'mkdir -p /home/work/.cache/torch/hub/checkpoints/ &&'
          'cp ./pretrained/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth '
          '/home/work/.cache/torch/hub/checkpoints/'
          'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')
os.system('cd /home/work/user-job-dir/code && '
          'mkdir -p /home/work/.cache/torch/hub/checkpoints/ &&'
          'cp ./pretrained/model_last.pth '
          '/home/work/.cache/torch/hub/checkpoints/'
          'model_last.pth')

import numpy as np
from PIL import Image
import moxing as mox
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.engine import train_one_epoch, evaluate
from utils import transforms as T
from utils import utils

# avoid random effect on result
torch.manual_seed(31)
"""
消除随机因素的影响
"""
torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--data_url', type=str, required=True,
                        help='the dataset dir of dataset')
    parser.add_argument('--train_url', type=str, required=True,
                        help='the checkpoint dir obs')
    parser.add_argument('--init_method', type=str, required=False,
                        help='')
    parser.add_argument('--num_gpus', type=int, required=False, default=1,
                        help='')
    parser.add_argument('--last_path', type=str, required=False,
                        help='')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='the dataset dir of training dataset')
    parser.add_argument('--validate-dir', type=str, required=False,
                        default=None,
                        help='the dataset dir of validation dataset')
    parser.add_argument('--ckpt-dir', type=str, required=True,
                        help='the checkpoint dir')

    parser.add_argument('--num-classes', type=int, required=True,
                        help='num-classes, do not include bg')
    parser.add_argument('--batch-size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--num-epochs', type=int, required=True,
                        help='the number of epochs')
    args = parser.parse_args()
    return args


args = parse_args()


class CustomDataset(object):
    def __init__(self, root, transforms, ignore_area=250):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.ignore_area = ignore_area
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.label_mapping = {
            "connection_edge_defect": 1,
            "right_angle_edge_defect": 2,
            "cavity_defect": 3,
            "burr_defect": 4,
            "huahen": 5,
            "mosun": 6,
            "yanse": 7,
            'basi': 8,
            'jianju': 9,
            'chuizhidu': 10
        }

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "Annotations",
                                       self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        with open(annotation_path, 'r') as f:
            annotation_obj = json.load(f)
        # convert the PIL Image into a numpy array
        boxes = []
        labels = []
        area = []
        is_crowd = []
        for instance_obj in annotation_obj['shapes']:

            labels.append(self.label_mapping[instance_obj['label']])
            points = np.asarray(instance_obj['points'])
            cur_box = [
                np.min(points[:, 0]),
                np.min(points[:, 1]),
                np.max(points[:, 0]),
                np.max(points[:, 1])
            ]
            cur_area = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
            if cur_area < self.ignore_area:
                # print('ignore small bbox < '
                #      '{} {}'.format(self.ignore_area,
                #                     os.path.basename(img_path)))
                continue
            boxes.append(
                cur_box
            )
            area.append(
                cur_area
            )
            is_crowd.append(0)
        boxes = np.asarray(boxes, np.float)
        labels = np.asarray(labels, np.float)
        is_crowd = np.asarray(is_crowd, np.int)
        area = np.asarray(area, np.float)
        target = dict()
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(
            [-1, 4])
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(is_crowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform():
    transforms = [T.ToTensor()]
    # transforms.append(T.RandomAutocontrast())
    # applier = T.RandomApply(transforms=transforms, p=0.5)
    return T.Compose(transforms)


def get_object_detector(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_trained_obejct_detector(num_classes):
    model = torch.load('/home/work/.cache/torch/hub/checkpoints/model_last.pth')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if \
        torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 1 + args.num_classes
    # use our dataset and defined transformations
    dataset = CustomDataset(args.train_dir, get_transform())
    if args.validate_dir is not None:
        dataset_test = CustomDataset(args.validate_dir,
                                     get_transform())
    else:
        dataset_test = None
    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)
    else:
        data_loader_test = None

    resume = True
    if resume:
        model = get_trained_obejct_detector(num_classes)
    else:
        # get the model using our helper function
        model = get_object_detector(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.95, weight_decay=0.0005)
    learning_rate = 0.001
    # optimizer = torch.optim.RMSprop(params, lr=learning_rate,
    #                                 alpha=0.99, eps=1e-08, weight_decay=0.0005, momentum=0.95, centered=False)
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.95, weight_decay=0.0005)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=2,
    #                                                gamma=0.2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0.00001, last_epoch=-1)

    # let's train it for 10 epochs
    num_epochs = args.num_epochs
    local_ckpt_path = None
    for epoch in range(num_epochs):
        # train for one epoch, printing every 200 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch,
                        print_freq=200)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if data_loader_test is not None:
            evaluate(model, data_loader_test, device=device)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        if args.ckpt_dir is not None:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
                print('mkdir: {}'.format(args.ckpt_dir))
        local_ckpt_path = os.path.join(args.ckpt_dir,
                                       'model_{}.pth'.format(epoch))
        utils.save_on_master(
            checkpoint, local_ckpt_path)
        if args.train_url is not None:
            # obs://obs-2021hwsz-baseline/data
            obs_target_path = os.path.join('{}'.format(args.train_url),
                                           os.path.basename(local_ckpt_path))
            # OBS 路径每隔30s会将内容同步到线上OBS桶中
            os.system("cp {} {}".format(local_ckpt_path, obs_target_path))
            print('finish upload {}->{}'.format(local_ckpt_path,
                                                obs_target_path))
    if args.train_url is not None and local_ckpt_path is not None and \
            args.last_path is not None:
        # 将最后一个model pth文件上传至模型发布路径下
        obs_target_path = os.path.join('{}'.format(args.last_path),
                                       'model_best.pth')
        print("upload {} -> {}".format(os.path.basename(local_ckpt_path),
                                       obs_target_path))
        os.system("cp {} {}".format(local_ckpt_path, obs_target_path))


def prepare_data(data_url, training_dir):
    print(data_url)
    mox.file.copy_parallel('{}'.format(data_url), training_dir)
    os.system('ls {}'.format(training_dir))
    os.system('cd {} && unzip -qq Images.zip '
              '&& unzip -qq Annotations.zip'.format(training_dir))


if __name__ == '__main__':
    prepare_data(args.data_url, args.train_dir)
    print(args.data_url)
    print(args.train_url)
    print(args.last_path)
    os.system('ls {}'.format(args.train_url))
    main()
