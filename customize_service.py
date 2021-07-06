# -*- coding: utf-8 -*-
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
from torchvision.transforms import functional as F
import log
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

logger.info(torch.__version__)
logger.info(torchvision.__version__)


def get_object_detector(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    logger.info('remove pretrained')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    logger.info('{}-{}'.format(in_features, num_classes))
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path, **kwargs):
        self.model_name = model_name
        num_classes = 1 + 10
        logger.info('{}-{}'.format(num_classes, model_path))
        for key in kwargs:
            logger.info('{}-{}'.format(key, kwargs[key]))
        self.model_path = model_path
        self.model = get_object_detector(num_classes)
        self.use_cuda = False
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info('Using GPU for inference')
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.model_path)['model'])
        else:
            logger.info('Using CPU for inference')
            self.model.load_state_dict(torch.load(self.model_path,
                                                  map_location='cpu')['model'])
        self.model.eval()
        print("model already")

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content).convert("RGB")
                preprocessed_data[k] = torch.unsqueeze(F.to_tensor(img),
                                                       dim=0).to(self.device)
        return preprocessed_data

    def _inference(self, data):
        img = data["input_img"]
        data = img
        result = self.model(data)
        for idx in range(len(result)):
            result[idx]['boxes'] = result[idx][
                'boxes'].cpu().detach().numpy().tolist()
            result[idx]['labels'] = result[idx][
                'labels'].cpu().detach().numpy().tolist()
            result[idx]['scores'] = result[idx][
                'scores'].cpu().detach().numpy().tolist()
        result = {"result": result}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data
