import numpy as np
import matplotlib.pyplot as plt
import json

from pycocotools_lrp_ReDC.coco import COCO
from pycocotools_lrp_ReDC.cocoeval import COCOeval

import torch
import pickle
import os
import torch.nn.functional as F

def load_detections_from_file(path, dataset_classes):
    """
    Loads object detections from a file.

    This function supports both COCO-style JSON files and pickle files
    containing detector outputs. Detection results stored in pickle
    format are converted into the COCO detection format required by
    the calibration pipeline.

    Arguments:
        path (str)                 : path to the detection file
        dataset_classes (list)     : mapping from class indices to
                                    dataset category IDs

    Returns:
        list : list of detections in COCO format, where each detection
            contains the image ID, bounding box, confidence score,
            category ID, classification logits, class-wise scores,
            class-wise logits, and query features (bounding box features).

    Note:
        - JSON files are loaded directly without modification.
        - Pickle files are expected to contain bounding boxes,
        confidence scores, class labels, logits, class-wise scores,
        class-wise logits, and query features, which are converted
        into COCO-style detection dictionaries.
    """
    ext = os.path.splitext(path)[1].lower()


    
    if ext == '.json':
        f = open(path)
        detections = json.load(f)
        f.close()

    elif ext in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        detections = []

        for image_id, det in data.items():
            boxes = det['bboxes'] 
            scores = det['scores']

            if 'labels' in det:
                labels = det['labels']
            elif 'category_ids' in det:
                labels = det['category_ids']
            else:
                raise KeyError("No label key found")
            logits = det['logits']
            all_scores = det['all_scores']
            all_logits = det['all_logits']
            features = det['bbox_feature']

            num_det = boxes.shape[0]

            for i in range(num_det):
                x1, y1, x2, y2 = boxes[i]

                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)

                sample = {
                    'image_id': image_id,
                    'bbox': [x, y, w, h],
                    'score': float(scores[i]),
                    'category_id': dataset_classes[int(labels[i])],
                    'logit': logits[i],
                    'all_scores': all_scores[i],
                    'all_logits': all_logits[i],
                    'bbox_feature': features[i],
                }
                detections.append(sample)
    
    return detections

def inverse_sigmoid(confidences):
    """
    Converts confidence scores into logit values by applying the
    inverse sigmoid (logit) transformation.

    This function supports both NumPy arrays and PyTorch tensors.
    Input confidence values are clipped to avoid numerical instability
    when computing the logarithm.

    Arguments:
        confidences (torch.Tensor or np.ndarray) : confidence scores
                                                in the range [0, 1]

    Returns:
        torch.Tensor or np.ndarray : logit values with the same type
                                    as the input.

    Note:
        Confidence values are clipped to the valid open interval
        (ε, 1−ε) before applying the inverse sigmoid to prevent
        numerical overflow when the input contains values equal
        to 0 or 1.
    """
    if isinstance(confidences, torch.Tensor):
        eps = torch.finfo(confidences.dtype if confidences.is_floating_point() else torch.float32).eps
        clipped = torch.clamp(confidences, eps, 1. - eps)
        inv_clipped = torch.clamp(1. - confidences, eps, 1. - eps)
        return torch.log(clipped) - torch.log(inv_clipped)

    confidences = np.asarray(confidences, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(confidences, eps, 1. - eps)
    inv_clipped = np.clip(1. - confidences, eps, 1. - eps)
    return np.log(clipped) - np.log(inv_clipped)


def threshold_detections(detections, detection_level_threshold, dataset_classes):
    """
    Filters detections based on class-specific confidence thresholds.

    Each detection is retained only if its confidence score is greater
    than or equal to the threshold assigned to its predicted class.

    Arguments:
        detections (list)                  : list of model detections
        detection_level_threshold (float
            or array-like)                 : confidence threshold(s) for
                                            filtering detections. A
                                            single value is applied to
                                            all classes, while an array
                                            specifies one threshold per
                                            class.
        dataset_classes (list)             : dataset category IDs used to
                                            map detections to their
                                            corresponding thresholds

    Returns:
        list : detections that satisfy the corresponding confidence
            thresholds.

    Note:
        Detections with confidence scores below the threshold of their
        predicted class are removed from the returned list.
    """

    detection_level_threshold = np.asarray(detection_level_threshold)

    if detection_level_threshold.ndim == 0:
        detection_level_threshold = np.array([float(detection_level_threshold)])

    del_items = []
    for idx, detection in enumerate(detections):
        if detection['score'] < detection_level_threshold[dataset_classes.index(detection['category_id'])]:
            del_items.append(idx)
    for idx in sorted(del_items, reverse=True):
        del detections[idx]
    return detections


def get_detection_thresholds(annFile, detections, benchmark, thr, tau, eval_type='bbox', max_dets=[100]):
    """
    Computes class-specific detection confidence thresholds.

    This function evaluates the detections on the validation set and
    determines the confidence threshold for each class. Thresholds can
    either be obtained automatically using the LRP-optimal operating
    point or set to a fixed confidence value.

    Arguments:
        annFile (str)               : file path to the ground-truth
                                    annotations
        detections (list)           : list of model detections
        benchmark (str)             : benchmark type
        thr (float)                 : confidence threshold. If set to
                                    -1, LRP-optimal thresholds are
                                    computed automatically;
                                    otherwise, the specified value is
                                    used for all classes.
        tau (float)                 : IoU threshold used to determine
                                    true positives and false positives
        eval_type (str)             : evaluation type, either 'bbox'
                                    or 'segm'
        max_dets (list)             : maximum number of detections per
                                    image considered during evaluation

    Returns:
        np.ndarray : class-specific detection confidence thresholds.

    Note:
        If `thr == -1`, the thresholds are obtained from the
        LRP-optimal operating points computed by COCO evaluation.
        Otherwise, a fixed threshold is assigned to every class.
    """
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(detections)
    id_evaluator = COCOeval(cocoGt, cocoDt, eval_type)

    id_evaluator.params.areaRng = [id_evaluator.params.areaRng[0]]
    id_evaluator.params.areaRngLbl = ['all']
    id_evaluator.params.iouThrs = np.array([tau])
    id_evaluator.params.maxDets = max_dets

    id_evaluator.evaluate()
    id_evaluator.accumulate()

    # LRP-Optimal Thresholds
    if thr == -1:
        print('Obtaining detection-level threshold using LRP-optimal thresholds...')
        return id_evaluator.eval['lrp_opt_thr'].squeeze()
    else:
        print('Obtaining detection-level threshold using a fixed confidence score...')
        return np.ones(len(id_evaluator.eval['lrp_opt_thr'])) * thr


def cal_iou(sample):
    """
    Computes the approximated IoU from coordinate-wise calibration
    outputs.

    The input consists of four coordinate-wise confidence values with
    their associated directions encoded in the sign. The function
    reconstructs an approximation of the object box-level quality
    using the proposed geometric formulation.

    Arguments:
        sample (array-like)         : four coordinate-wise calibration
                                    outputs in the form
                                    [x1, y1, x2, y2]

    Returns:
        float : approximated IoU computed from the coordinate-wise
                confidence values.

    Note:
        The sign of each coordinate is used to model the direction of
        the localization error, while the magnitude represents the
        confidence of the coordinate prediction.
    """
    x1,y1,x2,y2 = sample

    a = np.abs(x1)
    b = np.abs(y1)
    c = np.abs(x2)
    d = np.abs(y2)

    sx1 = np.sign(x1)
    sy1 = np.sign(y1)
    sx2 = np.sign(x2)
    sy2 = np.sign(y2)
    
    EPS = 1e-12

    alpha = ((1/(a+EPS))+(1/(c+EPS))-1.0)
    beta = ((1/(b+EPS))+(1/(d+EPS))-1.0)
    
    omega = (((1/(a+EPS)) - 1.0) * ((1/(b+EPS)) - 1.0))
    xai   = (((1/(a+EPS)) - 1.0) * ((1/(d+EPS)) - 1.0))
    gamma = (((1/(c+EPS)) - 1.0) * ((1/(b+EPS)) - 1.0))
    seta  = (((1/(c+EPS)) - 1.0) * ((1/(d+EPS)) - 1.0))

    omega = omega * ((sx1 * sy1) < 0)
    xai   = xai   * ((sx1 * sy2) < 0)
    gamma = gamma * ((sx2 * sy1) < 0)
    seta  = seta  * ((sx2 * sy2) < 0)
    
    error = omega + xai + gamma + seta
    iou_app = 1 / (((alpha * beta))-error+EPS)

    return iou_app


def cal_iou_batch(samples, eps: float = 1e-12):
    """
    Computes the approximated IoU for a batch of coordinate-wise
    calibration outputs.

    This function is the batched PyTorch implementation of
    `cal_iou()`, enabling efficient inference on multiple detections
    simultaneously.

    Arguments:
        samples (torch.Tensor or np.ndarray) : coordinate-wise
                                            calibration outputs of
                                            shape (N, 4), where each
                                            row is
                                            [x1, y1, x2, y2]
        eps (float)                          : small constant used for
                                            numerical stability

    Returns:
        torch.Tensor : approximated IoU values of shape (N,).

    Note:
        The sign of each coordinate represents the predicted direction
        of the localization error, while its magnitude indicates the
        confidence of the coordinate prediction. NumPy inputs are
        automatically converted to PyTorch tensors before computation.
    """
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)

    samples = samples.float()

    x1, y1, x2, y2 = samples.unbind(dim=1)

    a = torch.abs(x1)
    b = torch.abs(y1)
    c = torch.abs(x2)
    d = torch.abs(y2)

    sx1 = torch.where(x1 >= 0, 1.0, -1.0)
    sy1 = torch.where(y1 >= 0, 1.0, -1.0)
    sx2 = torch.where(x2 >= 0, 1.0, -1.0)
    sy2 = torch.where(y2 >= 0, 1.0, -1.0)

    EPS = eps
    EPS = eps

    alpha = (1.0/(a+EPS) + 1.0/(c+EPS) - 1.0)
    beta  = (1.0/(b+EPS) + 1.0/(d+EPS) - 1.0)

    omega = ((1.0/(a+EPS) - 1.0) * (1.0/(b+EPS) - 1.0))
    xai   = ((1.0/(a+EPS) - 1.0) * (1.0/(d+EPS) - 1.0))
    gamma = ((1.0/(c+EPS) - 1.0) * (1.0/(b+EPS) - 1.0))
    seta  = ((1.0/(c+EPS) - 1.0) * (1.0/(d+EPS) - 1.0))
    
    omega = omega * ((sx1 * sy1) < 0).float()
    xai   = xai   * ((sx1 * sy2) < 0).float()
    gamma = gamma * ((sx2 * sy1) < 0).float()
    seta  = seta  * ((sx2 * sy2) < 0).float()

    error = omega + xai + gamma + seta

    denom = (alpha * beta) - error
    denom = denom.clamp_min(EPS)

    
    iou_app = 1.0 / denom
    
    return iou_app


def COCO_evaluation(annFile, detections, eval_type='bbox', valid_img=None, remove_img=None, tau=None):
    """
    Performs standard COCO evaluation on a set of detections.

    This function evaluates the detections using the official COCO
    evaluation protocol and reports the standard detection metrics,
    including AP, AR, and LRP. Evaluation can optionally be restricted
    to a subset of images or performed at a specific IoU threshold.

    Arguments:
        annFile (str)              : file path to the ground-truth
                                    annotations
        detections (list)          : list of model detections
        eval_type (str)            : evaluation type, either 'bbox'
                                    or 'segm'
        valid_img (list or None)   : image IDs to include in the
                                    evaluation. If None, all images
                                    are evaluated.
        remove_img (list or None)  : image IDs to exclude from the
                                    evaluation. Ignored when None.
        tau (float or None)        : IoU threshold used for evaluation.
                                    If None, the default COCO IoU
                                    thresholds are used.

    Returns:
        COCOeval : initialized and evaluated COCO evaluation object.

    Note:
        When `verbose=True` in the calibration pipeline, this function
        is used to report the standard COCO detection metrics.
    """

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(detections)
    id_evaluator = COCOeval(cocoGt, cocoDt, eval_type)
    if tau:
        id_evaluator.params.iouThrs = np.array([tau])
    if remove_img is not None:
        id_evaluator.params.imgIds = list(
            set(id_evaluator.params.imgIds).difference(remove_img))
    elif valid_img is not None:
        id_evaluator.params.imgIds = list(valid_img)
    id_evaluator.evaluate()
    id_evaluator.accumulate()
    id_evaluator.summarize()
    return id_evaluator


def to_onehot(cls, num_classes=16):
    """
    Converts class indices into one-hot encoded vectors.

    Arguments:
        cls (torch.Tensor)         : tensor containing class indices
        num_classes (int)          : total number of classes

    Returns:
        torch.Tensor : one-hot encoded class labels with shape
                    (..., num_classes).
    """
    return F.one_hot(cls, num_classes).float()