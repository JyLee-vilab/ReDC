import numpy as np

def distance(pred, gt):
    """
    Computes the absolute distance and its corresponding sign between
    the predicted value and the ground truth.

    Arguments:
        pred (float or np.ndarray) : predicted value(s)
        gt (float or np.ndarray)   : ground truth value(s)

    Returns:
        distance (float or np.ndarray) : absolute difference |pred - gt|
        sign (int or np.ndarray)       : sign of the difference
                                         (1 if pred >= gt, -1 otherwise)

    Note:
        If the difference is zero, the sign is set to 1 instead of 0
        to avoid undefined behavior in subsequent computations.
    """

    diff = pred - gt

    distance = np.abs(diff)
    sign = np.sign(diff)
    
    if sign == 0:
        sign = 1
        
    return distance, sign

def xywh_to_xyxy(boxes):
    """
    Converts bounding box coordinates from (x, y, w, h) format
    to (x1, y1, x2, y2) format.

    Arguments:
        boxes (list or np.ndarray) : bounding box(es) in
                                     (x, y, width, height) format.
                                     Can be a single box with shape (4,)
                                     or multiple boxes with shape (N, 4).

    Returns:
        list : bounding box(es) in (x1, y1, x2, y2) format, where
               x1, y1 : top-left corner
               x2, y2 : bottom-right corner
    """

    boxes = np.array(boxes)
    if boxes.ndim == 1:
        x, y, w, h = boxes
        return [x, y, x + w, y + h]
    return [[x, y, x + w, y + h] for x, y, w, h in boxes]


def compute_intersection(box1, box2):
    """
    Computes the width and height of the intersection region
    between two bounding boxes.

    Arguments:
        box1 (list or np.ndarray) : first bounding box in
                                    (x1, y1, x2, y2) format
        box2 (list or np.ndarray) : second bounding box in
                                    (x1, y1, x2, y2) format

    Returns:
        inter_w (float) : width of the intersection region
        inter_h (float) : height of the intersection region

    Note:
        If the two bounding boxes do not overlap, both the width
        and height of the intersection are returned as 0.
    """
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    return inter_w, inter_h

def calculateCAR(dt_box, gt_box):
    """
    Computes the Coordinate-wise Alignment Ratio (CAR) between a predicted
    bounding box and a ground-truth bounding box.

    The CAR is calculated independently for each box coordinate
    (x1, y1, x2, y2) based on the overlap size and the coordinate
    distance. The sign of each value indicates whether the predicted
    coordinate is greater than or less than the corresponding
    ground-truth coordinate.

    Arguments:
        dt_box (list or np.ndarray) : predicted bounding box in
                                      (x, y, width, height) format
        gt_box (list or np.ndarray) : ground-truth bounding box in
                                      (x, y, width, height) format

    Returns:
        np.ndarray : Coordinate Alignment Ratio (CAR) for
                     (x1, y1, x2, y2) with shape (4,).

    Note:
        If the predicted and ground-truth boxes do not overlap,
        a zero vector is returned.
    """

    pred = xywh_to_xyxy(dt_box)
    gt = xywh_to_xyxy(gt_box)

    inter_w, inter_h = compute_intersection(pred, gt)

    if inter_w <= 0 or inter_h <= 0:
        return np.zeros(4, dtype=np.float32)

    car_pair = []

    for coor in range(len(pred)):

        dist, sign = distance(pred[coor], gt[coor])

        if coor in [0, 2]:
            car = inter_w / (dist + inter_w)
        else:
            car = inter_h / (dist + inter_h)

        car_pair.append(car * sign)

    return np.array(car_pair, dtype=np.float32)