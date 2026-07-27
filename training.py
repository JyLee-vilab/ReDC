import numpy as np
import torch
from tqdm import tqdm

from utils import inverse_sigmoid,cal_iou_batch
from coordinate_level.training import Train_CR, Train_DDE

from sklearn.isotonic import IsotonicRegression
from box_level.scaling import PlattScaling

def Trainer(calibration_info, box_calibration_type, is_dece, lr, epochs, sign_lr, sign_epochs, car_cut=0.9, batch_size=64, seed = 1234):
    """
    Trains the calibration models for each object class.

    For every class, this function prepares the training data,
    trains the Confidence Re-encoder (CR) and the Directional
    Displacement Estimator (DDE), determines the optimal decision
    thresholds for directional prediction, and fits the final
    box-level confidence calibrator.

    Arguments:
        calibration_info (dict)      : calibration data prepared from
                                    the validation set
        box_calibration_type (str)   : confidence calibration method
                                    ('isotonic_regression', or
                                        'platt_scaling')
        is_dece (bool)               : whether to use binary TP/FP
                                    targets (D-ECE setting)
        lr (float)                   : learning rate for the
                                    Confidence Re-encoder (CR)
        epochs (int)                 : number of training epochs for CR
        sign_lr (float)              : learning rate for the Directional
                                    Displacement Estimator (DDE)
        sign_epochs (int)            : number of training epochs for DDE
        car_cut (float)              : CAR threshold used when selecting
                                    samples for directional threshold
                                    optimization
        batch_size (int)             : reserved for future use
        seed (int)                   : random seed for reproducibility

    Returns:
        CRs (dict)                   : trained Confidence Re-encoder
                                    models for each class
        DDEs (dict)                  : trained Directional Displacement
                                    Estimator models for each class
        box_calibrators (dict)       : trained confidence calibration
                                    models for each class
        thr (dict)                   : optimized decision thresholds
                                    for directional prediction
        sign_norm_stats (dict)       : feature normalization statistics
                                    used for DDE training and inference

    Note:
        The final confidence calibrator is trained using the outputs
        of the CR and DDE models. Depending on
        `box_calibration_type`, either Isotonic Regression or
        Platt Scaling is used to calibrate the predicted confidence.
    """
    if box_calibration_type is None:
        return None, None
        
    CRs = dict()
    DDEs = dict()
    box_calibrators = dict()
    thr = dict()
    sign_norm_stats = {}

    for cl, cl_input in tqdm(calibration_info.items()):
        # For corrupted images, all images with this class is already rejected
        if 'tps' not in cl_input.keys():
            CRs[cl] = np.zeros(0)
            DDEs[cl] = np.zeros(0)
            box_calibrators[cl] = np.zeros(0)
            continue

        # Find total number of valid detections for this class
        valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
        # valid_dets = cl_input['tps']
        num_valid_dets = valid_dets.sum()

        # If no detection, then ignore
        if num_valid_dets == 0:
            CRs[cl] = np.zeros(0)
            DDEs[cl] = np.zeros(0)
            continue

        # Note that scores and ious are sorted wrt scores
        valid_feature = np.asarray(cl_input['feature'][valid_dets])

        valid_logit = inverse_sigmoid(cl_input['scores'][valid_dets]).reshape(-1, 1)

        if is_dece:
            valid_cls_labels = cl_input['tps'][valid_dets]
            valid_target = [
                1.0 if label else 0.0 for label in valid_cls_labels]

        else:
            valid_target = cl_input['iou'][valid_dets]
            
        valid_labels = np.asarray(cl_input['car'])[valid_dets]
        valid_bbox = np.asarray(cl_input['bbox'])[valid_dets].reshape(-1, 4)
        valid_sign = np.asarray(cl_input['sign'])[valid_dets]
        valid_logits = np.asarray(cl_input['all_logits'])[valid_dets]

        cx = valid_bbox[:,0]
        cy = valid_bbox[:,1]
        w  = valid_bbox[:,2]
        h  = valid_bbox[:,3]

        area = w * h
        aspect = w / (h + 1e-6)

        bbox_features = np.stack([
            cx, cy, w, h, area,aspect
        ], axis=1)

        CR_input = np.concatenate(
            [valid_logit, valid_feature],
            axis=1)

        DDE_input = np.concatenate(
            [valid_logits, bbox_features],
            axis=1)

        mean = DDE_input.mean(axis=0, keepdims=True)
        std = DDE_input.std(axis=0, keepdims=True) + 1e-6

        sign_norm_stats[cl] = {
            "mean": mean,
            "std": std
        }
        DDE_input = (DDE_input - mean) / std

        CRs[cl] = Train_CR(CR_input, valid_labels, lr=lr, epochs=epochs, seed=seed)

        DDEs[cl] = Train_DDE(DDE_input, valid_sign, valid_labels, car_cut = car_cut, lr=sign_lr, epochs=sign_epochs, seed=seed)

        CRs[cl].eval()
        DDEs[cl].eval()

        device = next(CRs[cl].parameters()).device

        CR_inputs  = torch.from_numpy(CR_input).float().to(device)
        DDE_inputs = torch.from_numpy(DDE_input).float().to(device)

        with torch.no_grad():

            value_logits = CRs[cl](CR_inputs)
            sign_logits  = DDEs[cl](DDE_inputs)

            pred_value = torch.sigmoid(value_logits)
            pred_sign = torch.sigmoid(sign_logits)

        gt = np.where(valid_sign > 0, 1, -1)

        pred_sign_np = pred_sign.detach().cpu().numpy()

        thr_cl = np.zeros(4, dtype=np.float32)
        thr_acc_cl = np.zeros(4, dtype=np.float32)

        for dim in range(4):

            mask = (np.abs(valid_labels[:, dim]) < car_cut) & (valid_sign[:, dim] != 0)

            if mask.sum() == 0:
                thr_cl[dim] = 0.5
                thr_acc_cl[dim] = np.nan
                continue

            gt_dim = gt[mask, dim]
            prob_dim = pred_sign_np[mask, dim]

            best_score = -1.0
            best_thr = 0.5

            for t in np.linspace(0.4, 0.6, 21):
                pred_dim = np.where(prob_dim > t, 1, -1)
                acc = (pred_dim == gt_dim).mean()

                if acc > best_score:
                    best_score = acc
                    best_thr = t

            thr_cl[dim] = best_thr
            thr_acc_cl[dim] = best_score
        thr[cl] = thr_cl.copy()

        threshold = torch.tensor(thr_cl, device=pred_sign.device, dtype=pred_sign.dtype).view(1, 4)
        
        thresholded_sign = torch.where(
            pred_sign > threshold,
            torch.tensor(1.0, device=pred_sign.device, dtype=pred_sign.dtype),
            torch.tensor(-1.0, device=pred_sign.device, dtype=pred_sign.dtype)
        )

        signed_value = pred_value * thresholded_sign
        pred_iou = cal_iou_batch(signed_value).detach().cpu().numpy()
        
        if box_calibration_type == 'isotonic_regression':
            calibrator = IsotonicRegression()
            calibrator.fit(pred_iou.reshape(-1, 1), valid_target)

        elif box_calibration_type == 'platt_scaling':
            calibrator = PlattScaling()
            calibrator.fit(pred_iou, valid_target)

        else:
            raise ValueError(f"Unsupported calibration type: {box_calibration_type}")

        box_calibrators[cl] = calibrator

    return CRs, DDEs, box_calibrators, thr, sign_norm_stats