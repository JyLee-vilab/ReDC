import torch
import numpy as np
import matplotlib.pyplot as plt

from pycocotools_lrp_ReDC.coco import COCO
from pycocotools_lrp_ReDC.cocoeval import COCOeval

from utils import threshold_detections, get_detection_thresholds, cal_iou, load_detections_from_file, COCO_evaluation, inverse_sigmoid
from training import Trainer


class ReDCCOCO(COCOeval):
    def __init__(self, val_annotations, test_annotations, eval_type='bbox', bin_count=20, tau=0.0, is_dece=False, is_ace=False, is_coordinate_wise = True, max_dets=100):
        """
        Class for learning a post-hoc calibrator, calibrating the
        outputs of an object detector, and performing joint
        accuracy/calibration benchmarking.

        Arguments:
            val_annotations (str)     : file path to validation set annotations
            test_annotations (str)    : file path to test set annotations
            eval_type (str)           : evaluation type, either 'bbox' or 'segm'
            bin_count (int)           : number of bins used for calibration evaluation
            tau (float)               : IoU threshold for determining TP/FP during evaluation
            is_dece (bool)            : whether to use D-ECE-style binary (TP/FP) targets
            is_ace (bool)             : whether to evaluate using Adaptive Calibration Error (ACE)
            is_coordinate_wise (bool) : whether to perform coordinate-wise bounding box calibration
            max_dets (int)            : maximum number of detections per image

        For the remaining attributes, please refer to the base class (COCOeval).
        """

        self.val_annotations = val_annotations
        self.test_annotations = test_annotations

        super(ReDCCOCO, self).__init__(
            cocoGt=COCO(val_annotations), iouType=eval_type)
        self.dataset_classes = list(COCO(val_annotations).cats.keys())

        self.num_cls = len(self.dataset_classes)

        # COCOeval related parameters
        self.params.areaRng = [self.params.areaRng[0]]
        self.params.areaRngLbl = ['all']
        self.params.iouThrs = np.array([tau])

        # usually max_dets=100 for COCO
        self.params.maxDets = [max_dets]

        # evaluation-specific parameters
        self.tau = tau

        self.is_dece = is_dece
        self.is_ace = is_ace

        # For ReDC training and evaluation
        self.is_coordinate_wise = is_coordinate_wise

        self.eval_type = eval_type
        self.bin_count = bin_count
        self.bins = np.linspace(0.0, 1.0, self.bin_count + 1, endpoint=True)
        self.bins_sign = np.linspace(-1.0, 1.0, self.bin_count + 1, endpoint=True)

        # Calibrator-specific options can be further set with fit()
        self.box_calibration_type = 'platt_scaling'

        self.calibration_info = dict()
        self.calibration_info_all = dict()

        # Follow D-ECE-style evaluation directly
        if self.is_dece:
            self.errors = np.zeros(self.bin_count)
            self.weights_per_bin = np.zeros(self.bin_count)
            self.prec_iou = np.zeros(self.bin_count)

        else:
            # For LaACE_0, follow class-wise and bin_width==1 strategy
            if self.is_ace:
                self.errors = np.zeros(len(self.params.catIds))
                self.weights_per_bin = np.zeros(len(self.params.catIds))
                self.prec_iou = np.zeros(len(self.params.catIds))

            # Else, follow LaECE-style binning
            else:
                self.errors = np.zeros(
                    [len(self.params.catIds), self.bin_count])
                self.weights_per_bin = np.zeros(
                    [len(self.params.catIds), self.bin_count])
                self.prec_iou = np.zeros(
                    [len(self.params.catIds), self.bin_count])
        
        self.cw_errors = np.zeros([len(self.params.catIds), self.bin_count, 4])
        self.weights_per_bin_cw_errors = np.zeros([len(self.params.catIds), self.bin_count, 4])
        self.prec_car = np.zeros([len(self.params.catIds), self.bin_count, 4])
        
        self.cw_errors_sign = np.zeros([self.num_cls, self.bin_count, 4])
        self.weights_per_bin_cw_errors_sign = np.zeros([self.num_cls, self.bin_count, 4])
        self.prec_car_sign = np.zeros([self.num_cls, self.bin_count, 4])
            
        self.lrps = {'lrp': np.zeros(len(self.params.catIds)) - 1, 'lrp_loc': np.zeros(len(self.params.catIds)) - 1,
                     'lrp_fp': np.zeros(len(self.params.catIds)) - 1, 'lrp_fn': np.zeros(len(self.params.catIds)) - 1}
        
        self.cw_error_name = ["Cₓ₁-ECE", "Cᵧ₁-ECE", "Cₓ₂-ECE", "Cᵧ₂-ECE"]
        self.Dauce_name = ["Daₓ₁-CE", "Daᵧ₁-CE", "Daₓ₂-CE", "Daᵧ₂-CE"]

    
    def prepare_input(self, p=None, eval_cal = False):
        """
        Accumulate per image evaluation results and
        store the result in self.eval
        
        Arguments:
            p (Params, optional) : evaluation parameters. If None,
                                the default COCO evaluation
                                parameters are used.
            eval_cal (bool)      : whether to load additional calibration
                                outputs (e.g., coordinate-wise scores
                                and predicted signs) for evaluation.

        Returns:
            dict : calibration information indexed by category, containing
                detection scores, TP/FP labels, IoUs, CAR values,
                bounding box features, image IDs, and other metadata
                required for calibration training or evaluation.
        """

        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            self.calibration_info[k] = dict()
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])
                    
                    if self.is_coordinate_wise:
                        arrays = {}

                        for key, dtype in (
                            ("dtfeature", np.float32),
                            ("dtlogits", None),
                        ):
                            items = [
                                np.atleast_2d(np.array(v, dtype=dtype))
                                for e in E
                                for v in e[key][:maxDet]
                            ]

                            arrays[key] = (
                                np.concatenate(items, axis=0)
                                if items
                                else np.empty((0, self.num_cls), dtype=dtype)
                            )

                        dtfeature = arrays["dtfeature"]
                        dtlogits = arrays["dtlogits"]
                        

                    # different sorting method generates slightly
                    # different results.
                    # mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]
                    
                    if self.is_coordinate_wise == True:
                        dtfeature = dtfeature[inds]
                        dtlogits  = dtlogits[inds]

                        self.calibration_info[k]['all_logits'] = dtlogits
                        self.calibration_info[k]['feature'] = dtfeature

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]

                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    dtIoU = np.concatenate(
                        [e['dtIoUs'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    
                    dtCAR = np.concatenate(
                        [e['dtCARs'][:, 0:maxDet, :] for e in E], axis=1)[:, inds, :]
                                        
                    dtbbox = np.concatenate(
                        [np.asarray(e['dtbbox'])[0:maxDet] for e in E], axis=0)[inds]

                    dtSign = np.concatenate(
                        [np.asarray(e['dtSIGN'])[0:maxDet] for e in E], axis=0
                    )[inds]

                    bbox = dtbbox.astype(np.float32)

                    cxcywh = np.zeros_like(bbox)
                    cxcywh[:, 0] = bbox[:, 0] + 0.5 * bbox[:, 2]   # cx
                    cxcywh[:, 1] = bbox[:, 1] + 0.5 * bbox[:, 3]   # cy
                    cxcywh[:, 2] = bbox[:, 2]                      # w
                    cxcywh[:, 3] = bbox[:, 3]                      # h
                    
                    dtImg = np.concatenate([e['image_id'][0:maxDet] for e in E]).astype(np.int64)

                    if eval_cal:
                        for key in ("cw_score", "pred_sign"):
                            items = [
                                np.array(v).reshape(1, -1)
                                for e in E if key in e
                                for v in e[key][:maxDet]
                            ]

                            self.calibration_info[k][key] = (
                                np.vstack(items)[inds, :]
                                if items
                                else np.zeros((0, 4))
                            )
                
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue

                    self.calibration_info[k]['scores'] = dtScoresSorted
                    self.calibration_info[k]['tps'] = np.logical_and(
                        dtm, np.logical_not(dtIg))[0]

                    self.calibration_info[k]['fps'] = np.logical_and(np.logical_not(dtm),
                                                                     np.logical_not(dtIg))[0]

                    self.calibration_info[k]['iou'] = np.multiply(
                        dtIoU, self.calibration_info[k]['tps'])[0]
                    
                    self.calibration_info[k]['npig'] = npig

                    dtCAR = np.array(dtCAR).reshape(-1, 4)
                    tps = np.array(self.calibration_info[k]['tps']).reshape(-1, 1)
                    self.calibration_info[k]['car'] = dtCAR * tps
                    self.calibration_info[k]['image_id'] = dtImg
                    
                    self.calibration_info[k]['bbox'] = cxcywh
                    self.calibration_info[k]['sign'] = dtSign
                    

        return self.calibration_info
    

    def combine_calibration_info(self):
        """
        Combines per-class calibration information into a single
        class-agnostic representation for D-ECE evaluation.

        This function merges detection scores and TP/FP labels from all
        classes and sorts them in descending order of confidence to
        facilitate D-ECE computation.

        Returns:
            None

        Note:
            This function is only used for D-ECE evaluation, where
            detections from all classes are treated jointly rather
            than on a per-class basis.
        """

        self.calibration_info_all['scores'] = np.array([])
        self.calibration_info_all['tps'] = np.array([])
        self.calibration_info_all['fps'] = np.array([])
        
        # Join all lists into one
        for cl, cl_input in self.calibration_info.items():
            self.calibration_info_all['scores'] = np.append(
                self.calibration_info_all['scores'], cl_input['scores'])
            self.calibration_info_all['tps'] = np.append(
                self.calibration_info_all['tps'], cl_input['tps'])
            self.calibration_info_all['fps'] = np.append(
                self.calibration_info_all['fps'], cl_input['fps'])

        sorted_idx = (-self.calibration_info_all['scores']).argsort()
        self.calibration_info_all['scores'] = self.calibration_info_all['scores'][sorted_idx]
        self.calibration_info_all['tps'] = self.calibration_info_all['tps'][sorted_idx]
        self.calibration_info_all['fps'] = self.calibration_info_all['fps'][sorted_idx]

        return
    

    def compute_single_errors(self, comparison = False):
        """
        Computes calibration errors.

        Depending on the evaluation settings, this function computes
        D-ECE, ACE, or LaECE for confidence calibration. It additionally
        computes coordinate-wise calibration errors (C-ECE) and
        direction-aware calibration errors (DaCE) for bounding box
        calibration.

        Arguments:
            comparison (bool, optional) : reserved for compatibility.

        Returns:
            None

        Note:
            - If `is_dece` is True, detections from all classes are merged
            and D-ECE is computed in a class-agnostic manner.
            - Otherwise, calibration errors are computed independently for
            each class.
            - If `is_coordinate_wise` is enabled, coordinate-wise confidence
            scores and predicted coordinate directions are used to compute
            C-ECE and DaCE. Otherwise, the detection confidence is shared
            across all coordinates.
        """
        # Class-agnostic evaluation
        if self.is_dece:
            self.combine_calibration_info()
            # Find total number of valid detections for this class
            total_det = self.calibration_info_all['tps'].sum(
            ) + self.calibration_info_all['fps'].sum()

            # If no detection, then ignore
            if total_det == 0:
                return

            for i in range(self.bin_count):
                # Find detections in this bin
                if i == 0:
                    bin_all_det = np.logical_and(self.bins[i] <= self.calibration_info_all['scores'],
                                                 self.calibration_info_all['scores'] <= self.bins[i + 1])
                else:
                    bin_all_det = np.logical_and(self.bins[i] < self.calibration_info_all['scores'],
                                                 self.calibration_info_all['scores'] <= self.bins[i + 1])

                bin_tps = np.logical_and(
                    self.calibration_info_all['tps'], bin_all_det)
                bin_fps = np.logical_and(
                    self.calibration_info_all['fps'], bin_all_det)
                bin_det = np.logical_or(bin_tps, bin_fps)

                bin_scores = self.calibration_info_all['scores'][bin_det]

                # Count number of tps in this bin
                num_tp = bin_tps.sum()

                # Count number of fps in this bin
                num_fp = bin_fps.sum()

                # Count number of detections in this bin
                num_det = num_tp + num_fp

                if num_det == 0:
                    self.errors[i] = np.nan
                    self.weights_per_bin[i] = 0
                    self.prec_iou[i] = np.nan
                    continue
                else:
                    self.prec_iou[i] = num_tp / num_det

                # Average of Scores in this bin
                mean_score = bin_scores.mean()

                self.errors[i] = np.abs(self.prec_iou[i] - mean_score)

                # Weight of the bin
                self.weights_per_bin[i] = num_det / total_det


        # Class-wise evaluation
        else:
            for cl, cl_input in self.calibration_info.items():
                # For corrupted images, all images with this class is already rejected
                if 'tps' not in cl_input.keys():
                    continue

                # Find total number of valid detections for this class
                total_det = cl_input['tps'].sum() + cl_input['fps'].sum()

                # If no detection, then ignore
                if total_det == 0:
                    continue

                # LaACE_0 evaluation
                if self.is_ace:
                    # Get the TP-FP information for all the detection of the current class
                    cl_tps = cl_input['tps'].sum()
                    cl_fps = cl_input['fps'].sum()
                    cl_dets = np.logical_or(cl_tps, cl_fps)

                    # Get the IoU and score information
                    ious = cl_input['iou'][cl_dets]
                    scores = np.abs(cl_input['scores'][cl_dets])

                    # Accumulate the errors for the current class for eval
                    self.errors[cl] = np.mean(np.abs(ious - scores))

                # LaECE-style evaluatiion
                else:
                    for i in range(self.bin_count):
                        # Find detections in this bin

                        if i == 0:
                            bin_all_det = np.logical_and(self.bins[i] <= np.abs(cl_input['scores']),
                                                         np.abs(cl_input['scores']) <= self.bins[i + 1])
                        else:
                            bin_all_det = np.logical_and(self.bins[i] < np.abs(cl_input['scores']),
                                                         np.abs(cl_input['scores']) <= self.bins[i + 1])

                        bin_tps = np.logical_and(cl_input['tps'], bin_all_det)
                        bin_fps = np.logical_and(cl_input['fps'], bin_all_det)
                        bin_det = np.logical_or(bin_tps, bin_fps)
                        bin_scores = np.abs(cl_input['scores'][bin_det])
                        bin_ious = cl_input['iou'][bin_tps]

                        # Count number of tps in this bin
                        num_tp = bin_tps.sum()

                        # Count number of fps in this bin
                        num_fp = bin_fps.sum()

                        # Count number of detections in this bin
                        num_det = num_tp + num_fp

                        if num_det == 0:
                            self.errors[cl, i] = np.nan
                            self.weights_per_bin[cl, i] = 0
                            self.prec_iou[cl, i] = np.nan
                            continue

                        # Find error
                        if len(bin_ious) > 0:
                            # norm_iou = (bin_ious - 0.10) / (1 - 0.10)
                            norm_iou = bin_ious
                            norm_total_iou = norm_iou.sum()
                        else:
                            norm_total_iou = 0

                        self.prec_iou[cl, i] = norm_total_iou / num_det

                        # Average of Scores in this bin
                        mean_score = bin_scores.mean()

                        self.errors[cl, i] = np.abs(
                            self.prec_iou[cl, i] - mean_score)

                        # Weight of the bin
                        self.weights_per_bin[cl, i] = num_det / total_det

                # Calculate C-ECE
                if not self.is_coordinate_wise:
                    cw_scores = np.tile(np.abs(cl_input['scores']).reshape(-1, 1), (1, 4))

                else:
                    cw_scores = np.abs(cl_input['cw_score'])
                
                for dim in range(4):
                    car_all = np.array(cl_input['car']).reshape(-1, 4)

                    for i in range(self.bin_count):
                        car = np.abs(car_all[:, dim])

                        if i == 0:
                            bin_all_det = np.logical_and(self.bins[i] <= cw_scores[:, dim],
                                                        cw_scores[:, dim] <= self.bins[i + 1])
                        else:
                            bin_all_det = np.logical_and(self.bins[i] < cw_scores[:, dim],
                                                        cw_scores[:, dim] <= self.bins[i + 1])

                        # TP/FP in this bin
                        bin_tps = np.logical_and(cl_input['tps'], bin_all_det)
                        bin_fps = np.logical_and(cl_input['fps'], bin_all_det)
                        bin_det = np.logical_or(bin_tps, bin_fps)

                        # Scores and IoUs for this bin
                        bin_scores = cw_scores[:, dim][bin_det]
                        bin_cars   = car[bin_tps]

                        num_tp = bin_tps.sum()
                        num_fp = bin_fps.sum()
                        num_det = num_tp + num_fp

                        if num_det == 0:
                            self.cw_errors[cl, i, dim] = np.nan
                            self.prec_car[cl, i, dim] = np.nan
                            self.weights_per_bin_cw_errors[cl, i, dim] = 0
                            continue
                        
                        if len(bin_cars) > 0:
                            norm_total_car = bin_cars.sum()
                        else:
                            norm_total_car = 0

                        self.prec_car[cl, i, dim] = norm_total_car / num_det

                        # Mean score in this bin
                        mean_score = bin_scores.mean()

                        # Error for this dim
                        self.cw_errors[cl, i, dim] = np.abs(self.prec_car[cl, i, dim] - mean_score)

                        # Weight for this dim
                        self.weights_per_bin_cw_errors[cl, i, dim] = num_det / total_det

                        
                # Calculates DaCE
                if self.is_coordinate_wise:
                    car_all   = np.array(cl_input['car']).reshape(-1, 4)
                    pred_sign_all = np.array(cl_input['pred_sign']).reshape(-1, 4)
                    cw_score_all = np.array(cl_input['cw_score']).reshape(-1,4)
                else:
                    car_all   = np.array(cl_input['car']).reshape(-1, 4)
                    pred_sign_all = np.tile(np.sign(cl_input['scores']).reshape(-1, 1), (1, 4))
                    cw_score_all = np.tile(np.abs(cl_input['scores']).reshape(-1, 1), (1, 4))

                tps = np.array(cl_input['tps'])
                fps = np.array(cl_input['fps'])

                valid = np.logical_or(tps, fps)

                if valid.sum() == 0:
                    continue

                pred_sign_all   = pred_sign_all[valid]
                car_all         = car_all[valid]
                cw_score_all   = cw_score_all[valid]
                tps             = tps[valid]
                fps             = fps[valid]

                
                gt_all = np.sign(car_all) * (1.0 - np.abs(car_all))
                pred_all = pred_sign_all * (1.0 - np.abs(cw_score_all))

                gt_raw_all = np.abs(car_all)

                total_det = len(pred_all)

                bins = np.linspace(-1, 1, self.bin_count + 1)

                for dim in range(4):

                    pred_dim = pred_all[:, dim]
                    gt_dim   = gt_all[:, dim]

                    gt_raw_dim = gt_raw_all[:, dim]

                    for i in range(self.bin_count):

                        if i == 0:
                            bin_mask = np.logical_and(bins[i] <= pred_dim,
                                                        pred_dim <= bins[i+1])
                        else:
                            bin_mask = np.logical_and(bins[i] < pred_dim,
                                                        pred_dim <= bins[i+1])

                        num_det = bin_mask.sum()

                        if num_det == 0:
                            self.cw_errors_sign[cl, i, dim] = np.nan
                            self.prec_car_sign[cl, i, dim] = np.nan
                            self.weights_per_bin_cw_errors_sign[cl, i, dim] = 0
                            continue

                        bin_tp_mask = np.logical_and(bin_mask, tps)
                        num_tp = bin_tp_mask.sum()

                        if num_tp == 0:
                            signed_error = np.nan
                        else:
                            pred_bin_error = pred_dim[bin_tp_mask]
                            gt_bin_error   = gt_dim[bin_tp_mask]

                            signed_error = np.sum(np.abs(pred_bin_error - gt_bin_error)) / num_tp

                        gt_raw_bin = gt_raw_dim[bin_mask]
                        gt_mean = np.sum(gt_raw_bin) / num_det

                        self.cw_errors_sign[cl, i, dim] = signed_error

                        self.prec_car_sign[cl, i, dim] = 1.0 - gt_mean

                        self.weights_per_bin_cw_errors_sign[cl, i, dim] = num_det / total_det

        return                    

    
    def calibrate(self, detections, box_calibration_type, \
                  CR, DDE, box_calibrator, thr, sign_norm_stats, coco_classes):
        """
        Applies the trained calibration models to detector outputs.

        This function calibrates the confidence score of each detection
        using the learned Confidence Re-encoder (CR), Directional
        Displacement Estimator (DDE), and the selected box-level confidence
        calibrator.

        Arguments:
            detections (list)             : detector outputs to be calibrated
            box_calibration_type (str)    : bounding box calibration method
                                            ('isotonic_regression', or
                                            'platt_scaling')
            CR (list)                     : trained Confidence Re-encoder models
            DDE (list)                    : trained Directional Displacement
                                            Estimator models
            box_calibrator (list)         : trained confidence calibration models
            thr (list)                    : decision thresholds for the DDE
            sign_norm_stats (dict)        : normalization statistics for the
                                            DDE input features
            coco_classes (list)           : dataset category IDs

        Returns:
            list : calibrated detections with updated confidence scores.
                When coordinate-wise calibration is enabled, each
                detection additionally contains:
                    - 'cw_score'  : coordinate-wise confidence scores
                    - 'pred_sign' : predicted coordinate directions

        Note:
            If all calibration models are None and the thresholds
            correspond to the identity configuration, the input
            detections are returned without modification.
        """

        # For comparison experiments (Identity)
        if CR is None and DDE is None and box_calibrator is None and thr == [0.5,0.5,0.5,0.5]:
            print("identity")
            return detections, None, None

        for detection in detections:
            cl = coco_classes.index(detection['category_id'])

            if box_calibration_type == 'isotonic_regression':
                if CR[cl] is None and DDE[cl] is None:
                    continue
                    
                feature = np.array(detection['bbox_feature'])[None, :]
                all_logits = np.array(detection['all_logits'])[None, :]
                valid_scores = np.array(detection['score']).reshape(1,1)
                valid_bbox = np.array(detection['bbox'], dtype=np.float32)

                x, y, w, h = valid_bbox

                cx = x + 0.5*w
                cy = y + 0.5*h
                area = w*h
                aspect = w/(h+1e-6)

                bbox_features = np.array([
                    cx, cy, w, h,
                    area,
                    aspect
                ], dtype=np.float32).reshape(1,-1)

                logit = inverse_sigmoid(valid_scores)

                CR_input = np.concatenate([logit, feature], axis=1)

                DDE_input = np.concatenate(
                    [all_logits, bbox_features], axis=1)

                mean = sign_norm_stats[cl]['mean']
                std = sign_norm_stats[cl]['std']
                DDE_input = (DDE_input - mean) / std

                loc_dev = next(CR[cl].parameters()).device

                CR_inputs = torch.from_numpy(CR_input).to(device=loc_dev, dtype=torch.float32)
                DDE_inputs = torch.from_numpy(DDE_input).to(device=loc_dev, dtype=torch.float32)

                with torch.no_grad():
                    CR_outputs = CR[cl].predict(CR_inputs)
                    DDE_outputs = DDE[cl].predict(DDE_inputs, thr[cl])

                if isinstance(CR_outputs, torch.Tensor):
                    preds = CR_outputs.detach().cpu().numpy()
                    signs = DDE_outputs.detach().cpu().numpy()
                else:
                    preds = np.asarray(CR_outputs)
                    signs = np.asarray(DDE_outputs)

                preds = preds.squeeze()
                signs = signs.squeeze()
                value = preds * signs

                transformed_conf = box_calibrator[cl].predict([[cal_iou(np.array(value))]])[0]
                detection['score'] = np.clip(transformed_conf, 0, 1)
                detection['cw_score'] = np.array(preds)
                detection['pred_sign'] = np.array(signs)
                                        
            elif box_calibration_type == 'platt_scaling':
                
                if type(CR) is not np.ndarray:
                    
                    feature = np.array(detection['bbox_feature'])[None, :]
                    all_logits = np.array(detection['all_logits'])[None, :]
                    valid_scores = np.array(detection['score']).reshape(1,1)
                    valid_bbox = np.array(detection['bbox'], dtype=np.float32)

                    x, y, w, h = valid_bbox

                    cx = x + 0.5*w
                    cy = y + 0.5*h
                    area = w*h
                    aspect = w/(h+1e-6)

                    bbox_features = np.array([
                        cx, cy, w, h,
                        area,
                        aspect
                    ], dtype=np.float32).reshape(1,-1)

                    logit = inverse_sigmoid(valid_scores)

                    CR_input = np.concatenate([logit, feature], axis=1)

                    DDE_input = np.concatenate(
                    [all_logits, bbox_features], axis=1)

                    mean = sign_norm_stats[cl]['mean']
                    std = sign_norm_stats[cl]['std']
                    DDE_input = (DDE_input - mean) / std

                    
                    loc_dev = next(CR[cl].parameters()).device

                    CR_inputs = torch.from_numpy(CR_input).to(device=loc_dev, dtype=torch.float32)
                    DDE_inputs = torch.from_numpy(DDE_input).to(device=loc_dev, dtype=torch.float32)

                    with torch.no_grad():
                        CR_outputs = CR[cl].predict(CR_inputs)
                        DDE_outputs = DDE[cl].predict(DDE_inputs, thr[cl])

                    if isinstance(CR_outputs, torch.Tensor):
                        preds = CR_outputs.detach().cpu().numpy()
                        signs = DDE_outputs.detach().cpu().numpy()
                    else:
                        preds = np.asarray(CR_outputs)
                        signs = np.asarray(DDE_outputs)

                    preds = preds.squeeze()
                    signs = signs.squeeze()
                    value = preds * signs

                    transformed_conf = box_calibrator[cl].predict(cal_iou(np.array(value)))
                    detection['score'] = np.clip(transformed_conf, 0, 1)
                    detection['cw_score'] = np.array(preds)
                    detection['pred_sign'] = np.array(signs)
                        
        return detections


    def accumulate_errors(self):
        """
        Aggregates the computed calibration errors into the final
        evaluation metrics.

        Depending on the evaluation, this function computes
        the overall confidence calibration error (D-ECE, ACE, or LaECE)
        and aggregates the coordinate-wise calibration metrics
        (C-ECE and DaCE) across bins and classes.

        Returns:
            ECE (float)        : overall confidence calibration error
                                (D-ECE, ACE, or LaECE, depending on the
                                evaluation setting)
            C_ECE (np.ndarray) : coordinate-wise Expected Calibration
                                Error for (x1, y1, x2, y2)
            DaCE (np.ndarray)  : Direction-aware Calibration Error for
                                (x1, y1, x2, y2)

        Note:
            - If `is_dece` is enabled, D-ECE is computed using all
            detections jointly.
            - If `is_ace` is enabled, ACE is computed by averaging
            class-wise calibration errors.
            - Otherwise, LaECE is computed by aggregating bin-wise
            calibration errors for each class.
            - C-ECE and DaCE are always aggregated independently for
            each bounding box coordinate.
        """

        # Reports D-ECE
        if self.is_dece:
            ECE = np.nansum(self.weights_per_bin * self.errors)
        else:
            # Reports LaACE
            if self.is_ace:
                class_errors = self.errors
                class_errors[class_errors == 0] = np.nan
                ECE = np.nanmean(self.errors)
            # Reports LaECE
            else:
                bin_sum = np.nansum(self.weights_per_bin * self.errors, axis=1)
                bin_sum[bin_sum == 0] = np.nan
                ECE = np.nanmean(bin_sum)
    
        # Reports C-ECE
        cw_bin_sum = np.nansum(self.weights_per_bin_cw_errors * self.cw_errors, axis=1)
        cw_bin_sum[cw_bin_sum == 0] = np.nan
        C_ECE = np.nanmean(cw_bin_sum, axis=0)

        cw_bin_sum_sign = np.nansum(self.weights_per_bin_cw_errors_sign * self.cw_errors_sign, axis=1)

        cw_bin_sum_sign[cw_bin_sum_sign == 0] = np.nan

        DaCE = np.nanmean(cw_bin_sum_sign, axis=0)

        return ECE, C_ECE, DaCE


    def plot_reliability_diagram(self, ECE, cl=-1, fontsize=22):
        """
        Plots the reliability diagram for LaECE.

        The reliability diagram compares the average confidence score with
        the average localization quality (IoU) in each confidence bin,
        providing a visual assessment of calibration quality.

        Arguments:
            ECE (float)               : computed LaECE value
            cl (int)                  : class index. If set to -1, the
                                        reliability diagram is generated
                                        using the mean statistics across
                                        all classes.
            fontsize (int)            : font size used for labels,
                                        legends, and annotations

        Returns:
            None

        Note:
            The blue bars represent the average IoU of detections within
            each confidence bin, while the pink bars indicate the fraction
            of detections assigned to each bin. The dashed diagonal line
            corresponds to perfect calibration.
        """

        delta = 1.0 / self.bin_count
        x = np.arange(0, 1, delta)

        if cl == -1:
            bin_acc = np.nanmean(self.prec_iou, axis=0)
            bin_weights = np.nanmean(self.weights_per_bin, axis=0)
        else:
            bin_acc = self.prec_iou[cl]
            bin_weights = self.weights_per_bin[cl]
        nan_idx = (bin_weights == 0)
        bin_acc[nan_idx] = 0

        # size and axis limit
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plot grid
        plt.grid(color='tab:grey', linestyle=(
            0, (1, 5)), linewidth=1, zorder=0)
        # plot bars and identity line
        plt.bar(x, bin_acc, color='b', width=delta, align='edge', edgecolor='k',
                label=r'IoU',
                zorder=5)
        plt.bar(x, bin_weights, color='mistyrose', alpha=0.5, width=delta, align='edge',
                edgecolor='r', label='% of Samples', zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
        # labels and legend
        plt.xlabel('Confidence', fontsize=fontsize+7)
        plt.legend(loc='upper left', framealpha=1.0, fontsize=fontsize-8)
        plt.text(0.05, 0.73, '$\mathrm{LaECE}_0$= %.1f%%' % (
            ECE * 100), fontsize=fontsize-8)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        plt.show()
        return


    def plot_reliability_diagram_CECE(self, cw_ECE, cl=-1, fontsize=22):
        """
        Plots coordinate-wise reliability diagrams for C-ECE.

        A separate reliability diagram is generated for each bounding
        box coordinate (x1, y1, x2, y2), comparing the predicted
        coordinate-wise confidence with the corresponding Coordinate
        Alignment Ratio (CAR).

        Arguments:
            cw_ECE (np.ndarray) : coordinate-wise Expected Calibration
                                Error for (x1, y1, x2, y2)
            cl (int)            : class index. If -1, the average over
                                all classes is plotted.
            fontsize (int)      : font size for plot labels and text

        Returns:
            None
        """
        delta = 1.0 / self.bin_count
        x = np.arange(0, 1, delta)

        n_dims = 4
        fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5))

        if cl == -1:
            bin_acc = np.nanmean(self.prec_car, axis=0)
            bin_weights = np.nanmean(self.weights_per_bin_cw_errors, axis=0)
        else:
            bin_acc = self.prec_car[cl]
            bin_weights = self.weights_per_bin_cw_errors[cl]

        for dim in range(n_dims):
            ax = axes[dim] if n_dims > 1 else axes
            acc = bin_acc[:, dim].copy()
            weights = bin_weights[:, dim].copy()

            nan_idx = (weights == 0)
            acc[nan_idx] = 0

            ax.bar(x, acc, color='b', width=delta, align='edge', edgecolor='k', label='CAR', zorder=5)
            ax.bar(x, weights, color='mistyrose', alpha=0.5, width=delta, align='edge',
                edgecolor='r', label='% of Samples', zorder=10)

            ax.plot([0, 1], [0, 1], linestyle='--', color='tab:grey', zorder=15)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Confidence', fontsize=fontsize+2)
            ax.set_title(f'{self.cw_error_name[dim]}', fontsize=fontsize+2)
            ax.legend(loc='upper left', framealpha=1.0, fontsize=fontsize-8)
            ax.text(0.05, 0.73, f'{self.cw_error_name[dim]}= {cw_ECE[dim]*100:.1f}%', fontsize=fontsize-8)
            ax.tick_params(axis='both', labelsize=fontsize)

        plt.tight_layout()
        plt.show()


    def plot_reliability_diagram_DaCE(self, cw_ECE, cl=-1, fontsize=22):
        """
        Plots coordinate-wise reliability diagrams for DaCE.

        A separate reliability diagram is generated for each bounding
        box coordinate (x1, y1, x2, y2), comparing the predicted
        directional uncertainty with the corresponding directional
        misalignment.

        Arguments:
            cw_ECE (np.ndarray) : Direction-aware Calibration Error
                                (DaCE) for (x1, y1, x2, y2)
            cl (int)            : class index. If -1, the average over
                                all classes is plotted.
            fontsize (int)      : font size for plot labels and text

        Returns:
            None
        """
        delta = 2.0 / self.bin_count
        x = np.arange(-1, 1, delta)

        n_dims = 4
        fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5))

        if cl == -1:
            bin_acc = np.nanmean(self.prec_car_sign, axis=0)
            bin_weights = np.nanmean(self.weights_per_bin_cw_errors_sign, axis=0)
        else:
            bin_acc = self.prec_car_sign[cl]
            bin_weights = self.weights_per_bin_cw_errors_sign[cl]

        line_x = np.linspace(-1, 1, 200)
        line_y = np.abs(line_x)

        for dim, ax in enumerate(np.atleast_1d(axes)):
            acc = bin_acc[:, dim].copy()
            weights = bin_weights[:, dim]

            acc[weights == 0] = 0

            ax.bar(x, acc, width=delta, align='edge',
                color='royalblue', edgecolor='k',
                label='Directional misalignment', zorder=5)

            ax.bar(x, weights, width=delta, align='edge',
                color='mistyrose', alpha=0.5,
                edgecolor='r', hatch='/',
                label='% of Samples', zorder=10)

            ax.plot(line_x, line_y, '--', color='tab:grey', linewidth=2, zorder=15)

            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])

            ax.set_xlabel('Directional Uncertainty', fontsize=fontsize + 2)
            ax.set_title(self.Dauce_name[dim], fontsize=fontsize + 2)

            ax.legend(loc='upper left', framealpha=1.0, fontsize=fontsize - 8)

            ax.text(0.05, 0.73,
                    f'{self.Dauce_name[dim]} = {cw_ECE[dim] * 100:.1f}%',
                    transform=ax.transAxes,
                    fontsize=fontsize - 8,
                    zorder=30)

            ax.tick_params(axis='both', labelsize=fontsize)

        plt.tight_layout()
        plt.show()
    

    def compute_LRP(self):
        """
        Computes the Localization Recall Precision (LRP) metrics for each
        object class.

        This function calculates the overall LRP score together with its
        three components: localization error (LRP Loc), false positive
        error (LRP FP), and false negative error (LRP FN), based on the
        true positives, false positives, false negatives, and IoU values
        accumulated during evaluation.

        Arguments:
            None

        Returns:
            None

        Note:
            The computed metrics are stored in `self.lrps`, which contains
            the overall LRP score (`lrp`) as well as its localization
            (`lrp_loc`), false positive (`lrp_fp`), and false negative
            (`lrp_fn`) components for each class. Classes without valid
            detections are assigned `NaN` or the maximum error value,
            following the standard LRP definition.
        """

        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                self.lrps['lrp_loc'][cl] = np.nan
                self.lrps['lrp_fp'][cl] = np.nan
                self.lrps['lrp_fn'][cl] = np.nan
                self.lrps['lrp'][cl] = np.nan
                continue

            # Find total number of valid detections for this class
            tp_num = cl_input['tps'].sum()
            fp_num = cl_input['fps'].sum()
            fn_num = cl_input['npig'] - tp_num

            # If there is detection
            if tp_num + fp_num > 0:
                # There is some TPs
                if tp_num > 0:
                    total_loc = tp_num - cl_input['iou'].sum()
                    self.lrps['lrp'][cl] = (total_loc / (1 - self.tau) + fp_num +
                                            fn_num) / (tp_num + fp_num + fn_num)
                    self.lrps['lrp_loc'][cl] = total_loc / tp_num
                    self.lrps['lrp_fp'][cl] = fp_num / (tp_num + fp_num)
                    self.lrps['lrp_fn'][cl] = fn_num / cl_input['npig']
                else:
                    self.lrps['lrp_loc'][cl] = np.nan
                    self.lrps['lrp_fp'][cl] = np.nan
                    self.lrps['lrp_fn'][cl] = 1.
                    self.lrps['lrp'][cl] = 1.
            else:
                self.lrps['lrp_loc'][cl] = np.nan
                self.lrps['lrp_fp'][cl] = np.nan
                self.lrps['lrp_fn'][cl] = 1.
                self.lrps['lrp'][cl] = 1.
        return

    
    def fit(self, val_detections, box_calibration_type, thresholds=[-1., -1.], eval_type='bbox',
            lr = 1e-1, epochs=100,sign_lr = 3e-2, sign_epochs=50, seed = 1234):
        """
        Trains the calibration models using validation detections.

        This function determines the pre-calibration detection thresholds,
        prepares calibration data from the validation set, trains the
        Confidence Re-encoder (CR), Directional Displacement Estimator (DDE),
        and box-level confidence calibrator, and finally computes the operating
        thresholds after calibration.

        Arguments:
            val_detections (str)         : file path to validation detections
            box_calibration_type (str)   : box-level calibration method
                                        ('identity',
                                            'isotonic_regression', or
                                            'platt_scaling')
            thresholds (list)            : detection thresholds used before
                                        and after calibration
            eval_type (str)              : evaluation type
            lr (float)                   : learning rate for the
                                        Confidence Re-encoder (CR)
            epochs (int)                 : number of training epochs for CR
            sign_lr (float)              : learning rate for the Directional
                                        Displacement Estimator (DDE)
            sign_epochs (int)            : number of training epochs for DDE
            seed (int)                   : random seed for reproducibility

        Returns:
            CR (list)                    : trained Confidence Re-encoder models
            DDE (list)                   : trained Directional Displacement
                                        Estimator models
            box_calibrator (list)        : trained box-level confidence calibration models
            thr (list)                   : learned decision thresholds for DDE
            thresholds (list)            : pre-calibration and operating
                                        detection thresholds
            sign_norm_stats (dict)       : normalization statistics for the
                                        DDE input features

        Note:
            If `box_calibration_type` is set to `'identity'`, no calibration
            models are trained and the identity configuration is returned.
        """

        # set the calibrator-specific parameters
        self.classification_type = box_calibration_type

        val_detections = load_detections_from_file(val_detections, self.dataset_classes)
        dataset_classes = self.dataset_classes

        # the following corresponds to the first set of thresholds learned pre-calibration stage
        pre_calibration_thresholds = get_detection_thresholds(
            self.val_annotations, val_detections, 'coco', thresholds[0], self.tau, eval_type, max_dets=self.params.maxDets)

        thresholded_val_detections = threshold_detections(
            val_detections, pre_calibration_thresholds, dataset_classes)
        
        self.cocoDt = COCO(self.val_annotations).loadRes(
            thresholded_val_detections)
        self.iouType = eval_type

        self.evaluate()
        self.prepare_input()

        calibration_info = self.calibration_info
            
        if box_calibration_type =='identity':
            CR, DDE, box_calibrator, thr, sign_norm_stats = None, None, None, [0.5, 0.5, 0.5, 0.5], None
        else:
            print()
            print('\n----------------Confidence Re-encoding...-----------------')
            CR, DDE, box_calibrator, thr, sign_norm_stats = Trainer(calibration_info, box_calibration_type, lr= lr, epochs = epochs,
                                                                    sign_lr= sign_lr, sign_epochs = sign_epochs, is_dece=self.is_dece, seed = seed)
            print('------------------------------------------------------------')
            print()
        
        calibrated_val_detections = self.calibrate(
            val_detections, box_calibration_type, CR, DDE, box_calibrator, thr, sign_norm_stats, dataset_classes)

        operating_thresholds = get_detection_thresholds(
            self.val_annotations, calibrated_val_detections, 'coco', thresholds[1], self.tau, eval_type, max_dets=self.params.maxDets)
        
        return CR, DDE, box_calibrator, thr, [pre_calibration_thresholds, operating_thresholds], sign_norm_stats


    def transform(self, test_detections, CR, DDE, box_calibrator, thr, thresholds, sign_norm_stats):
        """
        Applies the trained calibration models to a set of test detections.

        This function loads the test detections, applies the pre-calibration
        detection threshold, calibrates the remaining detections using the
        trained Confidence Re-encoder (CR), Directional Displacement
        Estimator (DDE), and confidence calibrator, and finally applies
        the operating threshold to the calibrated detections.

        Arguments:
            test_detections (str)     : file path to test detections
            CR (list)                 : trained Confidence Re-encoder models
            DDE (list)                : trained Directional Displacement
                                        Estimator models
            box_calibrator (list)     : trained confidence calibration models
            thr (list)                : learned decision thresholds for the DDE
            thresholds (list)         : pre-calibration and operating
                                        detection thresholds
            sign_norm_stats (dict)    : normalization statistics for the
                                        DDE input features

        Returns:
            list : calibrated test detections after applying both the
                calibration models and the operating threshold.

        Note:
            The pre-calibration threshold is applied before calibration,
            while the operating threshold is applied after calibration to
            ensure a fair comparison with the original detector.
        """

        test_detections = load_detections_from_file(test_detections, self.dataset_classes)

        # May or may not threshold test detections before the calibration step
        thresholded_test_detections = threshold_detections(
            test_detections, thresholds[0], self.dataset_classes)

        # Calibrate test data detections, whether all or only survived
        calibrated_test_detections = self.calibrate(
            thresholded_test_detections, self.classification_type, CR, DDE, box_calibrator, thr, sign_norm_stats, self.dataset_classes)
        
        # Re-threshold the calibrated detections for a fair comparison
        calibrated_test_detections = threshold_detections(
            calibrated_test_detections, thresholds[1], self.dataset_classes)

        # return calibrated_test_detections, sign_feature, sign_check
        return calibrated_test_detections


    def evaluate_calibration(self, calibrated_test_detections, is_dece=False, show_cw_plot = True, show_laece_plot=False, verbose=False, comparison=False):
        """
        Evaluates the calibrated detections in terms of both detection
        accuracy and calibration quality.

        This function computes the proposed coordinate-wise calibration
        metrics (C-ECE and DaCE), box-level confidence calibration
        metrics (LaECE, LaACE, and optionally D-ECE), and localization
        accuracy metrics (LRP). Reliability diagrams can also be
        displayed for visual inspection of the calibration quality.

        Arguments:
            calibrated_test_detections (list) : calibrated test detections
            is_dece (bool)                    : whether to additionally
                                                compute and report D-ECE
            show_cw_plot (bool)               : whether to display the
                                                reliability diagrams for
                                                C-ECE and DaCE
            show_laece_plot (bool)            : whether to display the
                                                reliability diagram for
                                                LaECE
            verbose (bool)                    : whether to perform and
                                                display the full COCO
                                                evaluation results
            comparison (bool)                 : whether to evaluate the
                                                baseline without
                                                coordinate-wise calibration

        Returns:
            dict : calibration information generated during evaluation,
                including detection scores, TP/FP labels, IoUs,
                Coordinate Alignment Ratios (CARs), and related
                calibration statistics.

        Note:
            Two evaluation passes are performed:
                1. Compute LaECE, C-ECE, and DaCE.
                2. Compute LaACE together with the LRP metrics.

            If `is_dece` is True, an additional D-ECE evaluation is
            performed using class-agnostic detections.
        """
        is_coordinate_wise = not comparison
        # Compute the joint evaluation measures of performance and calibration
        calibration_eval = ReDCCOCO(
            self.test_annotations, self.test_annotations, self.eval_type, self.bin_count, self.tau, False, False, is_coordinate_wise, self.params.maxDets[0])

        calibration_eval.cocoDt = COCO(
            self.test_annotations).loadRes(calibrated_test_detections)
        calibration_eval.evaluate(eval_cal = True)
        info = calibration_eval.prepare_input(eval_cal = True)
        _=calibration_eval.compute_single_errors(comparison)
        LaECE_0, C_ECE, Da_CE = calibration_eval.accumulate_errors()
        
        print('--------------------Coordinate-wise PLOT-------------------')

        if show_cw_plot:
            calibration_eval.plot_reliability_diagram_CECE(C_ECE, cl=-1, fontsize=22)
            calibration_eval.plot_reliability_diagram_DaCE(Da_CE)

        # Plot the reliability diagram when asked
        if show_laece_plot:
            print('-------------------Approximated IoU PLOT-------------------')
            calibration_eval.plot_reliability_diagram(LaECE_0, cl=-1, fontsize=22)

        # The final True is for the LaACE option, as it does not require an explicit set of binning
        calibration_eval = ReDCCOCO(self.test_annotations, self.test_annotations,
                                           self.eval_type, self.bin_count, self.tau, False, True, is_coordinate_wise, self.params.maxDets[0])

        calibration_eval.cocoDt = COCO(
            self.test_annotations).loadRes(calibrated_test_detections)
        calibration_eval.evaluate(eval_cal = True)
        _=calibration_eval.prepare_input(eval_cal = True)
        _ = calibration_eval.compute_single_errors(comparison)
        calibration_eval.compute_LRP()
        LaACE_0, _, _  = calibration_eval.accumulate_errors()
        
        

        print()
        print('--------------------------ACCURACY-------------------------')
        print()
        print(
            f'LRP       @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {np.nanmean(calibration_eval.lrps["lrp"]) * 100:.1f}')
        print(
            f'LRP Loc   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {np.nanmean(calibration_eval.lrps["lrp_loc"]) * 100:.1f}')
        print(
            f'LRP FP    @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {np.nanmean(calibration_eval.lrps["lrp_fp"]) * 100:.1f}')
        print(
            f'LRP FN    @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {np.nanmean(calibration_eval.lrps["lrp_fn"]) * 100:.1f}')

        print('\n-------------------------CALIBRATION-----------------------')
        print()

        print(
            f'CX₁-ECE   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {C_ECE[0] * 100:.1f}')
        print(
            f'CY₁-ECE   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {C_ECE[1] * 100:.1f}')
        print(
            f'CX₂-ECE   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {C_ECE[2] * 100:.1f}')
        print(
            f'CY₂-ECE   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {C_ECE[3] * 100:.1f}')

        print('\n-------------------ADDITIONAL CALIBRATION------------------')
        print()

        if self.tau == 0.0:
            print(
                f'LaECE_0   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {LaECE_0 * 100:.1f}')
            print(
                f'LaACE_0   @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {LaACE_0 * 100:.1f}')
        else:
            print(
                f'LaECE     @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {LaECE_0 * 100:.1f}')
            print(
                f'LaACE     @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {LaACE_0 * 100:.1f}')

        if is_dece:
            calibration_eval = ReDCCOCO(
                self.test_annotations, self.test_annotations, 'bbox', 10, 0.5, True, False,False, self.params.maxDets[0])
            calibration_eval.cocoDt = COCO(
                self.test_annotations).loadRes(calibrated_test_detections)
            calibration_eval.evaluate(eval_cal = True)
            _=calibration_eval.prepare_input(eval_cal = True)
            _=calibration_eval.compute_single_errors(compaison)
            calibration_eval.compute_LRP()

            DECE, _, _ = calibration_eval.accumulate_errors()
            print(
                f'D-ECE     @[ IoU={self.tau:.1f} | area=   all | maxDets={self.params.maxDets} ] = {DECE * 100:.1f}\n')
        else:
            print('\n')

        if verbose:
            COCO_evaluation(self.test_annotations,
                            calibrated_test_detections, self.eval_type)

        return info