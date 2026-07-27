from ReDC_coco import ReDCCOCO

class ReDC:
    def __init__(self, val_annotations, test_annotations, eval_type='bbox', bin_count=25, tau=0.0, is_dece=False, is_ace=False, is_coordinate_wise=True, max_dets=100):
        """
        Initializes the ReDC calibration framework.

        Arguments:
            val_annotations (str)             : file path to validation annotations
            test_annotations (str)            : file path to test annotations
            eval_type (str)                   : evaluation type ('bbox' or 'segm')
            bin_count (int)                   : number of calibration bins
            tau (float)                       : IoU threshold for TP/FP matching
            is_dece (bool)                    : whether to use D-ECE evaluation
            is_ace (bool)                     : whether to use Adaptive CE
            is_coordinate_wise (bool)         : whether to perform coordinate-wise bounding box calibration
            max_dets (int)                    : maximum detections per image
        """    
        self.val_annotations = val_annotations
        self.test_annotations = test_annotations

        self.calibration_scheme = ReDCCOCO(
            val_annotations, test_annotations, eval_type, bin_count, tau, is_dece, is_ace, is_coordinate_wise, max_dets)

    def fit(self, val_detections, box_calibration_type, thresholds=[-1., -1.], eval_type='bbox', lr=1e-1, epochs=100, sign_lr=3e-2, sign_epochs=50, seed = 1234):
        """
        Optimizes the confidence re-encoders (CRs) and the directional displacement estimators (DDEs) using the validation detections.

        Arguments:
            val_detections (str)      : file path to validation detections
            box_calibration_type (str): bounding box calibration method
            thresholds (list)         : confidence thresholds for calibration
            eval_type (str)           : evaluation type ('bbox' or 'segm')
            lr (float)                : learning rate for CRs
            epochs (int)              : number of training epochs for CRs
            sign_lr (float)           : learning rate for DDEs
            sign_epochs (int)         : number of training epochs for DDEs
            seed (int)                : random seed

        Returns:
            tuple : calibration models and parameters required for
                    bounding box calibration.
        """
        return self.calibration_scheme.fit(val_detections, box_calibration_type, thresholds, eval_type, lr, epochs, sign_lr, sign_epochs, seed)
    
    def transform(self, test_detections, CR, DDE, box_calibrator, thr, thresholds, sign_norm_stats):
        """
        Applies the learned calibration models to test detections.

        Arguments:
            test_detections (str)        : file path to test detections
            CR                           : confidence re-encoder models
            DDE                          : directional displacement estimators
            box_calibrator               : trained box calibration model
            thr                          : threshold for direction predictions
            thresholds (float array)     : pair of thresholds for two stages
            sign_norm_stats              : normalization statistics for coordinate sign calibration

        Returns:
            list : calibrated detection results.
        """
        return self.calibration_scheme.transform(test_detections, CR, DDE, box_calibrator, thr, thresholds, sign_norm_stats)
    
    def evaluate_calibration(self, calibrated_test_detections, is_dece=False, show_cw_plot = True, show_laece_plot=False, verbose=False, comparison=False):
        """
        Evaluates the calibration performance of the calibrated
        detection results.

        Arguments:
            calibrated_test_detections : calibrated detection results
            is_dece (bool)             : whether to use D-ECE evaluation
            show_cw_plot (bool)        : whether to display class-wise plots
            show_laece_plot (bool)     : whether to display LAECE plots
            verbose (bool)             : whether to print detailed metrics
            comparison (bool)          : whether to evaluate the baseline without coordinate calibration

        Returns:
            dict : calibration evaluation metrics.
        """
        return self.calibration_scheme.evaluate_calibration(calibrated_test_detections, is_dece, show_cw_plot, show_laece_plot, verbose, comparison)