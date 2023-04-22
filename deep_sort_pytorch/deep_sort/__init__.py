from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']
"""
'DEEPSORT': {'REID_CKPT': './deep_sort/deep/checkpoint/ckpt.t7', 'MAX_DIST': 0.2, 'MIN_CONFIDENCE': 0.3, 
        'NMS_MAX_OVERLAP': 0.5, 'MAX_IOU_DISTANCE': 0.7, 'MAX_AGE': 70, 'N_INIT': 3, 'NN_BUDGET': 100, 
"""


def build_tracker(cfg, use_cuda):
    return DeepSort(cfg.DEEPSORT.REID_CKPT, 
                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
    









