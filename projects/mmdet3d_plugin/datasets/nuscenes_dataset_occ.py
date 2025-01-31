import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import torch
from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesSweepDataset
from .occ_metrics import Metric_mIoU, Metric_FScore


@DATASETS.register_module()
class NuScenesSweepDatasetOcc(NuScenesSweepDataset):
    r"""NuScenes dataset for occupancy prediction tasks.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def evaluate(self, 
                 occ_results, 
                 runner=None, 
                 show_dir=None, 
                 use_image_mask=True, 
                 **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=use_image_mask)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(
                occ_pred, gt_semantics, mask_lidar, mask_camera)

        self.class_names, mIoU, self.cnt, final_mIoU, final_IoU = self.occ_eval_metrics.count_miou()

        return dict(mIoU=final_mIoU, IoU=final_IoU)