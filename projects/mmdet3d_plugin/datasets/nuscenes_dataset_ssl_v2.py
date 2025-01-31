import os
import os.path as osp
import numpy as np
import pickle
import torch
# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
from pyquaternion import Quaternion

import mmdet3d
from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose

from .nuscenes_dataset import NuScenesSweepDataset


def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat


def cam_i_to_cam_j_transform(cam_i_to_cam_i_global,
                             cam_j_infos):
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    # obtain the global to camera j ego transformation
    cam_j_global_to_cam_j_list = []
    for cam_type in camera_types:
        cam_info = cam_j_infos[cam_type]
        
        ego_to_cam = rt2mat(cam_info['sensor2ego_translation'],
                            cam_info['sensor2ego_rotation'],
                            inverse=True)
        
        global_to_ego = rt2mat(cam_info['ego2global_translation'],
                               cam_info['ego2global_rotation'],
                               inverse=True)

        global_to_cam = ego_to_cam @ global_to_ego
        cam_j_global_to_cam_j_list.append(global_to_cam)
    
    cam_j_global_to_cam_j = np.stack(cam_j_global_to_cam_j_list)
    cam_i_to_cam_j = cam_j_global_to_cam_j @ cam_i_to_cam_i_global
    return cam_i_to_cam_j


def load_adjacent_info(input_dict, extra_frames):
    curr_cam_to_ego = np.stack(input_dict['cam2camego'])  # (6, 4, 4)
    curr_camego_to_global = np.stack(input_dict['camego2global'])  # (6, 4, 4)

    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    
    output_dict = {}
    cam_intrinsic = np.stack(input_dict['cam_intrinsic'])
    output_dict['K'] = torch.from_numpy(cam_intrinsic).to(torch.float32)  # (6, 4, 4)
    
    seq_cam_to_cam_list = []
    info_adj_list = []
    for idx in extra_frames:
        if idx == -1:
            flag = 'prev'
        elif idx == 1:
            flag = 'next'
        
        adj_info = input_dict[flag]

        if len(adj_info['CAM_FRONT']) == 0:
            adj_info = input_dict['cams']
        
        info_adj_list.append({'cams': adj_info})

        adj_global_to_ego_list,  adj_ego_to_cam_list = [], []
        for cam_type in camera_types:
            cam_info = adj_info[cam_type]
            
            cam_to_ego = rt2mat(cam_info['sensor2ego_translation'],
                                cam_info['sensor2ego_rotation'])
            ego_to_cam = np.linalg.inv(cam_to_ego)
            
            ego_to_global = rt2mat(cam_info['ego2global_translation'],
                                    cam_info['ego2global_rotation'])
            global_to_ego = np.linalg.inv(ego_to_global)

            adj_global_to_ego_list.append(global_to_ego)
            adj_ego_to_cam_list.append(ego_to_cam)
        
        adj_global_to_ego = np.stack(adj_global_to_ego_list)
        adj_ego_to_cam = np.stack(adj_ego_to_cam_list)

        curr_cam_to_cami = adj_ego_to_cam @ adj_global_to_ego @ curr_camego_to_global @ curr_cam_to_ego
        seq_cam_to_cam_list.append(curr_cam_to_cami)
    
    cam_T_cam = np.stack(seq_cam_to_cam_list)
    output_dict['cam_T_cam'] = torch.from_numpy(cam_T_cam).to(torch.float32)  # (2, 6, 4, 4)
    output_dict['adjacent'] = info_adj_list
    return output_dict


@DATASETS.register_module()
class NuScenesSweepDatasetSSL_V2(NuScenesSweepDataset):
    r"""We load the adjecent sweep frames for the self-supervised learning.
    """
    def __init__(
        self,
        use_depth_consistency=False,
        extra_frames=[-1, 1],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_depth_consistency = use_depth_consistency
        self.extra_frames = extra_frames

    def get_data_info(self, index):
        """Get data info according to the given index.
        """
        input_dict = super().get_data_info(index)

        if self.use_depth_consistency:
            curr_adj_dict = load_adjacent_info(input_dict, self.extra_frames)
            input_dict.update(curr_adj_dict)
        return input_dict


@DATASETS.register_module()
class NuScenesSweepDatasetFuture(NuScenesSweepDataset):
    r"""Load the future frames information for the flow-based self-supervised learning.
    Currently we only load the future key frames.
    Args:
        use_depth_consistency (bool, optional): _description_. Defaults to False.
        extra_frames (list, optional): use for depth consistency. Defaults to [-1, 1].
        future_frames (list, optional): deprecated. Defaults to [1].
        adjacent_frames (list, optional): loading the adjacent frames for flow warping. Defaults to [-1, 1].
        use_flow_photometric_loss (bool, optional): whether use the depth ssl for flow warped volume feature. Defaults to False.
    """
    def __init__(
        self,
        use_depth_consistency=False,
        extra_frames=[-1, 1],
        future_frames=[1],
        adjacent_frames=None,
        use_flow_photometric_loss=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.future_frames = future_frames
        assert len(self.future_frames) == 1, 'Currently we only support one future key frames.'

        self.adjacent_frames = adjacent_frames
        if self.adjacent_frames is not None:
            self.future_frames = self.adjacent_frames

        self.use_depth_consistency = use_depth_consistency
        self.extra_frames = extra_frames
        self.use_flow_photometric_loss = use_flow_photometric_loss

    def load_data_info(self, index):
        info = self.data_infos[index]
        
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            cams=info["cams"],
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            img_filename=info["img_filename"],
            lidar2img=info["lidar2img"],
            lidar2cam=info["lidar2cam"],
            cam_intrinsic=info["cam_intrinsic"],
            cam2camego=info["cam2camego"],
            camego2global=info["camego2global"],
        )
        if self.return_gt_info:
            input_dict["info"] = info

        if "prev" in info:
            input_dict["prev"] = info["prev"]
        
        if "next" in info:
            input_dict["next"] = info["next"]
        
        if "occ_path" in info:
            input_dict["occ_gt_path"] = info["occ_path"]

        if "lidarseg" in info:
            input_dict["lidarseg"] = info["lidarseg"]

        # convert file path to nori and process sweep number in loading function
        if self.modality["use_lidar"]:
            input_dict["sweeps"] = info["sweeps"]
        
        # for conviently use
        image_paths = input_dict['img_filename']
        lidar2img_rts = input_dict['lidar2img']
        lidar2cam_rts = input_dict['lidar2cam']
        cam_intrinsics = input_dict['cam_intrinsic']

        if self.modality["use_camera"]:
            # use cam sweeps
            if "cam_sweep_num" in self.modality:
                cam_sweeps_paths = []
                cam_sweeps_id = []
                cam_sweeps_time = []
                lidar2img_sweeps_rts = []
                # add lidar2img matrix
                lidar2cam_sweeps_rts = []
                cam_sweeps_intrinsics = []
                cam_sweep_num = self.modality["cam_sweep_num"]
                for cam_idx, (cam_type, cam_infos) in enumerate(
                    info["cam_sweeps_info"].items()
                ):
                    # avoid none sweep
                    if len(cam_infos) == 0:
                        cam_sweeps = [
                            image_paths[cam_idx] for _ in range(cam_sweep_num)
                        ]
                        cam_ids = [0 for _ in range(cam_sweep_num)]
                        cam_time = [0.0 for _ in range(cam_sweep_num)]
                        lidar2img_sweeps = [
                            lidar2img_rts[cam_idx] for _ in range(cam_sweep_num)
                        ]
                        lidar2cam_sweeps = [
                            lidar2cam_rts[cam_idx] for _ in range(cam_sweep_num)
                        ]
                        intrinsics_sweeps = [
                            cam_intrinsics[cam_idx] for _ in range(cam_sweep_num)
                        ]
                    else:
                        cam_sweeps = []
                        cam_ids = []
                        cam_time = []
                        lidar2img_sweeps = []
                        lidar2cam_sweeps = []
                        intrinsics_sweeps = []
                        for sweep_id, sweep_info in enumerate(
                            cam_infos[:cam_sweep_num]
                        ):
                            cam_sweeps.append(sweep_info["data_path"])
                            cam_ids.append(sweep_id)
                            cam_time.append(
                                input_dict["timestamp"] - sweep_info["timestamp"] / 1e6
                            )
                            # obtain lidar to image transformation matrix
                            lidar2cam_r = np.linalg.inv(
                                sweep_info["sensor2lidar_rotation"]
                            )
                            lidar2cam_t = (
                                sweep_info["sensor2lidar_translation"] @ lidar2cam_r.T
                            )
                            lidar2cam_rt = np.eye(4)
                            lidar2cam_rt[:3, :3] = lidar2cam_r.T
                            lidar2cam_rt[3, :3] = -lidar2cam_t
                            intrinsic = sweep_info["cam_intrinsic"]
                            viewpad = np.eye(4)
                            viewpad[
                                : intrinsic.shape[0], : intrinsic.shape[1]
                            ] = intrinsic
                            lidar2img_rt = viewpad @ lidar2cam_rt.T
                            lidar2img_sweeps.append(lidar2img_rt)
                            lidar2cam_sweeps.append(lidar2cam_rt.T)
                            intrinsics_sweeps.append(viewpad)

                    # pad empty sweep with the last frame
                    if len(cam_sweeps) < cam_sweep_num:
                        cam_req = cam_sweep_num - len(cam_infos)
                        cam_ids = cam_ids + [cam_ids[-1] for _ in range(cam_req)]
                        cam_time = cam_time + [cam_time[-1] for _ in range(cam_req)]
                        cam_sweeps = cam_sweeps + [
                            cam_sweeps[-1] for _ in range(cam_req)
                        ]
                        lidar2img_sweeps = lidar2img_sweeps + [
                            lidar2img_sweeps[-1] for _ in range(cam_req)
                        ]
                        lidar2cam_sweeps = lidar2cam_sweeps + [
                            lidar2cam_sweeps[-1] for _ in range(cam_req)
                        ]
                        intrinsics_sweeps = intrinsics_sweeps + [
                            intrinsics_sweeps[-1] for _ in range(cam_req)
                        ]

                    # align to start time
                    cam_time = [_time - cam_time[0] for _time in cam_time]
                    # sweep id from 0->prev 1->prev 2
                    if cam_sweeps[0] == image_paths[cam_idx]:
                        cam_sweeps_paths.append(cam_sweeps[1:cam_sweep_num])
                        cam_sweeps_id.append(cam_ids[1:cam_sweep_num])
                        cam_sweeps_time.append(cam_time[1:cam_sweep_num])
                        lidar2img_sweeps_rts.append(lidar2img_sweeps[1:cam_sweep_num])
                        lidar2cam_sweeps_rts.append(lidar2cam_sweeps[1:cam_sweep_num])
                        cam_sweeps_intrinsics.append(intrinsics_sweeps[1:cam_sweep_num])
                    else:
                        raise ValueError

                if "cam_sweep_list" in self.modality:
                    sweep_list = self.modality["cam_sweep_list"]
                    for cam_idx in range(len(cam_sweeps_paths)):
                        cam_sweeps_paths[cam_idx] = [
                            cam_sweeps_paths[cam_idx][i] for i in sweep_list
                        ]
                        cam_sweeps_id[cam_idx] = [
                            cam_sweeps_id[cam_idx][i] for i in sweep_list
                        ]
                        cam_sweeps_time[cam_idx] = [
                            cam_sweeps_time[cam_idx][i] for i in sweep_list
                        ]
                        cam_sweeps_intrinsics[cam_idx] = [
                            cam_sweeps_intrinsics[cam_idx][i] for i in sweep_list
                        ]

                input_dict.update(
                    dict(
                        cam_sweeps_paths=cam_sweeps_paths,
                        cam_sweeps_id=cam_sweeps_id,
                        cam_sweeps_time=cam_sweeps_time,
                        lidar2img_sweeps=lidar2img_sweeps_rts,
                        lidar2cam_sweeps=lidar2cam_sweeps_rts,
                        cam_sweeps_intrinsics=cam_sweeps_intrinsics,
                    )
                )

        # self.test_mode = False
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_data_info(self, index):
        """Get data info according to the given index.
        """
        input_dict = self.load_data_info(index)
        # input_dict = super().get_data_info(index)

        # obtain the current lidar to ego transformation
        curr_lidar_to_ego = rt2mat(input_dict['lidar2ego_translation'],
                                   input_dict['lidar2ego_rotation'])
        curr_lidarego_to_global = rt2mat(input_dict['ego2global_translation'],
                                         input_dict['ego2global_rotation'])
        
        curr_cam_to_ego = np.stack(input_dict['cam2camego'])  # (6, 4, 4)
        curr_camego_to_global = np.stack(input_dict['camego2global'])  # (6, 4, 4)
        curr_cam_to_global = curr_camego_to_global @ curr_cam_to_ego

        for idx in self.future_frames:
            adj_idx = index + idx
            select_id = max(min(adj_idx, len(self) - 1), 0)
            adj_info = self.data_infos[select_id]
            curr_info = self.data_infos[index]
            if adj_info['scene_token'] != curr_info['scene_token']:
                adj_info = curr_info

            # obtain the current ego to future ego transformation, please 
            # note that the 'ego' here is lidar coordinate system.
            adj_global2lidarego = rt2mat(adj_info['ego2global_translation'],
                                         adj_info['ego2global_rotation'],
                                         inverse=True)
            adj_lidarego2lidar = rt2mat(adj_info['lidar2ego_translation'],
                                        adj_info['lidar2ego_rotation'],
                                        inverse=True)
            # (4, 4) matrix
            curr_lidar_to_adj_lidar = \
                adj_lidarego2lidar @ adj_global2lidarego @ curr_lidarego_to_global @ curr_lidar_to_ego

            future_pose_spatial = np.stack(adj_info['lidar2cam'])  # (6, 4, 4)
            future_cam_intrinsic = np.stack(adj_info['cam_intrinsic'])  # (6, 4, 4)

            if idx == -1:
                flag = 'prev'
            else:
                flag = 'future'
            
            input_dict[f'pose_spatial_{flag}'] = torch.from_numpy(future_pose_spatial).to(torch.float32)  # (6, 4, 4)
            input_dict[f'cam_intrinsic_{flag}'] = torch.from_numpy(future_cam_intrinsic).to(torch.float32)
            input_dict[f'curr_lidar_T_{flag}_lidar'] = torch.from_numpy(curr_lidar_to_adj_lidar)[None].to(torch.float32)

            curr_cam_to_cam_j = cam_i_to_cam_j_transform(curr_cam_to_global,
                                                         adj_info['cams'])
            input_dict[f'curr_cam_T_{flag}_cam'] = torch.from_numpy(curr_cam_to_cam_j).to(torch.float32)  # (6, 4, 4)

            input_dict[f'{flag}_info'] = adj_info

            if self.use_flow_photometric_loss:
                # load the adjacent information for the flow-based self-supervised learning
                future_adj_dict = load_adjacent_info(adj_info, self.extra_frames)
                ## add the "_future" suffix to the key
                future_adj_dict = {f'{key}_{flag}': value for key, value in future_adj_dict.items()}
                input_dict.update(future_adj_dict)

        if self.use_depth_consistency:
            curr_adj_dict = load_adjacent_info(input_dict, self.extra_frames)
            input_dict.update(curr_adj_dict)
        
        return input_dict
    

@DATASETS.register_module()
class NuScenesSweepDatasetFutureOverfit(NuScenesSweepDatasetFuture):
    r"""Overfit the future frames information for the flow-based self-supervised learning.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        idx = 50
        return super().__getitem__(idx)
    

@DATASETS.register_module()
class NuScenesSweepDatasetAdjacent(NuScenesSweepDatasetFuture):
    r"""Not only load the future frames information but also the previous frames information.
    """
    def __init__(
        self,
        prev_frames=[-1],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.prev_frames = prev_frames
        assert len(self.prev_frames) == 1, 'Currently we only support one previous key frames.'

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)

        for idx in self.prev_frames:
            adj_idx = index + idx
            select_id = max(min(adj_idx, len(self) - 1), 0)
            adj_info = self.data_infos[select_id]
            curr_info = self.data_infos[index]
            if adj_info['scene_token'] != curr_info['scene_token']:
                adj_info = curr_info

            # obtain the current ego to future ego transformation, please 
            # note that the 'ego' here is lidar coordinate system.
            adj_global2lidarego = rt2mat(adj_info['ego2global_translation'],
                                         adj_info['ego2global_rotation'],
                                         inverse=True)
            adj_lidarego2lidar = rt2mat(adj_info['lidar2ego_translation'],
                                        adj_info['lidar2ego_rotation'],
                                        inverse=True)
            # (4, 4) matrix
            curr_lidar_to_future_lidar = adj_lidarego2lidar @ adj_global2lidarego @ curr_lidarego_to_global @ curr_lidar_to_ego

            future_pose_spatial_list, future_cam_intrinsic_list = [], []
            lidar2img_list = []
            for cam_type, cam_info in adj_info['cams'].items():
                cam2lidar = rt2mat(cam_info['sensor2lidar_translation'],
                                   rotation=cam_info['sensor2lidar_rotation'])
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                
                future_pose_spatial_list.append(cam2lidar)
                future_cam_intrinsic_list.append(viewpad)

                lidar2img_rt = viewpad @ np.linalg.inv(cam2lidar)
                lidar2img_list.append(lidar2img_rt)
            
            adj_info['lidar2img'] = lidar2img_list

            future_pose_spatial = np.stack(future_pose_spatial_list)  # (6, 4, 4)
            future_cam_intrinsic = np.stack(future_cam_intrinsic_list)  # (6, 4, 4)

            input_dict['pose_spatial_future'] = torch.from_numpy(future_pose_spatial).to(torch.float32)  # (6, 4, 4)
            input_dict['cam_intrinsic_future'] = torch.from_numpy(future_cam_intrinsic).to(torch.float32)
            input_dict['curr_lidar_T_future_lidar'] = torch.from_numpy(curr_lidar_to_future_lidar)[None].to(torch.float32)
            input_dict['future_info'] = adj_info

            if self.use_flow_photometric_loss:
                # load the adjacent information for the flow-based self-supervised learning
                future_adj_dict = load_adjacent_info(adj_info, self.extra_frames)
                ## add the "_future" suffix to the key
                future_adj_dict = {key + '_future': value for key, value in future_adj_dict.items()}
                input_dict.update(future_adj_dict)