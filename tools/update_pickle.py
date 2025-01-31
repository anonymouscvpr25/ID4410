import os
import os.path as osp
import numpy as np
import pickle
from pyquaternion import Quaternion
from nuscenes import NuScenes
from tqdm import tqdm
import mmengine
import mmcv


def explore_dataset(split='train'):
    need_sample_token = "4bf467a49831433286692e10c5340f48"

    pickle_file = f'data/nuscenes/nuscenes_unified_infos_{split}_v2.pkl'
    print(f'Start process {pickle_file}...')

    data = mmengine.load(pickle_file)
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    print(f"Total length of data infos: {len(data_infos)}")

    for index, info in enumerate(tqdm(data_infos)):
        if info['token'] == need_sample_token:
            print(index)
            break


def obtain_cam_info(nusc,
                    sensor_token,
                    sensor_type):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
        'cam_intrinsic': np.array(cs_record['camera_intrinsic']),
        'is_key_frame': sd_rec['is_key_frame'],
    }
    return sweep


def add_prev_next_sweeps(split=['train', 'val']):
    if not isinstance(split, list):
        split = [split]
    
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nusc = NuScenes(nuscenes_version, dataroot)

    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_v2.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']

        frame_idx = [-1, 1]

        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        
        for info in mmcv.track_iter_progress(data_infos):
            sample = nusc.get('sample', info['token'])
            
            for adj_idx in frame_idx:
                if adj_idx == -1:
                    flag = 'prev'
                else:
                    flag = 'next'

                info[flag] = {}

                for cam in camera_types:
                    cam_sample = nusc.get('sample_data', sample['data'][cam])
                    
                    adj_sample_token = cam_sample[flag]  # get the previous or next sample token
                    if len(adj_sample_token) == 0:
                        info[flag][cam] = []
                    else:
                        cam_info = obtain_cam_info(nusc, adj_sample_token, cam)
                        info[flag][cam] = cam_info
                    
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        new_filename = f"{filename}_v3{ext}"
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


def update_infos(split=['train', 'val'],
                 add_occ_path=True,
                 add_scene_token=True,
                 add_lidarseg_path=False,
                 add_transforms=False):
    """Add different extra needed information in the nuScenes dataset to the pickle file.

    Args:
        split (list, optional): _description_. Defaults to ['train', 'val'].
        add_occ_path: whether to add the occ path to the pickle file.
        add_scene_token: whether to add the scene token to the pickle file.
        add_lidarseg_path: whether to add the lidarseg path to the pickle file.
    """
    if not isinstance(split, list):
        split = [split]
    
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    
    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_v3.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']

        for info in mmcv.track_iter_progress(data_infos):
            sample = nuscenes.get('sample', info['token'])
            scene = nuscenes.get('scene', sample['scene_token'])
            
            if add_occ_path:
                info['occ_path'] = \
                    './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
            if add_scene_token:
                info['scene_token'] = sample['scene_token']

            if add_lidarseg_path:
                lidar_token = sample['data']['LIDAR_TOP']
                lidarseg_label = os.path.join(nuscenes.dataroot, 
                                              nuscenes.get('lidarseg', lidar_token)['filename'])
                info['lidarseg'] = lidarseg_label

            if add_transforms:
                image_paths = []
                lidar2img_rts = []
                # add lidar2img matrix
                lidar2cam_rts = []
                cam_intrinsics = []
                cam2camego_list = []
                camego2global_list = []
                for cam_type, cam_info in info["cams"].items():
                    image_paths.append(cam_info["data_path"])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                    lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info["cam_intrinsic"]
                    viewpad = np.eye(4)
                    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = viewpad @ lidar2cam_rt.T
                    lidar2img_rts.append(lidar2img_rt)
                    lidar2cam_rts.append(lidar2cam_rt.T)
                    cam_intrinsics.append(viewpad)

                    # obtain the camera to ego transformation matrix
                    cam2camego = np.eye(4, dtype=np.float32)
                    cam2camego[:3, :3] = Quaternion(
                        cam_info['sensor2ego_rotation']).rotation_matrix
                    cam2camego[:3, 3] = cam_info['sensor2ego_translation']
                    cam2camego_list.append(cam2camego)

                    # obtain the ego to global transformation matrix
                    ego2global = np.eye(4, dtype=np.float32)
                    ego2global[:3, :3] = Quaternion(
                        cam_info['ego2global_rotation']).rotation_matrix
                    ego2global[:3, 3] = cam_info['ego2global_translation']
                    camego2global_list.append(ego2global)

                info.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                        lidar2cam=lidar2cam_rts,
                        cam_intrinsic=cam_intrinsics,
                        cam2camego=cam2camego_list,
                        camego2global=camego2global_list,
                    )
                )
        
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        # new_filename = f"{filename}_v2{ext}"
        new_filename = filename.replace('_v3', '_v4') + ext
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


def use_ego_centric_infos(split=['train', 'val']):
    """Here we use the ego-centric information to update the pickle file.
    In this case, the original lidar2xxx will be replaced by the camego2xxx.

    Args:
        split (list, optional): _description_. Defaults to ['train', 'val'].
    """
    if not isinstance(split, list):
        split = [split]
    
    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_v4.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']
        
        for info in mmcv.track_iter_progress(data_infos):
            ego2img_rts = []
            ego2cam_rts = []
            cam2camego = info['cam2camego']
            cam_intrinsic = info['cam_intrinsic']
            for cam_idx in range(len(info['cams'])):
                _cam2camego = cam2camego[cam_idx]
                _intrinsic = cam_intrinsic[cam_idx]
                _camego2cam = np.linalg.inv(_cam2camego)
                ego2cam_rts.append(_camego2cam)

                viewpad = np.eye(4)
                viewpad[: _intrinsic.shape[0], : _intrinsic.shape[1]] = _intrinsic

                ego2img = viewpad @ _camego2cam
                ego2img_rts.append(ego2img)
            info['lidar2cam'] = ego2cam_rts
            info['lidar2img'] = ego2img_rts

            for cam_type, cam_info in info["cams"].items():
                cam_info['sensor2lidar_rotation'] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                cam_info['sensor2lidar_translation'] = cam_info['sensor2ego_translation']

                for _cam_sweep in info['cam_sweeps_info'][cam_type]:
                    _cam_sweep['sensor2lidar_rotation'] = Quaternion(_cam_sweep['sensor2ego_rotation']).rotation_matrix
                    _cam_sweep['sensor2lidar_translation'] = _cam_sweep['sensor2ego_translation']

            ## make lidar2ego as the identity matrix
            info['lidar2ego_translation'] = [0.0, 0.0, 0.0]
            info['lidar2ego_rotation'] = [1, 0.0, 0.0, 0.0]
    
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        # new_filename = f"{filename}_v2{ext}"
        new_filename = filename.replace('_v4', '_v4_ego') + ext
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


def add_lidarseg_path(split=['train', 'val']):
    if not isinstance(split, list):
        split = [split]

    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_occ.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']

        bevdet_pickle_file = f'data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_{s}.pkl'
        bevdet_data = mmengine.load(bevdet_pickle_file)
        bevdet_data_infos = bevdet_data['infos']

        assert len(data_infos) == len(bevdet_data_infos)

        for idx in mmcv.track_iter_progress(range(len(data_infos))):
            assert data_infos[idx]['token'] == bevdet_data_infos[idx]['token']
            data_infos[idx]['lidarseg'] = bevdet_data_infos[idx]['lidarseg']
        
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        new_filename = f"{filename}_v2{ext}"
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


def add_lidarseg_path_v2(split=['train', 'val']):
    if not isinstance(split, list):
        split = [split]
    
    version = 'v1.0-trainval'
    data_root = 'data/nuscenes'
    nuscenes = NuScenes(version, data_root)

    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_v3.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']
    
        for info in mmcv.track_iter_progress(data_infos):
            token = info['token']
            lidar_token = nuscenes.get('sample', token)['data']['LIDAR_TOP']

            lidarseg_label = os.path.join(nuscenes.dataroot, 
                                          nuscenes.get('lidarseg', lidar_token)['filename'])
            info['lidarseg'] = lidarseg_label

        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        new_filename = f"{filename}_lidarseg{ext}"
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


if __name__ == '__main__':
    # update_infos(['train', 'val'], 
    #              add_occ_path=False, 
    #              add_scene_token=False,
    #              add_transforms=True)
    use_ego_centric_infos(['train', 'val'])
    exit()
    # add_lidarseg_path_v2(['train', 'val'])
    # exit()
    explore_dataset()
    exit()
    add_prev_next_sweeps(['train', 'val'])
    exit()

    pickle_file = 'data/nuscenes/nuscenes_unified_infos_train.pkl'
    pickle_file = 'data/nuscenes/nuscenes_unified_infos_val.pkl'
    update_infos(['train', 'val'], add_occ_path=True, add_scene_token=True)
    # add_lidarseg_path(['train', 'val'])
    exit()
    data = mmengine.load(pickle_file)
    data_infos = data['infos']
    print(data['metadata'])
    print(len(data_infos))
