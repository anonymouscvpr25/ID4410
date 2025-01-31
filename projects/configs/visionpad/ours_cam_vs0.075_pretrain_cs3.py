_base_ = [
    "../../../configs/_base_/datasets/nus-3d.py",
    "../../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
unified_voxel_size = [0.6, 0.6, 1.6]  # [180, 180, 5]
frustum_range = [0, 0, 1.0, 1600, 928, 60.0]
frustum_size = [32.0, 32.0, 0.5]
cam_sweep_num = 3
lidar_sweep_num = 10
fp16_enabled = True
unified_voxel_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
]

render_size = [360, 640]
depth_ssl_size = [360, 640]  # the image size used for warping in depth SSL
use_semantic = False
use_flow_photometric_loss = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=cam_sweep_num,
)

model = dict(
    type="UVTRSSL3DGS",
    img_backbone=dict(
        type="MaskConvNeXt",
        arch="small",
        drop_path_rate=0.2,
        out_indices=(3),
        norm_out=True,
        frozen_stages=1,
        with_cp=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="data/ckpts/convnextS_1kpretrained_official_style.pth",
        ),
        mae_cfg=dict(
            downsample_scale=32, downsample_dim=768, mask_ratio=0.3, learnable=False
        ),
    ),
    img_neck=dict(
        type="CustomFPN",
        in_channels=[768],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    depth_head=dict(type="ComplexDepth", use_dcn=False, aspp_mid_channels=96),
    pts_bbox_head=dict(
        type="PretrainHead",
        in_channels=256,
        # fp16_enabled=fp16_enabled,
        # save_dir="results/vis/3dgs_cam_vs0.1_pretrain_depth_ssl_filter_gs",
        pred_flow=True,
        use_flow_ssl=True,
        use_flow_photometric_loss=use_flow_photometric_loss,  # whether to use photometric loss or GT depth loss for flow
        use_flow_rgb=True,
        use_flow_refine_layer=True,
        use_sperate_render_head=True,
        flow_depth_loss_weight=0.1,
        rgb_future_loss_weight=0.15,
        voxel_shape=unified_voxel_shape,
        voxel_size=unified_voxel_size,
        use_depth_consistency=True,
        depth_loss_weight=10.0,
        rgb_loss_weight=0.25,
        opt=dict(
            avg_reprojection=False,
            disable_automasking=False,
            disparity_smoothness=0.001,
        ),
        use_semantic=use_semantic,
        depth_ssl_size=depth_ssl_size,  # the image size for image warping in depth SSL
        render_scale=[render_size[0] / 900, 
                      render_size[1] / 1600],
        render_head_cfg=dict(
            type="GaussianSplattingDecoder",
            in_channels=32,
            filter_opacities=True,
            semantic_head=use_semantic,
            render_size=render_size,
            depth_range=[0.1, 64],
            pc_range=point_cloud_range,
            voxels_size=unified_voxel_shape,
            volume_size=unified_voxel_shape,
            learn_gs_scale_rot=True,
            offset_scale=0.25,
            gs_scale=0.3,
            gs_scale_min=0.1,
            gs_scale_max=0.5,
        ),
        view_cfg=dict(
            type="Uni3DVoxelPoolDepth",
            pc_range=point_cloud_range,
            voxel_size=unified_voxel_size,
            voxel_shape=unified_voxel_shape,
            frustum_range=frustum_range,
            frustum_size=frustum_size,
            num_convs=0,
            kernel_size=(3,3,3),
            embed_dim=256,
            sweep_fusion=dict(type='sweep_cat_with_time'),
            num_sweeps=cam_sweep_num,
            keep_sweep_dim=False,
            fp16_enabled=False,
            loss_cfg=dict(close_radius=3.0, depth_loss_weights=[0.0]),
        ),
        uni_conv_cfg=dict(
            in_channels=256,
            out_channels=32, 
            kernel_size=3, 
            padding=1
        ),
    ),
)

dataset_type = "NuScenesSweepDatasetFuture"
data_root = "data/nuscenes/"

file_client_args = dict(backend="disk")

train_pipeline = [
    # dict(
    #     type="LoadPointsFromFile",
    #     coord_type="LIDAR",
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args,
    # ),
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(
        type='PrepapreImageInputs',
        input_size=depth_ssl_size,
        to_float32=True,
        load_future_img=True,
    ),
    # dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    # dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # dict(type="PointToMultiViewDepth",
    #      render_scale=[render_size[0] / 900, 
    #                    render_size[1] / 1600],
    #      render_size=render_size),
    # dict(type='PrepareTemporalData',
    #      pipeline=[
    #         dict(type='LoadPointsFromFile',
    #              coord_type='LIDAR',
    #              load_dim=5,
    #              use_dim=5,
    #              file_client_args=file_client_args),
    #         dict(type='PointsRangeFilter',
    #              point_cloud_range=point_cloud_range),
    #         dict(type='PointShuffle'),
    #         dict(type='PointToMultiViewDepth',
    #              render_scale=[render_size[0] / 900,
    #                            render_size[1] / 1600],
    #              render_size=render_size),
    #     ],
    # ),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["img",
                                        "source_imgs", "target_imgs",
                                        "K", "inv_K", "cam_T_cam",
                                        "source_imgs_future", 
                                        "target_imgs_future",
                                        "curr_lidar_T_future_lidar",
                                        "K_future", "inv_K_future",
                                        "cam_T_cam_future",
                                        "pose_spatial_future",
                                        "cam_intrinsic_future"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    # dict(type='LoadLiDARSegGTFromFile'),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="PointToMultiViewDepth",
         render_scale=[render_size[0] / 900, 
                      render_size[1] / 1600],
         render_size=render_size),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img",
                                        "render_gt_depth", 
                                        "render_gt_semantic"]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        use_depth_consistency=True,
        use_flow_photometric_loss=use_flow_photometric_loss,
        future_frames=[1],
        data_root=data_root,
        ann_file=data_root
        + "nuscenes_unified_infos_train_v4.pkl",  # please change to your own info file
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
        load_interval=1,
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val_v4.pkl",
        load_interval=1,
    ),  # please change to your own info file
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_train_v4.pkl",
        load_interval=1,
    ),
)  # please change to your own info file

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 12
evaluation = dict(interval=4, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

find_unused_parameters = False
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
load_from = None
resume_from = None
# fp16 setting
fp16 = dict(loss_scale=32.0)
