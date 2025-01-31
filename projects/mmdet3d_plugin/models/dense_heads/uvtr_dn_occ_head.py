
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply, multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
    build_bbox_coder,
)
from mmcv.cnn import xavier_init, constant_init
from .. import utils
from projects.mmdet3d_plugin.core.bbox.iou_calculators import PairedBboxOverlaps3D


@HEADS.register_module()
class UVTRDNOccHead(BaseModule):
    """Head of UVTR.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        in_channels,
        embed_dims=128,
        num_query=900,
        num_reg_fcs=2,
        bg_cls_weight=0,
        num_classes=10,
        use_mask=True,
        loss_occ=None,
        sync_cls_avg_factor=False,
        unified_conv=None,
        view_cfg=None,
        with_box_refine=False,
        transformer=None,
        bbox_coder=None,
        loss_bbox=None,
        loss_cls=None,
        train_cfg=None,
        test_cfg=None,
        code_weights=None,
        split=0.75,
        dn_weight=1.0,
        iou_weight=0.5,
        scalar=10,
        bbox_noise_scale=1.0,
        bbox_noise_trans=0.0,
        init_cfg=None,
        fp16_enabled=True,
        **kwargs,
    ):
        super(UVTRDNOccHead, self).__init__(init_cfg)

        mid_channels = 32
        self.final_conv = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        
        self.predicter = nn.Sequential(
            nn.Linear(mid_channels, mid_channels*2),
            nn.Softplus(),
            nn.Linear(mid_channels*2, num_classes),
        )

        self.use_mask = use_mask

        self.loss_occ = build_loss(loss_occ)

        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_box_refine = with_box_refine
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_reg_fcs = num_reg_fcs
        self.bg_cls_weight = bg_cls_weight
        self.unified_conv = unified_conv
        self.scalar = scalar
        self.bbox_noise_scale = bbox_noise_scale
        self.bbox_noise_trans = bbox_noise_trans
        self.dn_weight = dn_weight
        self.iou_weight = iou_weight
        self.split = split
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

        if view_cfg is not None:
            vtrans_type = view_cfg.pop("type", "Uni3DViewTrans")
            self.view_trans = getattr(utils, vtrans_type)(**view_cfg)

        if self.unified_conv is not None:
            self.conv_layer = []
            in_c = (
                embed_dims * 2 if self.unified_conv["fusion"] == "cat" else embed_dims
            )
            for k in range(self.unified_conv["num_conv"]):
                conv = nn.Sequential(
                    nn.Conv3d(
                        in_c, embed_dims, kernel_size=3, stride=1, padding=1, bias=True
                    ),
                    nn.BatchNorm3d(embed_dims),
                    nn.ReLU(inplace=True),
                )
                in_c = embed_dims
                self.add_module("{}_head_{}".format("conv_trans", k + 1), conv)
                self.conv_layer.append(conv)

        # self.transformer = build_transformer(transformer)
        # self._init_layers()

        # self.iou_calculator = PairedBboxOverlaps3D(coordinate="lidar")

        # if train_cfg:
        #     self.assigner = build_assigner(train_cfg["assigner"])
        #     sampler_cfg = dict(type="PseudoSampler")
        #     self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        iou_branch = []
        for _ in range(self.num_reg_fcs):
            iou_branch.append(Linear(self.embed_dims, self.embed_dims))
            iou_branch.append(nn.ReLU())
        iou_branch.append(Linear(self.embed_dims, 1))
        iou_branch = nn.Sequential(*iou_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # def _get_clones(module, N):
        #     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        # num_pred = self.transformer.decoder.num_layers

        # if self.with_box_refine:
        #     self.cls_branches = _get_clones(fc_cls, num_pred)
        #     self.iou_branches = _get_clones(iou_branch, num_pred)
        #     self.reg_branches = _get_clones(reg_branch, num_pred)
        # else:
        #     self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
        #     self.iou_branches = nn.ModuleList([iou_branch for _ in range(num_pred)])
        #     self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        # self.query_embedding = nn.Sequential(
        #     nn.Linear(3, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims, self.embed_dims),
        # )
        # self.reference_points = nn.Embedding(self.num_query, 3)

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(
        self,
        pts_feats,
        img_feats,
        img_metas,
        img_depth,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        with_image, with_point = True, True
        if img_feats is None:
            with_image = False
        elif isinstance(img_feats, dict) and img_feats["key"] is None:
            with_image = False

        if pts_feats is None:
            with_point = False
        elif isinstance(pts_feats, dict) and pts_feats["key"] is None:
            with_point = False
            pts_feats = None

        # transfer to voxel level
        if with_image:
            img_feats = self.view_trans(
                img_feats, img_metas=img_metas, img_depth=img_depth
            )
        # shape: (N, L, C, D, H, W)
        if with_point:
            if len(pts_feats.shape) == 5:
                pts_feats = pts_feats.unsqueeze(1)

        if self.unified_conv is not None:
            raw_shape = pts_feats.shape
            if self.unified_conv["fusion"] == "sum":
                unified_feats = pts_feats.flatten(1, 2) + img_feats.flatten(1, 2)
            else:
                unified_feats = torch.cat(
                    [pts_feats.flatten(1, 2), img_feats.flatten(1, 2)], dim=1
                )
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)
            unified_feats = unified_feats.reshape(*raw_shape)
        else:
            unified_feats = pts_feats if pts_feats is not None else img_feats

        unified_feats = unified_feats.squeeze(1)  # (B, C, Z, Y, X)

        occ_pred = self.final_conv(unified_feats).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        occ_pred = self.predicter(occ_pred)

        outs = dict(occ_pred=occ_pred)
        return outs

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        try:
            assign_result = self.assigner.assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels
            )
        except:
            print(
                "bbox_pred:{}, cls_score:{}, gt_bboxes:{}, gt_labels:{}".format(
                    (bbox_pred.max(), bbox_pred.min()),
                    (cls_score.max(), cls_score.min()),
                    (gt_bboxes.max(), gt_bboxes.min()),
                    gt_labels,
                )
            )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., : gt_bboxes.shape[1]]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def get_targets(
        self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list
    ):
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def dn_loss_single(self, cls_scores, bbox_preds, iou_preds, mask_dict):
        known_labels, known_bboxs = mask_dict["known_lbs_bboxes"]
        known_bid, map_known_indice = (
            mask_dict["known_bid"].long(),
            mask_dict["map_known_indice"].long(),
        )

        cls_scores = cls_scores[known_bid, map_known_indice]
        bbox_preds = bbox_preds[known_bid, map_known_indice]
        iou_preds = iou_preds[known_bid, map_known_indice]
        num_tgt = map_known_indice.numel()

        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(bbox_preds)
        bbox_code_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_code_weights[isnotnan, :10],
            avg_factor=num_tgt,
        )

        denormalized_bbox_preds = denormalize_bbox(
            bbox_preds[isnotnan, :8].detach(), self.pc_range
        ).clone()
        denormalized_bbox_preds[:, 2] = (
            denormalized_bbox_preds[:, 2] - denormalized_bbox_preds[:, 5] * 0.5
        )

        denormalized_bbox_targets = known_bboxs[
            isnotnan, :7
        ].clone()  # (x, y, z, w, l, h, rot)
        denormalized_bbox_targets[:, 2] = (
            denormalized_bbox_targets[:, 2] - denormalized_bbox_targets[:, 5] * 0.5
        )

        iou_preds = iou_preds[isnotnan, 0]
        iou_targets = self.iou_calculator(
            denormalized_bbox_preds, denormalized_bbox_targets
        )
        valid_index = torch.nonzero(
            iou_targets * bbox_weights[isnotnan, 0], as_tuple=True
        )[0]
        num_pos = valid_index.shape[0]
        iou_targets = iou_targets * 2 - 1
        loss_iou = F.l1_loss(
            iou_preds[valid_index], iou_targets[valid_index], reduction="sum"
        ) / max(num_pos, 1)

        # loss_cls = torch.nan_to_num(loss_cls)
        # loss_bbox = torch.nan_to_num(loss_bbox)
        return (
            self.dn_weight * loss_cls,
            self.dn_weight * loss_bbox,
            self.dn_weight * self.iou_weight * loss_iou,
        )

    def loss_single(
        self, cls_scores, bbox_preds, iou_preds, gt_bboxes_list, gt_labels_list
    ):
        batch_size = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        bbox_preds_list = [bbox_preds[i] for i in range(batch_size)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list
        )

        labels = torch.cat(labels_list, dim=0)
        label_weights = torch.cat(label_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        # classification loss
        cls_scores = cls_scores.flatten(0, 1)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.flatten(0, 1)
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_code_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_code_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )

        denormalized_bbox_preds = denormalize_bbox(
            bbox_preds[isnotnan, :8].detach(), self.pc_range
        ).clone()
        denormalized_bbox_preds[:, 2] = (
            denormalized_bbox_preds[:, 2] - denormalized_bbox_preds[:, 5] * 0.5
        )

        denormalized_bbox_targets = bbox_targets[
            isnotnan, :7
        ].clone()  # (x, y, z, w, l, h, rot)
        denormalized_bbox_targets[:, 2] = (
            denormalized_bbox_targets[:, 2] - denormalized_bbox_targets[:, 5] * 0.5
        )

        iou_preds = iou_preds.flatten(0, 1)[isnotnan, 0]
        iou_targets = self.iou_calculator(
            denormalized_bbox_preds, denormalized_bbox_targets
        )
        valid_index = torch.nonzero(
            iou_targets * bbox_weights[isnotnan, 0], as_tuple=True
        )[0]
        num_pos = valid_index.shape[0]
        iou_targets = iou_targets * 2 - 1
        loss_iou = F.l1_loss(
            iou_preds[valid_index], iou_targets[valid_index], reduction="sum"
        ) / max(num_pos, 1)

        # loss_cls = torch.nan_to_num(loss_cls)
        # loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, self.iou_weight * loss_iou

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, preds_dicts, target_dict):
        voxel_semantics = target_dict["voxel_semantics"]
        mask_camera = target_dict["mask_camera"]
        preds = preds_dicts["occ_pred"]

        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]["box_type_3d"](bboxes, bboxes.size(-1))
            scores = preds["scores"]
            labels = preds["labels"]
            ret_list.append([bboxes, scores, labels])
        return ret_list
