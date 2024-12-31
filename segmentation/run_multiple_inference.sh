#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_imgcat_kitti.py \
  --checkpoint work_dirs/mask2former_soft_internimage_b_imgcat_kitti_balanced_sampler_30_loss_cls_dice_soft_fix/best_mIoU_soft_iter_192000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_imgcurv_kitti.py \
  --checkpoint work_dirs/mask2former_soft_internimage_b_imgcurv_kitti_balanced_sampler_30_loss_cls_dice_soft_fix/best_mIoU_soft_iter_176000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_attcat_kitti.py \
  --checkpoint work_dirs/mask2former_soft_internimage_b_attcat_kitti_balanced_sampler_30_loss_cls_dice_soft/best_mIoU_soft_iter_96000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

#CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_attcurv_kitti.py \
#  --checkpoint work_dirs/mask2former_soft_internimage_b_attcurv_kitti_balanced_sampler_30_loss_cls_dice_soft/best_mIoU_soft_iter_144000.pth \
#  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
#  --split test \
#  --horizon 30 \
#  --save-dir to_del
#
#CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_atttext_kitti.py \
#  --checkpoint work_dirs/mask2former_soft_internimage_b_atttext_kitti_balanced_sampler_30_loss_cls_dice_soft_fix/best_mIoU_soft_iter_64000.pth \
#  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
#  --split test \
#  --horizon 30 \
#  --save-dir to_del

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_attcat_kitti.py \
  --checkpoint /raid/andreim/InternImage/work_dirs/mask2former_soft_internimage_b_attcat_kitti_balanced_sampler_30_final/best_mIoU_soft_iter_64000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_attcurv_kitti.py \
  --checkpoint /raid/andreim/InternImage/work_dirs/mask2former_soft_internimage_b_attcurv_kitti_balanced_sampler_30_final/best_mIoU_soft_iter_32000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_atttext_kitti.py \
  --checkpoint /raid/andreim/InternImage/work_dirs/mask2former_soft_internimage_b_atttext_kitti_balanced_sampler_30_final/best_mIoU_soft_iter_112000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_imgcat_kitti.py \
  --checkpoint /raid/andreim/InternImage/work_dirs/mask2former_soft_internimage_b_imgcat_kitti_balanced_sampler_30_final/best_mIoU_soft_iter_128000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --save-dir to_del

#CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_imgtext_kitti.py \
#  --checkpoint /raid/andreim/InternImage/work_dirs/mask2former_soft_internimage_b_imgtext_kitti_balanced_sampler_30_final/best_mIoU_soft_iter_192000.pth \
#  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
#  --split test \
#  --horizon 30 \
#  --save-dir to_del
#
#CUDA_VISIBLE_DEVICES=0 python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_kitti.py \
#  --checkpoint /raid/andreim/InternImage/work_dirs/mask2former_soft_internimage_b_kitti_balanced_sampler_30_final/best_mIoU_soft_iter_32000.pth \
#  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
#  --split test \
#  --horizon 30 \
#  --save-dir to_del