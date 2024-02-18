#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python run_dataset.py --config configs/mask2former/mask2former_internimage_b_kitti.py \
  --checkpoint work_dirs/mask2former_internimage_b_kitti_balanced_sampler_30/best_mIoU_iter_144000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --additional_config configs/mask2former/mask2former_internimage_b_imgcat_kitti.py \
  --additional_checkpoint work_dirs/mask2former_internimage_b_imgcat_kitti_balanced_sampler_30/best_mIoU_iter_160000.pth \
  --save-dir guidance_failure

CUDA_VISIBLE_DEVICES=0 python run_dataset.py --config configs/mask2former/mask2former_internimage_b_kitti.py \
  --checkpoint work_dirs/mask2former_internimage_b_kitti_balanced_sampler_30/best_mIoU_iter_144000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --additional_config configs/mask2former/mask2former_internimage_b_imgcurv_kitti.py \
  --additional_checkpoint work_dirs/mask2former_internimage_b_imgcurv_kitti_balanced_sampler_30/best_mIoU_iter_96000.pth \
  --save-dir guidance_failure

CUDA_VISIBLE_DEVICES=0 python run_dataset.py --config configs/mask2former/mask2former_internimage_b_kitti.py \
  --checkpoint work_dirs/mask2former_internimage_b_kitti_balanced_sampler_30/best_mIoU_iter_144000.pth \
  --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ \
  --split test \
  --horizon 30 \
  --additional_config configs/mask2former/mask2former_internimage_b_imgtext_kitti.py \
  --additional_checkpoint work_dirs/mask2former_internimage_b_imgtext_kitti_balanced_sampler_30/best_mIoU_iter_160000.pth \
  --save-dir guidance_failure