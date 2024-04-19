# Results segmentation
## Kitti dataset
### Experiment 1:
Experiment setup:
* Model: DeeplabV3+ 
* Segmentation type: hard seg 
* Segmentation labels: manydepth labels 
* Sampler: no sampler

Per-class results:

| Class | IoU   | Acc   |
|-------|-------|-------|
| rest  | 97.83 | 99.31 |
| path  | 57.40 | 66.18 |

Summary:

| aAcc  | mIou  | mAcc  |
|-------|-------|-------|
| 97.89 | 77.61 | 82.74 |


## Scripts
* `run_all_models.py`: runs different models given in the initial list as the configuration
file path and saves the IoU and Acc per frame and final metrics aswell. Can be used as:
    ```shell
    python run_all_models.py --save-dir res_per_model
    ```
* `save_figs.py`: saves figures for all models that are given in the input lists and which
satisfy some constraints given in the code (like the IoU of one model being higher than
the IoU of all other models). The model results are taken from the files resulted from the
`run_all_models.py` script. Can be used as:
    ```shell
    python save_figs.py --res-dir res_per_model/ --save-dir figures/guidance_failures
    ```
* `run_dataset_soft_threshold.py`: Runs a soft segmentation model with different thresholds
applied to it
    ```shell
    python run_dataset_soft_threshold.py --config configs/mask2former/mask2former_soft_internimage_b_kitti.py --checkpoint work_dirs/mask2former_soft_internimage_b_kitti_balanced_sampler_30_all_losses/best_mIoU_soft_iter_160000.pth --dataset_path /raid/andreim/kitti/data_odometry_color/segmentation/ --split test --horizon 30
    ```