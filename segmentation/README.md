# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).


## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.7 -y
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.5.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Data Preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Evaluation

To evaluate our `InternImage` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/upernet/upernet_internimage_t_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/upernet/upernet_internimage_b_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_b_512_160k_ade20k.py 8 --eval mIoU
```

### Training on ADE20K

To train an `InternImage` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node, run:

```bash
sh dist_train.sh configs/upernet/upernet_internimage_t_512_160k_ade20k.py 8
```

### Manage jobs with Srun

For example, to train `InternImage-XL` with 8 GPU on 1 node, run:

```bash
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/upernet/upernet_internimage_xl_640_160k_ade20k.py
```
