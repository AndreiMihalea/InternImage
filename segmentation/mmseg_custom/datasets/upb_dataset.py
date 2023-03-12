import bisect
import random
from functools import partial

import cv2
import math
import mmcv
import numpy as np
import torch
from mmcv import Config, print_log
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import digit_version
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset
from mmseg.utils import get_root_logger

from torch.utils.data import DataLoader, DistributedSampler, Sampler
import torch.distributed as dist

LIMIT_1 = 10
LIMIT_2 = 30


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # print(self.dataset.__dir__())
        # print(self.dataset.img_infos)
        targets = torch.tensor([x['ann']['category'] for x in self.dataset.img_infos])
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)
        print(weights)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedWeightedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs)

    return data_loader


@DATASETS.register_module()
class UPBDataset(CustomDataset):
    # CLASSES = ('everything', 'path')
    def __init__(self, split, **kwargs):
        super(UPBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        categories = dict()
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name, euler_pose = line.strip().split(',')
                    euler_pose = float(euler_pose)
                    img_info = dict(filename=img_name)
                    limits = [-float('inf'), -LIMIT_2, -LIMIT_1, LIMIT_1, LIMIT_2, float('inf')]
                    category = bisect.bisect_right(limits, euler_pose) - 1
                    if category in categories:
                        categories[category] += 1
                    else:
                        categories[category] = 1
                    if ann_dir is not None:
                        seg_map = img_name
                        img_info['ann'] = dict(seg_map=seg_map, euler_pose=euler_pose, category=category)
                    img_infos.append(img_info)
                img_infos = sorted(img_infos, key=lambda x: x['filename'])
            print(categories)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            # print(self.prepare_train_img(idx))
            return self.prepare_train_img(idx)


@PIPELINES.register_module()
class LoadCategory(object):

    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        results['category'] = results['ann_info']['category']
        return results

def main():
    dataset_type = 'UPBDataset'
    data_root = '/raid/andreim/nemodrive/upb_data/segmentation/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 512)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='LoadCategory'),
        # dict(type='PerspectiveAug', k=[[0.61, 0, 0.5], [0, 1.09, 0.5], [0, 0, 1]],
        #      m=[[1, 0, 0, 0.00], [0, 1, 0, 1.65], [0, 0, 1, 1.54], [0, 0, 0, 1]]),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg', 'category']),
    ]
    dataset = UPBDataset(
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels',
        split='splits/val_half.txt',
        pipeline=train_pipeline)

    dataloader = build_dataloader(dataset, 4, 4, 4, dist=True, seed=42, drop_last=True)
    categories = dict()
    for i, el in enumerate(dataloader):
        if i % 100 == 0:
            print(i)
            for cat in el['category']:
                print(cat)
        for cat in el['category']:
            if cat in categories:
                categories[cat] += 1
            else:
                categories[cat] = 1
    print(categories)


if __name__ == '__main__':
    main()