# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from mmaction.utils import multi_pipeline_collate_fn as meta_collate
from ..utils.multigrid import ShortCycleSampler
from .samplers import (ClassSpecificDistributedSampler,
                       DistributedInfiniteSampler, DistributedSampler,
                       InfiniteSampler)

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
BLENDINGS = Registry('blending')


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        videos_per_gpu (int): Number of videos on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed
            training. Default: 1.
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
            The argument also has effect in PyTorch>=1.8.0.
            Default: False
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    sample_by_class = getattr(dataset, 'sample_by_class', False)

    short_cycle = kwargs.pop('short_cycle', False)
    multigrid_cfg = kwargs.pop('multigrid_cfg', None)
    crop_size = kwargs.pop('crop_size', 224)

    if dist:
        if sample_by_class:
            dynamic_length = getattr(dataset, 'dynamic_length', True)
            sampler = ClassSpecificDistributedSampler(
                dataset,
                world_size,
                rank,
                dynamic_length=dynamic_length,
                shuffle=shuffle,
                seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
        shuffle = False
        batch_size = videos_per_gpu
        num_workers = workers_per_gpu

        if short_cycle:
            batch_sampler = ShortCycleSampler(sampler, batch_size,
                                              multigrid_cfg, crop_size)
            init_fn = partial(
                worker_init_fn, num_workers=num_workers, rank=rank,
                seed=seed) if seed is not None else None

            if digit_version(torch.__version__) >= digit_version('1.8.0'):
                kwargs['persistent_workers'] = persistent_workers

            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=init_fn,
                **kwargs)
            return data_loader

    else:
        if short_cycle:
            raise NotImplementedError(
                'Short cycle using non-dist is not supported')

        sampler = None
        batch_size = num_gpus * videos_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=videos_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def build_meta_dataloader(dataset,
                          meta_samples_per_gpu,
                          workers_per_gpu,
                          num_gpus=1,
                          dist=True,
                          shuffle=True,
                          seed=None,
                          drop_last=False,
                          pin_memory=True,
                          persistent_workers=False,
                          use_infinite_sampler=False,
                          **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        meta_samples_per_gpu (int): Number of meta samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed
            training. Default: 1.
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
            The argument also has effect in PyTorch>=1.8.0.
            Default: False
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    sample_by_class = getattr(dataset, 'sample_by_class', False)

    short_cycle = kwargs.pop('short_cycle', False)
    multigrid_cfg = kwargs.pop('multigrid_cfg', None)
    crop_size = kwargs.pop('crop_size', 224)

    if dist:
        if sample_by_class:
            dynamic_length = getattr(dataset, 'dynamic_length', True)
            sampler = ClassSpecificDistributedSampler(
                dataset,
                world_size,
                rank,
                dynamic_length=dynamic_length,
                shuffle=shuffle,
                seed=seed)
        elif use_infinite_sampler:
            sampler = DistributedInfiniteSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
        shuffle = False
        batch_size = meta_samples_per_gpu
        num_workers = workers_per_gpu

        if short_cycle:
            batch_sampler = ShortCycleSampler(sampler, batch_size,
                                              multigrid_cfg, crop_size)
            init_fn = partial(
                worker_init_fn, num_workers=num_workers, rank=rank,
                seed=seed) if seed is not None else None

            if digit_version(torch.__version__) >= digit_version('1.8.0'):
                kwargs['persistent_workers'] = persistent_workers

            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=init_fn,
                **kwargs)
            return data_loader

    else:
        if short_cycle:
            raise NotImplementedError(
                'Short cycle using non-dist is not supported')
        if use_infinite_sampler:
            sampler = InfiniteSampler(dataset, shuffle=shuffle, seed=seed)
        else:
            sampler = None
        batch_size = num_gpus * meta_samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(meta_collate, samples_per_gpu=meta_samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle if sampler is None else
        None,  # shuffle is useless when sampler is not None
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def build_meta_test_dataloader(dataset, meta_test_cfg, **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        meta_test_cfg (dict): Config of meta testing.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        tuple[:obj:`Dataloader`]: `support_data_loader`, `query_data_loader`
            and `test_set_data_loader`.
    """
    support_batch_size = meta_test_cfg.support['batch_size']
    query_batch_size = meta_test_cfg.query['batch_size']
    num_support_workers = meta_test_cfg.support.get('num_workers', 0)
    num_query_workers = meta_test_cfg.query.get('num_workers', 0)

    support_data_loader = DataLoader(
        copy.deepcopy(dataset).support(),
        batch_size=support_batch_size,
        num_workers=num_support_workers,
        collate_fn=partial(meta_collate, samples_per_gpu=support_batch_size),
        pin_memory=False,
        shuffle=True,
        drop_last=meta_test_cfg.support.get('drop_last', False),
        **kwargs)
    query_data_loader = DataLoader(
        copy.deepcopy(dataset).query(),
        batch_size=query_batch_size,
        num_workers=num_query_workers,
        collate_fn=partial(meta_collate, samples_per_gpu=query_batch_size),
        pin_memory=False,
        shuffle=False,
        **kwargs)
    # build test set dataloader for fast test
    if meta_test_cfg.get('fast_test', False):
        all_batch_size = meta_test_cfg.test_set.get('batch_size', 16)
        num_all_workers = meta_test_cfg.test_set.get('num_workers', 0)
        test_set_data_loader = DataLoader(
            copy.deepcopy(dataset).test_set(),
            batch_size=all_batch_size,
            num_workers=num_all_workers,
            collate_fn=partial(meta_collate, samples_per_gpu=all_batch_size),
            pin_memory=False,
            shuffle=False,
            **kwargs)
    else:
        test_set_data_loader = None
    return support_data_loader, query_data_loader, test_set_data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
