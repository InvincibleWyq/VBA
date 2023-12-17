# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import glob
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         IterBasedRunner, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.runner.hooks import Fp16OptimizerHook
from mmcv.utils import build_from_cfg
from torch.utils.tensorboard import SummaryWriter

from mmaction.apis.test import multi_gpu_test
from mmaction.core import (DistEvalHook, DistMetaTestEvalHook, EvalHook,
                           MetaTestEvalHook, OmniSourceDistSamplerSeedHook,
                           OmniSourceRunner)
from mmaction.datasets import (MetaTestDataset, build_dataloader,
                               build_dataset, build_meta_dataloader,
                               build_meta_test_dataloader)
from mmaction.utils import (PreciseBNHook, build_ddp, build_dp, default_device,
                            get_root_logger)


def init_random_seed(seed=None, device=default_device, distributed=True):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
        distributed (bool): Whether to use distributed training.
            Default: True.
    Returns:
        int: Seed to be used.
    """

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    if seed is None:
        seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    if distributed:
        dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)
    tb_writer = SummaryWriter()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    meta_samples_per_gpu = cfg.data.get('meta_samples_per_gpu', None)
    videos_per_gpu = cfg.data.get('videos_per_gpu', None)
    if (meta_samples_per_gpu is not None) and (videos_per_gpu is not None):
        raise ValueError('meta_samples_per_gpu and videos_per_gpu cannot be '
                         'set at the same time')
    elif meta_samples_per_gpu is not None:
        is_fewshot = True
    elif videos_per_gpu is not None:
        is_fewshot = False
    else:
        raise ValueError('meta_samples_per_gpu or videos_per_gpu must be set')

    use_infinite_sampler = cfg.get('use_infinite_sampler', False)
    if is_fewshot:
        dataloader_setting = dict(
            meta_samples_per_gpu=meta_samples_per_gpu,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            use_infinite_sampler=use_infinite_sampler)
    else:
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    if cfg.omnisource and (not is_fewshot):
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', [1] * len(dataset))
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        if omni_videos_per_gpu is None:
            dataloader_settings = [dataloader_setting] * len(dataset)
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                this_setting = cp.deepcopy(dataloader_setting)
                this_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(this_setting)
        data_loaders = [
            build_dataloader(ds, **setting)
            for ds, setting in zip(dataset, dataloader_settings)
        ]
    elif cfg.omnisource:
        raise NotImplementedError(
            'cfg.omnisource and few-shot can\'t be True at the same time,\
            please change the config settings')
    elif is_fewshot:
        data_loaders = [
            build_meta_dataloader(ds, **dataloader_setting) for ds in dataset
        ]
    else:
        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        model = build_ddp(
            model,
            default_device,
            default_args=dict(
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters))
    else:
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner', None) is not None:
        if use_infinite_sampler and cfg.runner['type'] == 'EpochBasedRunner':
            cfg.runner['type'] = 'InfiniteEpochBasedRunner'
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta,
                max_epochs=cfg.get('total_epochs', None)))
    else:
        if cfg.omnisource:
            runner = OmniSourceRunner(
                model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta,
                max_epochs=cfg.get('total_epochs', None))
        elif is_fewshot:
            runner = IterBasedRunner(
                model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta,
                max_iters=100000)
        else:
            runner = EpochBasedRunner(
                model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta,
                max_epochs=cfg.get('total_epochs', None))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    # multigrid setting
    multigrid_cfg = cfg.get('multigrid', None)
    if multigrid_cfg is not None:
        from mmaction.utils.multigrid import LongShortCycleHook
        multigrid_scheduler = LongShortCycleHook(cfg)
        runner.register_hook(multigrid_scheduler)
        logger.info('Finish register multigrid hook')

        # subbn3d aggregation is HIGH, as it should be done before
        # saving and evaluation
        from mmaction.utils.multigrid import SubBatchNorm3dAggregationHook
        subbn3d_aggre_hook = SubBatchNorm3dAggregationHook()
        runner.register_hook(subbn3d_aggre_hook, priority='VERY_HIGH')
        logger.info('Finish register subbn3daggre hook')

    # precise bn setting
    if cfg.get('precise_bn', False):
        precise_bn_dataset = build_dataset(cfg.data.train)
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=1,  # save memory and time
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        data_loader_precise_bn = build_dataloader(precise_bn_dataset,
                                                  **dataloader_setting)
        precise_bn_hook = PreciseBNHook(data_loader_precise_bn,
                                        **cfg.get('precise_bn'))
        runner.register_hook(precise_bn_hook, priority='HIGHEST')
        logger.info('Finish register precisebn hook')

    if distributed:
        if cfg.omnisource:
            runner.register_hook(OmniSourceDistSamplerSeedHook())
        elif isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    if validate:
        if is_fewshot:
            assert cfg.data.val['type'] == 'MetaTestDataset'
            val_dataset = MetaTestDataset(
                build_dataset(cfg.data.val['dataset']),
                num_episodes=cfg.data.val['num_episodes'],
                num_ways=cfg.data.val['num_ways'],
                num_shots=cfg.data.val['num_shots'],
                num_queries=cfg.data.val['num_queries'],
                episodes_seed=cfg.data.val.get('episodes_seed', None),
                subset=cfg.data.val.get('subset', None),
                train_labels=cfg.data.val.get('train_labels', None),
                val_labels=cfg.data.val.get('val_labels', None),
                test_labels=cfg.data.val.get('test_labels', None))
            meta_test_cfg = cfg.data.val['meta_test_cfg']
            support_data_loader, query_data_loader, all_data_loader = \
                build_meta_test_dataloader(val_dataset, meta_test_cfg)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_cfg['meta_test_cfg'] = meta_test_cfg
            # register meta test hooks
            eval_hook = DistMetaTestEvalHook if distributed \
                else MetaTestEvalHook
            # tb_name = cfg.optimizer['type'] + \
            #     ", lr" + str(cfg.optimizer.get('lr', '')) + \
            #     ", m"+str(cfg.optimizer.get('momentum', '')) + \
            #     ", wd" + str(cfg.optimizer.get('weight_decay', '')) + \
            #     ", query" + str(cfg.data.train.get('num_queries', '')) + \
            #     ", temp" + str(cfg.model.cls_head.get('temperature', ''))
            # split cfg.work_dir by '/' and get the last part
            tb_name = cfg.work_dir.split('/')[-1]
            # tb_name = cfg.optimizer['type'] + \
            #     ", lr" + str(cfg.optimizer.get('lr', '')) + \
            #     ", m"+str(cfg.optimizer.get('momentum', '')) + \
            #     ", wd" + str(cfg.optimizer.get('weight_decay', '')) + \
            #     ", query" + str(cfg.data.train.get('num_queries', '')) + \
            #     ", temp" + str(cfg.model.cls_head.get('temperature', ''))
            # tb_name = cfg.model.cls_head['type'] + " " + \
            #     cfg.model.backbone['type'] + " " + \
            #     cfg.ann_file_train.split('/')[2] + \
            #     str(cfg.data.train.num_ways) + "w" + \
            #     str(cfg.data.train.num_shots) + "s"
            runner.register_hook(
                eval_hook(
                    support_data_loader,
                    query_data_loader,
                    all_data_loader,
                    num_test_tasks=meta_test_cfg['num_episodes'],
                    tb_dict={
                        'writer': tb_writer,
                        'name': tb_name,
                        'eval_count': 0
                    },
                    **eval_cfg),
                priority='LOW')

            # user-defined hooks
            if cfg.get('custom_hooks', None):
                custom_hooks = cfg.custom_hooks
                assert isinstance(custom_hooks, list), f'custom_hooks expect \
                    list type, but got {type(custom_hooks)}'

                for hook_cfg in cfg.custom_hooks:
                    assert isinstance(hook_cfg, dict), f'Each item in custom_\
                        hooks expects dict type, but got {type(hook_cfg)}'

                    hook_cfg = hook_cfg.copy()
                    priority = hook_cfg.pop('priority', 'NORMAL')
                    hook = build_from_cfg(hook_cfg, HOOKS)
                    runner.register_hook(hook, priority=priority)
        else:
            eval_cfg = cfg.get('evaluation', {})
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            dataloader_setting = dict(
                videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
                workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
                persistent_workers=cfg.data.get('persistent_workers', False),
                # cfg.gpus will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                shuffle=False)
            dataloader_setting = dict(dataloader_setting,
                                      **cfg.data.get('val_dataloader', {}))
            val_dataloader = build_dataloader(val_dataset,
                                              **dataloader_setting)
            eval_hook = DistEvalHook(val_dataloader, **eval_cfg) \
                if distributed else EvalHook(val_dataloader, **eval_cfg)
            runner.register_hook(eval_hook, priority='LOW')

    if cfg.resume_from:
        if '*' in cfg.resume_from:
            # cfg.resume_from may have wildcard char '*'
            cfg.resume_from = glob.glob(cfg.resume_from)[0]
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()
    if cfg.omnisource:
        runner_kwargs = dict(train_ratio=train_ratio)
    runner.run(data_loaders, cfg.workflow, **runner_kwargs)

    if distributed:
        dist.barrier()
    time.sleep(5)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
            ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                runner.logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                epoch_ids = [
                    int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
                ]
                best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
            else:
                best_ckpt_path = ckpt_paths[0]
            if best_ckpt_path:
                best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        gpu_collect = cfg.get('evaluation', {}).get('gpu_collect', False)
        tmpdir = cfg.get('evaluation', {}).get('tmpdir',
                                               osp.join(cfg.work_dir, 'tmp'))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))

        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []

        if test['test_last']:
            names.append('last')
            ckpts.append(None)
        if test['test_best'] and best_ckpt_path is not None:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                runner.load_checkpoint(ckpt)

            outputs = multi_gpu_test(runner.model, test_dataloader, tmpdir,
                                     gpu_collect)
            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(cfg.work_dir, f'{name}_pred.pkl')
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect',
                        'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                runner.logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    runner.logger.info(f'{metric_name}: {val:.04f}')

    tb_writer.close()
