# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import pickle
import shutil
import tempfile
# TODO import test functions from mmcv and delete them from mmaction2
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import build_optimizer, get_dist_info
from mmcv.utils import print_log

from mmaction.utils import MetaTestParallel, label_wrapper

try:
    from mmcv.engine import (collect_results_cpu, collect_results_gpu,
                             multi_gpu_test, single_gpu_test)
    from_mmcv = True
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from_mmcv = False

# if from_mmcv:
#     raise NotImplementedError('meta_test has not implemented in mmcv')

# z scores of different confidence intervals
Z_SCORE = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.98: 2.326,
    0.99: 2.576,
}


def single_gpu_meta_test(model,
                         num_test_tasks,
                         support_dataloader,
                         query_dataloader,
                         test_set_dataloader=None,
                         meta_test_cfg=None,
                         eval_kwargs=None,
                         logger=None,
                         tb_dict=None,
                         confidence_interval=0.95,
                         show_task_results=False):
    """Meta testing on single gpu.

    During meta testing, model might be further fine-tuned or added extra
    parameters. While the tested model need to be restored after meta
    testing since meta testing can be used as the validation in the middle
    of training. To detach model from previous phase, the model will be
    copied and wrapped with :obj:`MetaTestParallel`. And it has full
    independence from the training model and will be discarded after the
    meta testing.

    Args:
        model (:obj:`MMDataParallel` | nn.Module): Model to be meta tested.
        num_test_tasks (int): Number of meta testing tasks.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data and it is used to fetch support data for each task.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of query
            data and it is used to fetch query data for each task.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of all
            test data and it is used for feature extraction from whole dataset
            to accelerate the testing. Default: None.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
        tb_dict (dict): A dict for tensorboard.
            It has 3 keys: 'writer', 'temperature', 'eval_count'.
            'writer' is a SummaryWriter object in tensorboard,
            'name' is the name that shows in tensorboard,
            'eval_count' is a int that counts the number of evaluation.
        confidence_interval (float): Confidence interval. Default: 0.95.
        show_task_results (bool): Whether to record the eval result of
            each task. Default: False.

    Returns:
        dict: Dict of meta evaluate results, containing `accuracy_mean`
            and `accuracy_std` of all test tasks.
    """
    assert confidence_interval in Z_SCORE.keys()
    # Wrap the model with a :class:`MetaTestParallel`.
    # MetaTestParallel will send data to the same device as model.
    model = MetaTestParallel(copy.deepcopy(model))

    # for the backbone-fixed methods, the features can be pre-computed
    # and saved in dataset to achieve acceleration
    if meta_test_cfg.get('fast_test', False):
        print_log('extracting features from all images.', logger=logger)
        extract_features_for_fast_test(model, support_dataloader,
                                       query_dataloader, test_set_dataloader)
    print_log('start meta testing', logger=logger)

    # prepare for meta test
    model.before_meta_test(meta_test_cfg)

    results_list = []
    prog_bar = mmcv.ProgressBar(num_test_tasks)
    for task_id in range(num_test_tasks):
        # set support and query dataloader to the same task by task id
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        # test a task
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        # evaluate predict result
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels, logger=logger, **eval_kwargs)
        eval_result['task_id'] = task_id
        results_list.append(eval_result)
        prog_bar.update()

    if show_task_results:
        # the result of each task will be logged into logger
        for results in results_list:
            msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
            print_log(msg, logger=logger)

    meta_eval_results = dict()
    # get the average accuracy and std
    for k in results_list[0].keys():
        if k == 'task_id':
            continue
        mean = np.mean([res[k] for res in results_list])
        std = np.std([res[k] for res in results_list])
        std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
        meta_eval_results[f'{k}_mean'] = mean
        meta_eval_results[f'{k}_std'] = std
        if tb_dict is not None:
            tb_name = "Val acc for:" + tb_dict['name']
            tb_dict['writer'].add_scalars(tb_name, {'mean': mean},
                                          tb_dict['eval_count'])
            tb_dict['writer'].flush()
    return meta_eval_results


def multi_gpu_meta_test(model: MMDistributedDataParallel,
                        num_test_tasks,
                        support_dataloader,
                        query_dataloader,
                        test_set_dataloader=None,
                        meta_test_cfg=None,
                        eval_kwargs=None,
                        logger=None,
                        tb_dict=None,
                        confidence_interval=0.95,
                        show_task_results=False):
    """Distributed meta testing on multiple gpus.

    During meta testing, model might be further fine-tuned or added extra
    parameters. While the tested model need to be restored after meta
    testing since meta testing can be used as the validation in the middle
    of training. To detach model from previous phase, the model will be
    copied and wrapped with :obj:`MetaTestParallel`. And it has full
    independence from the training model and will be discarded after the
    meta testing.

    In the distributed situation, the :obj:`MetaTestParallel` on each GPU
    is also independent. The test tasks in few shot leaning usually are very
    small and hardly benefit from distributed acceleration. Thus, in
    distributed meta testing, each task is done in single GPU and each GPU
    is assigned a certain number of tasks. The number of test tasks
    for each GPU is ceil(num_test_tasks / world_size). After all GPUs finish
    their tasks, the results will be aggregated to get the final result.

    Args:
        model (:obj:`MMDistributedDataParallel`): Model to be meta tested.
        num_test_tasks (int): Number of meta testing tasks.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            all test data. Default: None.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | None): Logger used for printing
            related information during evaluation. Default: None.
        tb_dict (dict): A dict for tensorboard.
            It has 3 keys: 'writer', 'temperature', 'eval_count'.
            'writer' is a SummaryWriter object in tensorboard,
            'name' is the name that shows in tensorboard,
            'eval_count' is a int that counts the number of evaluation.
        confidence_interval (float): Confidence interval. Default: 0.95.
        show_task_results (bool): Whether to record the eval result of
            each task. Default: False.

    Returns:
        dict | None: Dict of meta evaluate results, containing `accuracy_mean`
            and `accuracy_std` of all test tasks.
    """
    assert confidence_interval in Z_SCORE.keys()
    rank, world_size = get_dist_info()
    # Note that each task is tested on a single GPU. Thus the data and model
    # on different GPU should be independent. :obj:`MMDistributedDataParallel`
    # always automatically synchronizes the grad in different GPUs when doing
    # the loss backward, which can not meet the requirements. Thus we simply
    # copy the module and wrap it with an :obj:`MetaTestParallel`, which will
    # send data to the device model.
    model = MetaTestParallel(copy.deepcopy(model))

    # for the backbone-fixed methods, the features can be pre-computed
    # and saved in dataset to achieve acceleration
    if meta_test_cfg.get('fast_test', False):
        print_log('extracting features from all images.', logger=logger)
        extract_features_for_fast_test(model, support_dataloader,
                                       query_dataloader, test_set_dataloader)
    print_log('start meta testing', logger=logger)
    # prepare for meta test
    model.before_meta_test(meta_test_cfg)

    results_list = []

    # tasks will be evenly distributed on each gpus
    sub_num_test_tasks = num_test_tasks // world_size
    sub_num_test_tasks += 1 if num_test_tasks % world_size != 0 else 0
    if rank == 0:
        prog_bar = mmcv.ProgressBar(num_test_tasks)
    for i in range(sub_num_test_tasks):
        task_id = (i * world_size + rank)
        if task_id >= num_test_tasks:
            continue
        # set support and query dataloader to the same task by task id
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        # test a task
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        # evaluate predict result
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels, logger=logger, **eval_kwargs)
        eval_result['task_id'] = task_id
        results_list.append(eval_result)
        if rank == 0:
            prog_bar.update(world_size)

    collect_results_list = collect_results_cpu(
        results_list, num_test_tasks, tmpdir=None)
    if rank == 0:
        if show_task_results:
            # the result of each task will be logged into logger
            for results in collect_results_list:
                msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
                print_log(msg, logger=logger)

        meta_eval_results = dict()
        print_log(
            f'number of tasks: {len(collect_results_list)}', logger=logger)
        # get the average accuracy and std
        for k in collect_results_list[0].keys():
            if k == 'task_id':
                continue
            mean = np.mean([res[k] for res in collect_results_list])
            std = np.std([res[k] for res in collect_results_list])
            std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
            meta_eval_results[f'{k}_mean'] = mean
            meta_eval_results[f'{k}_std'] = std
            if tb_dict is not None:
                tb_name = "Val acc for:" + tb_dict['name']
                tb_dict['writer'].add_scalars(tb_name, {'mean': mean},
                                              tb_dict['eval_count'])
                tb_dict['writer'].flush()
        return meta_eval_results
    else:
        return None


def extract_features_for_fast_test(model: MetaTestParallel, support_dataloader,
                                   query_dataloader,
                                   test_set_dataloader) -> None:
    """Extracting and saving features for testing acceleration.

    In some methods, the backbone is fixed during meta testing, which results
    in the features from backbone are also fixed for whole dataset. So we can
    calculate the features in advance and save them into `support_dataloader`
    and `query_dataloader`. In this way, the model can skip the feature
    extraction phase during the meta testing, which can obviously accelerate
    the meta testing.

    Args:
        model (:obj:`MetaTestParallel`): Model to be meta tested.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            all test data.
    """
    feats_list, img_metas_list = [], []
    rank, _ = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_set_dataloader.dataset))
    model.eval()
    # traverse the whole dataset and compute the features from backbone
    with torch.no_grad():
        for data in test_set_dataloader:
            img_metas_list.extend(data['img_metas'].data[0])
            # forward in `extract_feat` mode
            feats = model(imgs=data['imgs'], mode='extract_feat')
            feats_list.append(feats)
            if rank == 0:
                prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
        # cat all element in feats_list
        ret_feats = {}
        for k in feats_list[0].keys():
            ret_feats[k] = torch.cat([feats[k] for feats in feats_list], dim=0)

        # feats = torch.cat(feats_list, dim=0)
    # cache the pre-computed features into dataset
    query_dataloader.dataset.cache_feats(ret_feats, img_metas_list)
    support_dataloader.dataset.cache_feats(ret_feats, img_metas_list)
    # query_dataloader.dataset.cache_feats(feats, img_metas_list)
    # support_dataloader.dataset.cache_feats(feats, img_metas_list)


def test_single_task(model: MetaTestParallel, support_dataloader,
                     query_dataloader, meta_test_cfg):
    """Test a single task.

    A task has two stages: handling the support set and predicting the
    query set. In stage one, it currently supports fine-tune based and
    metric based methods. In stage two, it simply forward the query set
    and gather all the results.

    Args:
        model (:obj:`MetaTestParallel`): Model to be meta tested.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        meta_test_cfg (dict): Config for meta testing.

    Returns:
        tuple:

            - results_list (list[np.ndarray]): Predict results.
            - gt_labels (np.ndarray): Ground truth labels.
    """
    # use copy of model for each task
    model = copy.deepcopy(model)
    # get ids of all classes in this task
    task_class_ids = query_dataloader.dataset.get_task_class_ids()

    # forward support set
    model.before_forward_support()
    support_cfg = meta_test_cfg.get('support', dict())
    # methods with fine-tune stage
    if support_cfg.get('train', False):
        optimizer = build_optimizer(model, support_cfg.train['optimizer'])
        num_steps = support_cfg.train['num_steps']
        dataloader_iterator = iter(support_dataloader)
        for i in range(num_steps):
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(support_dataloader)
                data = next(dataloader_iterator)
            # map input labels into range of 0 to numbers of classes-1
            data['label'] = label_wrapper(data['label'], task_class_ids)
            optimizer.zero_grad()
            # forward in `support` mode
            outputs = model.forward(**data, mode='support')
            outputs['loss'].backward()
            optimizer.step()
    # methods without fine-tune stage
    else:
        for i, data in enumerate(support_dataloader):
            # map input labels into range of 0 to numbers of classes-1
            data['label'] = label_wrapper(data['label'], task_class_ids)
            # forward in `support` mode
            model.forward(**data, mode='support')

    # forward query set
    model.before_forward_query()
    results_list, gt_label_list = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(query_dataloader):
            gt_label_list.append(data.pop('label'))
            # forward in `query` mode
            result = model.forward(**data, mode='query')
            results_list.extend(result)
        gt_labels = torch.cat(gt_label_list, dim=0).cpu().numpy()
    # map gt labels into range of 0 to numbers of classes-1.
    gt_labels = label_wrapper(gt_labels, task_class_ids)
    return results_list, gt_labels


if not from_mmcv:

    def single_gpu_test(model, data_loader):  # noqa: F811
        """Test model with a single gpu.

        This method tests model with a single gpu and
        displays test progress bar.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader.

        Returns:
            list: The prediction results.
        """
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            results.extend(result)

            # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size):
                prog_bar.update()
        return results

    def multi_gpu_test(  # noqa: F811
            model, data_loader, tmpdir=None, gpu_collect=True):
        """Test model with multiple gpus.

        This method tests model with multiple gpus and collects the results
        under two different modes: gpu and cpu modes. By setting
        'gpu_collect=True' it encodes results to gpu tensors and use gpu
        communication for results collection. On cpu mode it saves the results
        on different gpus to 'tmpdir' and collects them by the rank 0 worker.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader.
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode. Default: None
            gpu_collect (bool): Option to use either gpu or cpu to collect
                results. Default: True

        Returns:
            list: The prediction results.
        """
        model.eval()
        results = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            results.extend(result)

            if rank == 0:
                # use the first key as main key to calculate the batch size
                batch_size = len(next(iter(data.values())))
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        if gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
        return results

    def collect_results_cpu(result_part, size, tmpdir=None):  # noqa: F811
        """Collect results in cpu mode.

        It saves the results on different gpus to 'tmpdir' and collects
        them by the rank 0 worker.

        Args:
            result_part (list): Results to be collected
            size (int): Result size.
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode. Default: None

        Returns:
            list: Ordered results.
        """
        rank, world_size = get_dist_info()
        # create a tmp dir if it is not specified
        if tmpdir is None:
            MAX_LEN = 512
            # 32 is whitespace
            dir_tensor = torch.full((MAX_LEN, ),
                                    32,
                                    dtype=torch.uint8,
                                    device='cuda')
            if rank == 0:
                mmcv.mkdir_or_exist('.dist_test')
                tmpdir = tempfile.mkdtemp(dir='.dist_test')
                tmpdir = torch.tensor(
                    bytearray(tmpdir.encode()),
                    dtype=torch.uint8,
                    device='cuda')
                dir_tensor[:len(tmpdir)] = tmpdir
            dist.broadcast(dir_tensor, 0)
            tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        else:
            tmpdir = osp.join(tmpdir, '.dist_test')
            mmcv.mkdir_or_exist(tmpdir)
        # synchronizes all processes to make sure tmpdir exist
        dist.barrier()
        # dump the part result to the dir
        mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
        # synchronizes all processes for loading pickle file
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

    def collect_results_gpu(result_part, size):  # noqa: F811
        """Collect results in gpu mode.

        It encodes results to gpu tensors and use gpu communication for results
        collection.

        Args:
            result_part (list): Results to be collected
            size (int): Result size.

        Returns:
            list: Ordered results.
        """
        rank, world_size = get_dist_info()
        # dump result part to tensor with pickle
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)),
            dtype=torch.uint8,
            device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_list.append(
                    pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            return ordered_results
        return None
