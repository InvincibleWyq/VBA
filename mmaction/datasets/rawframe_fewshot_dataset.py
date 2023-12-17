# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np

from mmaction.core.evaluation import precision_recall_f1, support
from mmaction.models.losses import accuracy
from .builder import DATASETS
from .rawframe_dataset import RawframeDataset


@DATASETS.register_module()
class RawframeFewShotDataset(RawframeDataset):

    @staticmethod
    def evaluate(results,
                 gt_labels,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            gt_labels (np.ndarray): Ground truth labels.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict | None): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Default: None.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, \
            'dataset testing results should be of the same ' \
            'length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        if metric_options is None:
            metric_options = {'topk': 1}
        topk = metric_options.get('topk', 1)
        thrs = metric_options.get('thrs', 0.0)
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode, thrs=thrs)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
