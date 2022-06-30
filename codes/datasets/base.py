import os
import torch
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.datasets.custom import CustomDataset
from mmseg.core.evaluation.metrics import f_score
from mmseg.datasets.builder import DATASETS


def confusion_matrices(pred_label,
                       label,
                       num_classes,
                       ignore_index,
                       label_map=dict(),
                       reduce_zero_label=False):
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    tp = torch.zeros((num_classes,), dtype=torch.float64)
    fp = torch.zeros((num_classes,), dtype=torch.float64)
    tn = torch.zeros((num_classes,), dtype=torch.float64)
    fn = torch.zeros((num_classes,), dtype=torch.float64)

    for i in range(num_classes):
        pred_ = (pred_label == i).to(torch.float32)
        label_ = (label == i).to(torch.float32)
        confusion_vector = pred_ / label_

        tp[i] += torch.sum(confusion_vector == 1).item()  # 1/1
        fp[i] += torch.sum(confusion_vector == float('inf')).item()  # 1/0
        tn[i] += torch.sum(torch.isnan(confusion_vector)).item()  # 0/0
        fn[i] += torch.sum(confusion_vector == 0).item()  # 0/1

    return tp, fp, tn, fn


def total_confusion_matrices(results,
                             gt_seg_maps,
                             num_classes,
                             ignore_index,
                             label_map=dict(),
                             reduce_zero_label=False):
    # assert len(list(gt_seg_maps)) == len(results)
    total_tp = torch.zeros((num_classes,), dtype=torch.float64)
    total_fp = torch.zeros((num_classes,), dtype=torch.float64)
    total_tn = torch.zeros((num_classes,), dtype=torch.float64)
    total_fn = torch.zeros((num_classes,), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        tp, fp, tn, fn = confusion_matrices(result, gt_seg_map, num_classes, ignore_index, label_map,
                                            reduce_zero_label)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
    return total_tp, total_fp, total_tn, total_fn


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mAcc', 'mFscore', 'mSpecificity']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    total_tp, total_fp, total_tn, total_fn = \
        total_confusion_matrices(results, gt_seg_maps, num_classes, ignore_index, label_map, reduce_zero_label)

    all_acc = (total_tp + total_tn).sum() / (total_fn + total_tp + total_fp + total_tn).sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_tp / (total_fp + total_tp + total_fn)
            ret_metrics['IoU'] = iou
        elif metric == 'mDice':
            dice = 2 * total_tp / (total_fn + 2 * total_tp + total_fp)
            ret_metrics['Dice'] = dice
        elif metric == 'mAcc':
            acc = (total_tp + total_tn) / (total_fn + total_tp + total_fp + total_tn)
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_tp / (total_fp + total_tp)
            recall = total_tp / (total_tp + total_fn)
            f_value = torch.tensor([f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        elif metric == 'mSpecificity':
            specificity = total_tn / (total_tn + total_fp)
            ret_metrics['Specificity'] = specificity

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


@DATASETS.register_module()
class MoreMetricDataset(CustomDataset):

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mAcc', 'mFscore', 'mSpecificity']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        gt_seg_maps = self.get_gt_seg_maps()
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metrics=['mIoU', 'mDice', 'mFscore'],
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
