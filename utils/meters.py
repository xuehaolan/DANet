import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AveragePrecisionMetric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.num_pos = np.zeros(self.num_classes)
        self.all_cls = []

    def update(self, labels, preds):
        pred_scores = preds.asnumpy()
        label_vecs = labels.asnumpy()
        # pred_probs = np.where(pred_scores > 0, 1 / (1 + np.exp(-pred_scores)),
        #                       np.exp(pred_scores) / (1 + np.exp(pred_scores)))  # more numerical stable
        scores_tflag = np.stack([pred_scores, label_vecs], axis=-1)

        self.all_cls.append(scores_tflag)
        self.num_pos += np.sum(label_vecs, axis=0)

    def get(self):
        ap = np.zeros(self.num_classes)
        all_cls = np.concatenate(self.all_cls, axis=0)
        for c in range(self.num_classes):
            all_cls_c = all_cls[:, c, :]
            arg_sort = np.argsort(all_cls_c[:, 0])[::-1]
            all_cls_c = all_cls_c[arg_sort]
            num_tp = np.cumsum(all_cls_c[:, 1])
            num_fp = np.cumsum(1 - all_cls_c[:, 1])
            rec = num_tp / float(self.num_pos[c])
            prec = num_tp / np.maximum(num_tp + num_fp, np.finfo(np.float64).eps)
            ap[c] = voc_ap(rec, prec)
        return ap.mean()


def ious(pred, gt):
    pred = pred.astype(float)
    gt = gt.astype(float)

    numObj = len(gt)
    gt = np.tile(gt, [len(pred), 1])
    pred = np.repeat(pred, numObj, axis=0)
    bi = np.minimum(pred[:, 2:], gt[:, 2:]) - np.maximum(pred[:, :2], gt[:, :2]) + 1
    area_bi = np.prod(bi.clip(0), axis=1)
    area_bu = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1) + (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1) - area_bi
    return area_bi / area_bu


def corloc(pred_boxes, ground_truth):
    class_corloc = []
    gt_bboxes = ground_truth['gt_bboxes']
    for c, cls in enumerate(ground_truth['class_names']):
        cls_pred_boxes = pred_boxes[pred_boxes[:, 1] == c, :]
        cls_gt_bboxes = gt_bboxes[gt_bboxes[:, 1] == c, :]
        cls_inds = (ground_truth['gt_labels'][:, c] == 1).nonzero()
        cor = 0
        for cidx in cls_inds[0]:
            pred = cls_pred_boxes[cls_pred_boxes[:, 0] == cidx, 2:6]
            if len(pred) > 0:
                gt = cls_gt_bboxes[cls_gt_bboxes[:, 0] == cidx, 2:]
                if max(ious(pred, gt)) >= 0.5:
                    cor += 1
        class_corloc.append(float(cor)/len(cls_inds[0]))
    return sum(class_corloc)/len(class_corloc)


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
