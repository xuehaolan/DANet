import torch
import cv2
import numpy as np


def accuracy(logits, target, topk=(1,)):
    """
    Compute the top k accuracy of classification results.
    :param target: the ground truth label
    :param topk: tuple or list of the expected k values.
    :return: A list of the accuracy values. The list has the same lenght with para: topk
    """
    maxk = max(topk)
    batch_size = target.size(0)
    scores = logits

    _, pred = scores.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def locerr(topk_boxes, gt_labels, gt_boxes, topk=(1,)):
    assert len(topk_boxes) == len(topk)
    gt_label = gt_labels[0]
    gt_box = gt_boxes

    topk_rslt = []
    for topk_box in topk_boxes:
        err = 1
        for cls_box in topk_box:
            if cls_box[0] == gt_label and cal_iou(cls_box[1:], gt_box) > 0.5:
                err = 0
                break
        topk_rslt.append(float(err*100.0))

    return topk_rslt

def colocerr(topk_boxes, gt_labels, gt_boxes, topk=(1,)):
    assert len(topk_boxes) == len(topk)
    gt_label = gt_labels[0]
    gt_box = gt_boxes

    topk_rslt = []
    for topk_box in topk_boxes:
        err = 1
        for cls_box in topk_box:
            if cal_iou(cls_box[1:], gt_box) > 0.5:
                err = 0
                break
        topk_rslt.append(float(err*100.0))

    return topk_rslt

def colIoU(topk_boxes, gt_labels, gt_boxes, topk=(1,)):
    assert len(topk_boxes) == len(topk)
    gt_label = gt_labels[0]
    gt_box = gt_boxes

    topk_rslt = []
    Iou = cal_iou(topk_boxes[0][0][1:], gt_box)

    if topk_boxes[0][0][0] != gt_label:
        Iou = 2

    return Iou

def get_locerr_array(pred_boxes, gt_boxes):
    num_imgs = len(gt_boxes)
    iou_val = np.zeros((num_imgs, 5))
    for k in range(5):
        pred_box = pred_boxes[:, 4 * k: 4 * k + 4]
        iou_val[:, k] = cal_iou(pred_box, gt_boxes)
    return iou_val < 0.5


def cal_iou(box1, box2):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    iou_val = i_area / (box1_area + box2_area - i_area)

    return iou_val



from sklearn import metrics


def get_mAP(gt_labels, pred_scores):
    n_classes = np.shape(gt_labels)[1]
    results = []
    for i in range(n_classes):
        res = metrics.average_precision_score(gt_labels[:, i], pred_scores[:, i])
        results.append(res)

    results = map(lambda x: '%.3f' % (x), results)
    cls_map = np.array(map(float, results))
    return cls_map


def get_AUC(gt_labels, pred_scores):
    res = metrics.roc_auc_score(gt_labels, pred_scores)
    return res


def _to_numpy(v):
    v = torch.squeeze(v)
    if torch.is_tensor(v):
        v = v.cpu()
        v = v.numpy()
    elif isinstance(v, torch.autograd.Variable):
        v = v.cpu().data.numpy()

    return v


def get_iou(pred, gt):
    '''
    IoU which is averaged by images
    :param pred:
    :param gt:
    :return:
    '''
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    pred[gt == 255] = 255

    assert pred.shape == gt.shape

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    # max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)
    count = np.zeros((20 + 1,))
    for j in range(20 + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    unique_classes = len(np.unique(gt)) - 1 if 255 in np.unique(gt).tolist() else len(np.unique(gt))
    # unique_classes = len(np.unique(gt))
    Aiou = np.sum(result_class[:]) / float(unique_classes)

    return Aiou


def fast_hist(pred, gt, n=21):
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * pred[k].astype(int) + gt[k], minlength=n ** 2).reshape(n, n)


def get_voc_iou(hist):
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return miou


if __name__ == '__main__':
    import scipy.io as scio
    import cPickle

    # load annotations
    TEST_LABEL_FILE = '../data/CUB_200_2011/list/test_list.txt'
    TEST_BOX_FILE = '../data/CUB_200_2011/list/test_box.txt'
    with open(TEST_LABEL_FILE, 'r') as f:
        gt_labels = [int(x.strip().split(' ')[-1]) for x in f.readlines()]
    gt_labels = np.asarray(gt_labels, dtype=int)
    with open(TEST_BOX_FILE, 'r') as f:
        gt_boxes = [map(float, x.strip().split(' ')[1:]) for x in f.readlines()]
    gt_boxes = [(box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1) for box in gt_boxes]
    gt_boxes = np.asarray(gt_boxes, dtype=float)

    # load prediction
    PRED_PROB_FILE = '../output/pred_prob.pkl'
    PRED_BOX_FILE = '../output/pred_bboxes.mat'
    with open(PRED_PROB_FILE, 'r') as f:
        pred_prob = cPickle.load(f)
    mat_contents = scio.loadmat(PRED_BOX_FILE)
    pred_boxes_1 = mat_contents['predictionResult_bbox1']
    pred_boxes_2 = mat_contents['predictionResult_bbox2']
    pred_boxes_c = mat_contents['predictionResult_bboxCombine']
    pred_boxes_1 = np.delete(pred_boxes_1, [5*k for k in range(5)], axis=1)
    pred_boxes_2 = np.delete(pred_boxes_2, [5*k for k in range(5)], axis=1)
    pred_boxes_c = np.delete(pred_boxes_c, [5*k for k in range(5)], axis=1)

    top5_predidx = np.argsort(pred_prob, axis=1)[:, ::-1][:, :5]
    clserr = (top5_predidx != gt_labels[:, np.newaxis])
    top1_clserr = clserr[:, 0].sum() / float(clserr.shape[0])
    top5_clserr = np.min(clserr, axis=1).sum() / float(clserr.shape[0])

    locerr_1 = get_locerr_array(pred_boxes_1, gt_boxes)
    clsloc_err_1 = np.logical_or(locerr_1, clserr)
    top1_locerr_1 = clsloc_err_1[:, 0].sum() / float(clsloc_err_1.shape[0])
    top5_locerr_1 = np.min(clsloc_err_1, axis=1).sum() / float(clsloc_err_1.shape[0])

    locerr_2 = get_locerr_array(pred_boxes_2, gt_boxes)
    clsloc_err_2 = np.logical_or(locerr_2, clserr)
    top1_locerr_2 = clsloc_err_2[:, 0].sum() / float(clsloc_err_2.shape[0])
    top5_locerr_2 = np.min(clsloc_err_2, axis=1).sum() / float(clsloc_err_2.shape[0])

    locerr_c = get_locerr_array(pred_boxes_c, gt_boxes)
    clsloc_err_c = np.logical_or(locerr_c, clserr)
    top1_locerr_c = clsloc_err_c[:, 0].sum() / float(clsloc_err_c.shape[0])
    top5_locerr_c = np.min(clsloc_err_c, axis=1).sum() / float(clsloc_err_c.shape[0])

    print('\t\t\t top1\t top5')
    print('cls err\t\t {:.2f}\t {:.2f}'.format(top1_clserr*100.0, top5_clserr*100.0))
    print('loc err1\t {:.2f}\t {:.2f}'.format(top1_locerr_1*100.0, top5_locerr_1*100.0))
    print('loc err2\t {:.2f}\t {:.2f}'.format(top1_locerr_2*100.0, top5_locerr_2*100.0))
    print('loc errc\t {:.2f}\t {:.2f}'.format(top1_locerr_c*100.0, top5_locerr_c*100.0))

