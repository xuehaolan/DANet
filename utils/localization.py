import numpy as np
import cv2
from scipy.ndimage import label
from .vistools import norm_atten_map
import torch.nn.functional as F

def get_topk_boxes(logits, cam_map, im_file, input_size, crop_size, topk=(1, ), threshold=0.2, mode='union', gt=None):
    maxk = max(topk)
    maxk_cls = np.argsort(logits)[::-1][:maxk]

    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    maxk_boxes = []
    maxk_maps = []
    for cls in maxk_cls:
        if gt:
            cls = gt
        cam_map_ = cam_map[0, cls, :, :]
        cam_map_ = norm_atten_map(cam_map_)  # normalize cam map
        cam_map_cls = cv2.resize(cam_map_, dsize=(w, h))
        maxk_maps.append(cam_map_cls.copy())

        # segment the foreground
        fg_map = cam_map_cls >= threshold

        if mode == 'max':
            objects, count = label(fg_map)
            max_area = 0
            max_box = None
            for idx in range(1, count+1):
                obj = (objects == idx)
                box = extract_bbox_from_map(obj)
                area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                if area > max_area:
                    max_area = area
                    max_box = box
            if max_box is None:
                max_box = (0, 0, 0, 0)
            max_box = (cls, ) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            maxk_boxes.append((cls, ) + box)
            # maxk_boxes.append((cls, int(box[0] / scale), int(box[1] / scale), int(box[2] / scale), int(box[3] / scale)))
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]

    return result, maxk_maps

def get_topk_boxes_hier(logits3, logits2, logits1, cam_map, parent_map, root_map, im_file, input_size, crop_size, topk=(1, ), threshold=0.2, mode='union', gt=None):
    maxk = max(topk)
    species_cls = np.argsort(logits3)[::-1][:maxk]
    parent_cls = np.argsort(logits2)[::-1][:maxk]
    root_cls = np.argsort(logits1)[::-1][:maxk]

    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    maxk_boxes = []
    maxk_maps = []
    for i in range(maxk):
        if gt:
            species_cls[i] = gt
        cam_map_ = cam_map[0, species_cls[i], :, :]
        parent_map_ = parent_map[0, parent_cls[i], :, :]
        root_map_ = root_map[0, root_cls[i], :, :]
        cam_map_ = cam_map_ + parent_map_ + root_map_
        cam_map_ = norm_atten_map(cam_map_)  # normalize cam map
        cam_map_cls = cv2.resize(cam_map_, dsize=(w, h))
        maxk_maps.append(cam_map_cls.copy())
        # segment the foreground
        fg_map = cam_map_cls >= threshold

        if mode == 'max':
            objects, count = label(fg_map)
            max_area = 0
            max_box = None
            for idx in range(1, count+1):
                obj = (objects == idx)
                box = extract_bbox_from_map(obj)
                area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                if area > max_area:
                    max_area = area
                    max_box = box
            if max_box is None:
                max_box = (0, 0, 0, 0)
            max_box = (species_cls[i], ) + max_box
            maxk_boxes.append(max_box)
        elif mode == 'union':
            box = extract_bbox_from_map(fg_map)
            maxk_boxes.append((species_cls[i], ) + box)
        else:
            raise KeyError('invalid mode! Please set the mode in [\'max\', \'union\']')

    result = [maxk_boxes[:k] for k in topk]

    return result, maxk_maps



def get_masks(logits3, logits2, logits1, cam_map, parent_map, root_map, im_file, input_size, crop_size, topk=(1, ), threshold=0.2, mode='union'):
    maxk = max(topk)
    species_cls = np.argsort(logits3)[::-1][:maxk]
    parent_cls = np.argsort(logits2)[::-1][:maxk]
    root_cls = np.argsort(logits1)[::-1][:maxk]

    # get original image size and scale
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)


    maxk_maps = []
    for i in range(1):
        cam_map_ = cam_map[0, species_cls[i], :, :]
        parent_map_ = parent_map[0, parent_cls[i], :, :]
        root_map_ = root_map[0, root_cls[i], :, :]

        cam_map_cls = [cam_map_, parent_map_, root_map_]
        cam_map_ = (cam_map_ + parent_map_ + root_map_)/3
        # cam_map_ = norm_atten_map(cam_map_)  # normalize cam map
        cam_map_cls.append(cam_map_)
        maxk_maps.append(np.array(cam_map_cls).copy())


    return maxk_maps

def extract_bbox_from_map(boolen_map):
    assert boolen_map.ndim == 2, 'Invalid input shape'
    rows = np.any(boolen_map, axis=1)
    cols = np.any(boolen_map, axis=0)
    if rows.max() == False or cols.max() == False:
        return 0, 0, 0, 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax
