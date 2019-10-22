import os
import numpy as np
import xml.etree.ElementTree as ET


def cub():
    IMAGE_LIST_FILE = '../data/CUB_200_2011/images.txt'
    IMAGE_LABEL_FILE = '../data/CUB_200_2011/image_class_labels.txt'
    BOX_FILE = '../data/CUB_200_2011/bounding_boxes.txt'
    SPLIT_FILE = '../data/CUB_200_2011/train_test_split.txt'
    SPLIT_SET_TEMPLATE = '../data/CUB_200_2011/split_{}.txt'
    SPLIT_SET_BOX_TEMPLATE = '../data/CUB_200_2011/split_{}_box.txt'

    with open(IMAGE_LIST_FILE, 'r') as f:
        lines = f.readlines()
        image_names = [x.strip().split(' ')[-1] for x in lines]
        image_names = np.asarray(image_names)

    with open(IMAGE_LABEL_FILE, 'r') as f:
        lines = f.readlines()
        img_labels = [int(x.strip().split(' ')[-1]) for x in lines]
        img_labels = np.asarray(img_labels)

    with open(BOX_FILE, 'r') as f:
        lines = f.readlines()
        all_box = [x for x in lines]

    with open(SPLIT_FILE, 'r') as f:
        lines = f.readlines()
        split_idx = [int(x.strip().split(' ')[-1]) for x in lines]
        split_idx = np.array(split_idx)

    for i in np.unique(split_idx):
        with open(SPLIT_SET_TEMPLATE.format(i), 'w') as f:
            for img_idx in np.where(split_idx == i)[0]:
                f.write('{} {}\n'.format(image_names[img_idx], img_labels[img_idx]-1))

        with open(SPLIT_SET_BOX_TEMPLATE.format(i), 'w') as f:
            for img_idx in np.where(split_idx == i)[0]:
                f.write(all_box[img_idx])


def generate_voc_listfile(set_file, list_file):
    classes = ['__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    IMAGE_FILE_NAME = 'JPEGImages/{}.jpg'
    ANNOTATION_FILE_NAME = '../data/voc2012/Annotations/{}.xml'

    with open(set_file, 'r') as f:
        lines = f.readlines()
        train_list_filenmae = [IMAGE_FILE_NAME.format(x.strip()) for x in lines]

        # get gt_labels
        gt_labels = []
        for x in lines:
            tree = ET.parse(ANNOTATION_FILE_NAME.format(x.strip()))

            objs = tree.findall('object')

            # filter difficult example
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
            num_objs = len(objs)

            gt_classes = np.zeros(num_objs, dtype=np.int32)
            class_to_index = dict(zip(classes, range(len(classes))))

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                cls = class_to_index[obj.find('name').text.lower().strip()]
                gt_classes[ix] = cls - 1
            gt_labels.append(np.unique(gt_classes))

    with open(list_file, 'w') as f:
        for img_name, labels in zip(train_list_filenmae, gt_labels):
            line_str = img_name
            for lbl in labels:
                line_str += ' {}'.format(lbl)
            line_str += '\n'
            f.write(line_str)


def voc():
    TRAINSET_FILE = '../data/voc2012/ImageSets/Main/train.txt'
    VALSET_FILE = '../data/voc2012/ImageSets/Main/val.txt'

    if not os.path.exists('../data/voc2012/list'):
        os.makedirs('../data/voc2012/list')

    TRAIN_LIST_FILE = '../data/voc2012/list/train_list.txt'
    VAL_LIST_FILE = '../data/voc2012/list/val_list.txt'

    generate_voc_listfile(TRAINSET_FILE, TRAIN_LIST_FILE)
    generate_voc_listfile(VALSET_FILE, VAL_LIST_FILE)


if __name__ == '__main__':
    cub()
