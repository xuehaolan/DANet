from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2


class dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, datalist_file, root_dir, transform=None, with_path=False, onehot_label=False, num_classes=20, datalist_file_root = None, datalist_file_parent = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file = datalist_file
        self.image_list, label_list_child = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform
        self.onehot_label = onehot_label
        self.num_classes = num_classes

        self.trainFlag = False

        if datalist_file_root is not None:
            self.trainFlag = True
            _, label_list_root = self.read_labeled_image_list(self.root_dir, datalist_file_root)
            _, label_list_parent = self.read_labeled_image_list(self.root_dir, datalist_file_parent)
            self.label_list = [label_list_root, label_list_parent, label_list_child]
        else:
            self.label_list = label_list_child

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        assert os.path.exists(img_name), 'file {} not exits'.format(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if not self.trainFlag:
            if self.onehot_label:
                gt_label = np.zeros(self.num_classes, dtype=np.float32)
                gt_label[self.label_list[idx].astype(int)] = 1
            else:
                gt_label = self.label_list[idx].astype(np.float32)

            if self.with_path:
                return img_name, image, gt_label
            else:
                return image, gt_label
        else:
            gt_label = []
            for i in range(3):
                if self.onehot_label:
                    gt_label_temp = np.zeros(self.num_classes, dtype=np.float32)
                    gt_label_temp[self.label_list[i][idx].astype(int)] = 1
                else:
                    gt_label_temp = self.label_list[i][idx].astype(np.float32)
                gt_label.append(gt_label_temp)

            if self.with_path:
                return img_name, image, gt_label
            else:
                return image, gt_label

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(np.asarray(labels))
        return img_name_list, img_labels


# class PascalVOC(Dataset)


def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id


class dataset_with_mask(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, datalist_file, root_dir, mask_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        mask_name = os.path.join(self.mask_dir, get_name_id(self.image_list[idx])+'.png')
        mask = cv2.imread(mask_name)
        mask[mask == 0] = 255
        mask = mask - 1
        mask[mask == 254] = 255

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, mask, self.label_list[idx]
        else:
            return image, mask, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)
