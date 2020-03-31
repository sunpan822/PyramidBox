"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from xml.dom.minidom import parse
import cv2
import numpy as np
import pickle


def readXml(file):
    labels = ["mask", "head", "back", "mid_mask"]
    labels_dict = {}
    for idx, label in enumerate(labels):
        labels_dict[label] = idx
    dom_tree = parse(file)
    rootNode = dom_tree.documentElement

    filename_nodes = rootNode.getElementsByTagName("filename")
    filename = filename_nodes[0].childNodes[0].data
    obejct_nodes = rootNode.getElementsByTagName("object")
    annotation = []
    for object_node in obejct_nodes:
        label_name = object_node.getElementsByTagName("name")[0].childNodes[0].data
        label = labels_dict[label_name]
        xmin = object_node.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = object_node.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = object_node.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = object_node.getElementsByTagName("ymax")[0].childNodes[0].data
        annotation.append([int(xmin), int(ymin), int(xmax), int(ymax), label])
    return filename, annotation


class AnnotationTransform(object):
    """Transforms a widerface annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self):
        pass

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # num = int(target[0])
        # for i in range(num):
        #     xmin = int(target[1 + i * 4])
        #     ymin = int(target[2 + i * 4])
        #     xmax = int(target[3 + i * 4]) + xmin
        #     ymax = int(target[4 + i * 4]) + ymin
        #     if int(target[3 + i * 4]) == 0 or int(target[4 + i * 4]) == 0:
        #         continue
        #
        #     elif int(target[3 + i * 4]) < 0:
        #         tmp = xmin
        #         xmin = xmax
        #         xmax = tmp
        #     elif int(target[4 + i * 4]) < 0:
        #         tmp = ymin
        #         ymin = ymax
        #         ymax = tmp
        for t_pair in target:
            xmin, ymin, xmax, ymax, label = t_pair[0], t_pair[1], t_pair[2], t_pair[3], t_pair[4]
            res.append([xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height), label])
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Detection(data.Dataset):
    """
    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, anno_path, transform=None, target_transform=None,
                 dataset_name='WiderFace'):
        self.anno_path = anno_path
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.annotation = list()
        self.counter = 0
        # for line in open(self.anno_file, 'r'):
        #     filename = line.strip().split()[0]
        #     self.ids.append(filename)
        #     self.annotation.append(line.strip().split()[1:])
        files = os.listdir(self.anno_path)
        anno_file = [file for file in files if '.xml' in file]
        for file in anno_file:
            filename, annotation = readXml(os.path.join(self.anno_path,file))
            self.ids.append(filename)
            self.annotation.append(annotation)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self.annotation[index]
        img = cv2.imread(os.path.join(self.anno_path,self.ids[index]))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
