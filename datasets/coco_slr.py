import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import torch
import torch.utils.data as data
import torchvision.datasets as datasets

from pycocotools.coco import COCO

class COCOSLR(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion

        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())

        with open('/home/horace/workspace/Wikim/MLP-PL-master/data/coco/category.json','r') as load_category:
            self.category_map = json.load(load_category)

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = []
        self.images = []
        self.locas = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            category_ids, bboxs = getCategoryList(target)

            for j in range(len(bboxs)):
                self.images.append(img_id)
                self.locas.append(bboxs[j])
                self.labels.append(getLabelVector(category_ids[j], self.category_map))

        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        img_id = self.images[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')
        input = input.crop(self.locas[index])
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.labels[index]

    def __len__(self):
        return len(self.images)

def getCategoryList(item):
    categories = []
    locas = []
    for t in item:
        x, y, w, h = t['bbox']
        x1, y1, x2, y2 = x, y, x+w, y+h
        categories.append(t['category_id'])
        locas.append([x1, y1, x2, y2])
    return categories, locas


def getLabelVector(category, category_map):
    label = np.zeros(80)
    label[category_map[str(category)]-1] = 1.0
    return label