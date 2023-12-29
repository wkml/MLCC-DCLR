import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pycocotools.coco import COCO

from utils.data_utils import get_unk_mask_indices


def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)

def getLabelVector(categories, category_map):
    label = np.zeros(80)
    for c in categories:
        label[category_map[str(c)]-1] = 1.0
    return label

def change_label_proportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label

class COCO2014(data.Dataset):

    def __init__(self, cfg, mode, image_dir, anno_path, input_transform=None):

        assert mode in ('train', 'val')

        self.mode = mode
        self.testing = True if self.mode == 'val' else False
        self.known_label = 0 if self.mode == 'val' else 100
        self.input_transform = input_transform
        self.label_proportion = cfg['dataset']['prob']

        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())
     
        with open(cfg['dataset']['data_category_map'],'r') as load_category:
            self.category_map = json.load(load_category)

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            # target : list, len->(number of objects in the image), 
            # each element is a dict, keys->(segmentation, area, iscrowd, image_id, bbox, category_id, id)
            target = self.coco.loadAnns(ann_ids)
            self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
        self.labels = np.array(self.labels)

        # changedLabels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changed_labels = self.labels
        if self.label_proportion != 1:
            print('Changing label proportion...')
            self.labels[self.labels == 0] = -1
            self.changed_labels = change_label_proportion(self.labels, self.label_proportion)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        partial_labels = self.changed_labels[index]
        full_labels = self.labels[index]

        unk_mask_indices = get_unk_mask_indices(input, self.testing, 80, self.known_label)

        mask = torch.from_numpy(partial_labels).clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long() , -1)
        
        batch = {}
        batch['index'] = index
        batch['input'] = input
        batch['partial_labels'] = partial_labels
        batch['full_labels'] = full_labels
        batch['mask'] = mask
        return batch

        return index, input, partial_labels, full_labels, mask

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    test_data_transform = transforms.Compose([transforms.Resize((160, 160), interpolation=Image.BICUBIC),
                                              transforms.ToTensor()])
    cfg = {
        'dataset': {
            'data_category_map': '/DATA/bvac/personal/projects/DPCAR/datasets/data/coco/category.json',
            'prob': 1.0
        }, 
    }
    datasets = COCO2014(cfg, 'train', '/DATA/bvac/personal/projects/DPCAR/data/coco2014/train2014', '/DATA/bvac/personal/projects/DPCAR/data/coco2014/annotations/instances_train2014.json', input_transform=test_data_transform)
    test_loader = DataLoader(dataset=datasets,
                            num_workers=1,
                            batch_size=16,
                            pin_memory=True,
                            drop_last=True,
                            shuffle=False
                            )
    for batch_index, batch in enumerate(test_loader):
        print(batch)
        break
    # for index, input, partial_labels, full_labels, mask in test_loader:
    #     print(index.shape)
    #     print(input.shape)
    #     print(partial_labels.shape)
    #     print(full_labels.shape)
    #     print(mask.shape)
    #     break