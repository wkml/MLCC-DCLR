import os
import PIL
import numpy as np
import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.vg import VG
from datasets.coco2014 import COCO2014

def get_graph_and_word_file(cfg, labels):

    def get_graph_file(labels):

        graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=float)

        for index in range(labels.shape[0]):
            indexs = np.where(labels[index] == 1)[0]
            for i in indexs:
                for j in indexs:
                    graph[i, j] += 1
        
        np.set_printoptions(suppress=True)

        for i in range(labels.shape[1]):
            graph[i] /= graph[i, i]

        np.nan_to_num(graph)

        return graph

    word_file_path = cfg.dataset.data_vector
    graph_file = get_graph_file(labels)
    word_file = np.load(word_file_path)

    return graph_file, word_file

def get_data_path(cfg):

    if cfg.dataset.name == 'COCO2014':
        train_dir, train_anno, train_label = os.path.join(cfg.dataset.data_dir, 'train2014'), os.path.join(cfg.dataset.data_dir, 'annotations/instances_train2014.json'), './data/coco/train_label_vectors.npy'
        test_dir, test_anno, test_label = os.path.join(cfg.dataset.data_dir, 'val2014'), os.path.join(cfg.dataset.data_dir, 'annotations/instances_val2014.json'), './data/coco/val_label_vectors.npy'

    elif cfg.dataset.name == 'VG':
        train_dir, train_anno, train_label = os.path.join(cfg.dataset.data_dir, 'VG_100K'), cfg.dataset.train_image_list, cfg.dataset.train_label
        test_dir, test_anno, test_label = os.path.join(cfg.dataset.data_dir, 'VG_100K'), cfg.dataset.test_image_list, cfg.dataset.test_label

    elif cfg.dataset.name == 'VOC2007':
        train_dir, train_anno, train_label = os.path.join(cfg.dataset.data_dir, 'JPEGImages'), os.path.join(cfg.dataset.data_dir, 'ImageSets/Main/trainval.txt'), os.path.join(cfg.dataDir, 'Annotations')
        test_dir, test_anno, test_label = os.path.join(cfg.dataset.data_dir, 'JPEGImages'), os.path.join(cfg.dataset.data_dir, 'ImageSets/Main/test.txt'), os.path.join(cfg.dataDir, 'Annotations')

    return train_dir, train_anno, train_label, test_dir, test_anno, test_label

def get_data_loader(cfg):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    randomCropList = [transforms.RandomCrop(Size) for Size in [640, 576, 512, 448, 384, 320]] if cfg.scale_size == 640 else \
                     [transforms.RandomCrop(Size) for Size in [512, 448, 384, 320, 256]]
    train_data_transform = transforms.Compose([transforms.Resize((cfg.scale_size, cfg.scale_size), interpolation=PIL.Image.BICUBIC),
                                               transforms.RandomChoice(randomCropList),
                                               transforms.Resize((cfg.crop_size, cfg.crop_size), interpolation=PIL.Image.BICUBIC),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
    
    test_data_transform = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size), interpolation=PIL.Image.BICUBIC),
                                              transforms.ToTensor(),
                                              normalize])
 
    train_dir, train_anno, train_label, test_dir, test_anno, test_label = get_data_path(cfg)

    if cfg.dataset.name == 'COCO2014':
        print("==> Loading COCO2014...")
        train_set = COCO2014(cfg, 'train', train_dir, train_anno, input_transform=train_data_transform)
        test_set = COCO2014(cfg, 'val', test_dir, test_anno,input_transform=test_data_transform)
    
    elif cfg.dataset.name == 'VG':
        print("==> Loading VG...")
        train_set = VG('train',
                       train_dir, train_anno, train_label,
                       input_transform=train_data_transform, label_proportion=cfg.dataset.prob)
        test_set = VG('val',
                      test_dir, test_anno, test_label,
                      input_transform=test_data_transform)

    train_loader = DataLoader(dataset=train_set,
                              num_workers=cfg.workers,
                              batch_size=cfg.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              # shuffle=True
                              )
    test_loader = DataLoader(dataset=test_set,
                             num_workers=cfg.workers,
                             batch_size=cfg.batch_size,
                             pin_memory=True,
                             drop_last=True,
                             # shuffle=False
                             )

    return train_loader, test_loader
