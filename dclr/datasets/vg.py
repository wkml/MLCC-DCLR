import json
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from dclr.utils.data_utils import get_unk_mask_indices


class VG(data.Dataset):

    def __init__(
        self,
        mode,
        image_dir,
        anno_path,
        labels_path,
        input_transform=None,
        label_proportion=1.0,
    ):

        assert mode in ("train", "val")

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion
        self.testing = True if self.mode == "val" else False
        self.known_label = 0 if self.mode == "val" else 300

        self.img_dir = image_dir
        self.imgName_path = anno_path
        self.img_names = open(self.imgName_path, "r").readlines()

        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels_path = labels_path
        labels = json.load(open(self.labels_path, "r"))
        self.labels = np.zeros((len(self.img_names), 200)).astype(np.int)
        for i in range(len(self.img_names)):
            self.labels[i][labels[self.img_names[i][:-1]]] = 1

        # changedLabels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
            print("Changing label proportion...")
            self.labels[self.labels == 0] = -1
            self.changedLabels = changeLabelProportion(
                self.labels, self.label_proportion
            )

    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        if self.input_transform:
            input = self.input_transform(input)
        partial_labels = self.changedLabels[index]
        full_labels = self.labels[index]

        unk_mask_indices = get_unk_mask_indices(
            input, self.testing, 200, self.known_label
        )

        mask = torch.from_numpy(partial_labels).clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        return index, input, partial_labels, full_labels, mask

    def __len__(self):
        return len(self.img_names)


# =============================================================================
# Help Functions
# =============================================================================
def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label


def getPairIndexes(labels):

    res = []
    for index in range(labels.shape[0]):
        tmp = []
        for i in range(labels.shape[1]):
            if labels[index, i] > 0:
                tmp += np.where(labels[:, i] > 0)[0].tolist()

        tmp = set(tmp)
        tmp.discard(index)
        res.append(np.array(list(tmp)))

    return res
