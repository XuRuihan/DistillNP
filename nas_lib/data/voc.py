import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import numpy as np

from mmseg.datasets.voc import PascalVOCDataset
from mmengine import Config
from mmengine.registry import init_default_scope


class PascalVOC(PascalVOCDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        images, labels = data['inputs'], data["data_samples"].gt_sem_seg.data
        return images.float(), labels


def get_voc_train_and_val_loader(
    root_path, train_portion=0.7, transform=None, batch_size=128
):
    config_file = "/disk2/xrh/NAS/NPENASv1/nas_lib/data/voc_config.py"
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    train_dataset = PascalVOC(**cfg.train_dataloader.dataset)
    val_dataset = PascalVOC(**cfg.val_dataloader.dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


def get_voc_test_loader(root_path, transform=None, batch_size=128):
    config_file = "/disk2/xrh/NAS/NPENASv1/nas_lib/data/voc_config.py"
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    dataset = PascalVOC(**cfg.test_dataloader.dataset)
    testloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return testloader


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
