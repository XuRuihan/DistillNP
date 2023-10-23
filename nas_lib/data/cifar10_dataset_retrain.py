import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
from typing import Any, Callable, Optional, Tuple


class Cifar10Train(torchvision.datasets.CIFAR10):
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
    ]


class Cifar10Val(torchvision.datasets.CIFAR10):
    train_list = [["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"]]


class Cifar10Distill(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        # self.classes, self.class_to_idx
        self.teacher_features = torch.load(
            os.path.join(self.root, "cifar-10-logits.pt")
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        teacher_features = self.teacher_features[index]  # image embeddings
        # teacher_features = self.teacher_features[target]  # text embeddings
        return img, target, teacher_features.to(torch.float32)


def get_cifar10_full_train_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    train_set = torchvision.datasets.CIFAR10(
        root=root_path, train=True, download=False, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader


def get_cifar10_full_train_loader_distill(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    train_set = Cifar10Distill(
        root=root_path, train=True, download=False, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader


def get_cifar10_full_test_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    test_set = torchvision.datasets.CIFAR10(
        root=root_path,
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return test_loader


def get_cifar10_train_and_val_loader(
    root_path, train_portion=0.7, transform=None, batch_size=128
):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    train_set = torchvision.datasets.CIFAR10(
        root=root_path, train=True, download=False, transform=transform
    )
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    )
    val_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    )
    return train_loader, val_loader


def get_cifar10_train_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    trainset = Cifar10Train(
        root=root_path, train=True, download=False, transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return trainloader


def get_cifar10_val_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    valset = Cifar10Val(root=root_path, train=True, download=False, transform=transform)
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return valloader


def get_cifar10_test_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                ),
            ]
        )
    testset = torchvision.datasets.CIFAR10(
        root=root_path, train=False, download=False, transform=transform
    )
    testloader = DataLoader(
        testset,
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


def transforms_cifar10(cutout, cutout_length):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


if __name__ == "__main__":
    trainloader = get_cifar10_train_loader(
        "/home/albert_wei/Disk_A/dataset_train/cifar10/"
    )
    valloader = get_cifar10_val_loader("/home/albert_wei/Disk_A/dataset_train/cifar10/")
    testloader = get_cifar10_test_loader(
        "/home/albert_wei/Disk_A/dataset_train/cifar10/"
    )
    print(len(trainloader))
    print(len(valloader))
    print(testloader)
