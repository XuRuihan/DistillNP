import sys

sys.path.extend(["./", "../"])


import unittest
import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.datasets.ade import ADE20KDataset
from mmseg.apis import inference_model, init_model, show_result_pyplot
from nas_lib.data.voc import get_voc_train_and_val_loader, get_voc_test_loader
from tqdm import tqdm, trange


class TestDataset(unittest.TestCase):
    # def test_ade(self):
    #     # config_file = "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
    #     config_file = "/disk2/xrh/NAS/NPENASv1/nas_lib/data/ade_config.py"

    #     config = Config.fromfile(config_file)
    #     # if 'init_cfg' in config.model.backbone:
    #     #     config.model.backbone.init_cfg = None
    #     # config.model.pretrained = None
    #     # config.model.train_cfg = None
    #     init_default_scope(config.get('default_scope', 'mmseg'))
    #     cfg = config

    #     dataset = ADE20KDataset(
    #         # ann_file="annotations",
    #         data_root=cfg.data_root,
    #         data_prefix=cfg.train_dataloader.dataset.data_prefix,
    #         pipeline=cfg.train_pipeline,
    #         reduce_zero_label=True,
    #     )

    #     max_label = 0
    #     min_label = 150
    #     # for i in trange(10):
    #     #     data = dataset[i]
    #     #     gt = data["data_samples"].gt_sem_seg.data
    #     #     true_label = gt.masked_select(gt!=255)
    #     #     max_label = max(max_label, true_label.max())
    #     #     min_label = min(min_label, true_label.min())

    #     total_counts = [0] * 150
    #     for i in trange(len(dataset)):
    #         data = dataset[i]
    #         gt = data["data_samples"].gt_sem_seg.data
    #         idx, counts = torch.unique(gt, return_counts=True)
    #         for id, count in zip(idx, counts):
    #             if id.item() == 255:
    #                 continue
    #             total_counts[id.item()] += count.item()
    #     print(total_counts)

    #     # print(max_label, min_label)
    #     # print(len(dataset))
    #     # print(dataset[0])
    #     # for i in trange(10):
    #     #     data = dataset[i]
    #     #     # print(data['img'].shape)
    #     #     # print(data["gt_seg_map"].shape)
    #     #     print(data['inputs'].shape)
    #     #     print(data["data_samples"].gt_sem_seg.data.shape)

    def test_voc(self):
        train_loader, val_loader = get_voc_train_and_val_loader("", batch_size=1)
        test_loader = get_voc_test_loader("", batch_size=1)

        total_counts = [0] * 21
        for images, labels in tqdm(train_loader):
            idx, counts = torch.unique(labels, return_counts=True)
            for id, count in zip(idx, counts):
                if id.item() == 255:
                    continue
                total_counts[id.item()] += count.item()
        print(total_counts)
        # max_ratio, min_ratio = 0, 10
        # for images, labels in tqdm(train_loader):
        #     max_ratio = max(max_ratio, images.shape[2] / images.shape[3])
        #     min_ratio = min(min_ratio, images.shape[2] / images.shape[3])
        # print(max_ratio, min_ratio)
            # if images.shape[1] != 3 or images.shape[-2] != 64 or images.shape[-1] != 64 or labels.shape[1] != 1 or labels.shape[-2] != 64 or labels.shape[-2] != 64:
                # print(images.shape, labels.shape)

    # def test_mmseg(self):
    #     config_file = "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
    #     checkpoint_file = (
    #         "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
    #     )

    #     # build the model from a config file and a checkpoint file
    #     model = init_model(config_file, checkpoint_file, device="cuda:0")

    #     # test a single image and show the results
    #     img = "/disk2/xrh/datasets/ade/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"  # or img = mmcv.imread(img), which will only load it once
    #     result = inference_model(model, img)
    #     # visualize the results in a new window
    #     show_result_pyplot(model, img, result, show=False, save_dir="")
    #     # or save the visualization results to image files
    #     # you can change the opacity of the painted segmentation map in (0, 1].
    #     show_result_pyplot(
    #         model, img, result, show=False, out_file="result.jpg", opacity=0.5
    #     )
    #     # # test a video and show the results
    #     # video = mmcv.VideoReader('video.mp4')
    #     # for frame in video:
    #     #     result = inference_model(model, frame)
    #     # show_result_pyplot(model, frame, result, wait_time=1)


if __name__ == "__main__":
    unittest.main()
