import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from nas_lib.utils.comm import setup_logger
from nas_lib.utils.utils_darts import (
    AverageMeter,
    top_accuracy,
    mean_iou,
    mean_accuracy,
)
import time
from nas_lib.configs import cifar10_path
from nas_lib.data.cifar10_dataset import (
    get_cifar10_test_loader,
    transforms_cifar10,
    get_cifar10_train_and_val_loader,
)
from nas_lib.data.ade import get_ade20k_train_and_val_loader, get_ade20k_test_loader
from nas_lib.data.voc import get_voc_train_and_val_loader, get_voc_test_loader
import pickle
import copy


def async_macro_model_train(model_data, gpus, save_dir, dataset="cifar10"):
    gpus = gpus * 2
    q = Queue(10)
    manager = multiprocessing.Manager()
    total_data_dict = manager.dict()
    p_producer = Process(target=model_producer, args=(model_data, q, gpus))
    # time.sleep(3)
    p_consumers = [
        Process(
            target=model_consumer,
            args=(q, i, save_dir, total_data_dict, model_data, dataset),
        )
        for i in range(gpus)
    ]

    p_producer.start()
    for p in p_consumers:
        p.start()

    p_producer.join()
    for p in p_consumers:
        p.join()

    data_dict = {}
    for model_idx, (hash_key, val_err, test_err) in total_data_dict.items():
        data_dict[hash_key] = (val_err, test_err)
    return data_dict


def model_producer(model_data, queue, gpus):
    for idx in model_data:
        queue.put({"idx": idx})
    for _ in range(gpus):
        queue.put("done")


def model_consumer(queue, gpu, save_dir, total_data_dict, model_data, dataset):
    file_name = f"log_gpus_{gpu}"
    logger = setup_logger(
        file_name, save_dir, gpu, log_level="DEBUG", filename=f"{file_name}.txt"
    )
    while True:
        msg = queue.get()
        if msg == "done":
            logger.info(f"thread {gpu} end")
            break
        model_idx = msg["idx"]
        model = model_data[model_idx]
        if dataset == "cifar10":
            hash_key, val_acc, test_acc = model_trainer_cifar10(
                model, gpu, logger, save_dir
            )
            total_data_dict[model_idx] = [hash_key, 100 - val_acc, 100 - test_acc]
        elif dataset == "voc":
            hash_key, val_acc, test_acc = model_trainer_voc(
                model, gpu, logger, save_dir
            )
            total_data_dict[model_idx] = [hash_key, 1 - val_acc, 1 - test_acc]
        elif dataset == "ade20k":
            raise NotImplementedError
            hash_key, val_acc, test_acc = model_trainer_ade20k(
                model, gpu, logger, save_dir
            )
            total_data_dict[model_idx] = [hash_key, 1 - val_acc, 1 - test_acc]


def model_trainer_cifar10(
    model,
    gpu,
    logger,
    save_dir,
    train_epochs=50,
    lr=0.025,
    momentum=0.9,
    weight_decay=3e-4,
    auxiliary=False,
    auxiliary_weight=0,
    cutout=False,
    cutout_length=0,
    drop_path_prob=0.0,
    grad_clip=5,
    train_portion=0.5,
    batch_size=64,
):
    torch.cuda.set_device(gpu % 4)
    hash_key = model.hashkey
    genotype = model.genotype
    train_trans, test_trans = transforms_cifar10(
        cutout=cutout, cutout_length=cutout_length
    )
    model_test_data = get_cifar10_test_loader(
        cifar10_path, transform=test_trans, batch_size=batch_size
    )

    model_train_data, model_val_data = get_cifar10_train_and_val_loader(
        cifar10_path,
        transform=train_trans,
        train_portion=train_portion,
        batch_size=batch_size,
    )
    device = torch.device("cuda:%d" % (gpu % 4))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_epochs, 0.000001, -1
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = []
    val_acc_list = []
    for epoch in range(train_epochs):
        model.train()
        model.drop_path_prob = drop_path_prob * epoch / train_epochs
        running_loss = 0.0
        total_inference_time = 0
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, data in enumerate(model_train_data):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            begin_inference = time.time()
            outputs, outputs_aux = model(input, device)
            loss = criterion(outputs, labels)

            if auxiliary:
                loss_aux = criterion(outputs_aux, labels)
                loss += auxiliary_weight * loss_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step(epoch)

            prec1, prec5 = top_accuracy(outputs, labels, topk=(1, 5))
            n = input.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if i % 100 == 0:
                logger.info(
                    f"Train iter: {i:03d} loss: {losses.avg:.4f} top1: {top1.avg:.2f}% top5: {top5.avg:.2f}%"
                )
            inference_time = time.time() - begin_inference
            total_inference_time += inference_time
            running_loss += loss.item()

        running_loss_avg = running_loss / len(model_train_data)
        duration = time.time() - start
        logger.info(
            f"Epoch: {epoch} training loss: {losses.avg:.6f} "
            f"top1: {top1.avg:.2f}% time duration: {duration:.5f} "
            f"avg inference time: {total_inference_time / (i * 1.0):.5f}"
        )
        loss_list.append(losses.avg)

        # if epoch != 0 and epoch % 5 == 0:
        if True:
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            model.eval()
            total = 0
            with torch.inference_mode():
                for data in model_val_data:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs, _ = model(images, device)
                    loss = criterion(outputs, labels)
                    prec1, prec5 = top_accuracy(outputs, labels, topk=(1, 5))
                    n = images.size(0)
                    losses.update(loss.item(), n)
                    top1.update(prec1.item(), n)
                    top5.update(prec5.item(), n)

                    total += labels.size(0)
            val_acc = top1.avg
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"Valid accuracy: {val_acc:.3f}%")
            val_acc_list.append(val_acc)
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        top1_test = AverageMeter()
        top5_test = AverageMeter()
        for data in model_test_data:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images, device)
            prec1, prec5 = top_accuracy(outputs, labels, topk=(1, 5))
            n = images.size(0)
            top1_test.update(prec1.item(), n)
            top5_test.update(prec5.item(), n)
        test_acc = top1_test.avg
    logger.info(f"Test accuracy: {test_acc:.3f}%")

    model_save_path = save_dir + "/model_pkl/" + hash_key + ".pkl"

    model_save_dict = {
        "genotype": genotype,
        "model": model.to("cpu"),
        "hash_key": hash_key,
        "running_loss_avg": running_loss_avg,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "loss_list": loss_list,
        "val_acc_list": val_acc_list,
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(model_save_dict, f)
    logger.info("#" * 100)

    return hash_key, best_val_acc, test_acc


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, smooth=1e-6):
        batch_size = target.size(0)
        input_flat = input.flatten(1)
        target_flat = target.flatten(1)
        intersection = input_flat * target_flat
        loss = (
            2
            * (intersection.sum(1) + smooth)
            / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        )
        loss = 1 - loss.sum() / batch_size
        return loss


class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, n_classes=150, weights=None):
        dice = DiceLoss()
        input = torch.sigmoid(input)
        losses = [dice(input[:, i], (target == i).float()) for i in range(n_classes)]
        return sum(loss for loss in losses if loss.item() > 0)


class SegCrossEntropy(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.weight = nn.Parameter(1 / torch.tensor(count), requires_grad=False)

    def forward(self, pred, targ, background=255):
        pred = pred.permute(0, 2, 3, 1)
        targ = targ.permute(0, 2, 3, 1)
        mask = targ != background
        pred_true = pred[mask.expand_as(pred)].reshape(-1, pred.shape[3])
        targ_true = targ[mask].reshape(-1)
        return F.cross_entropy(pred_true, targ_true, self.weight)


def model_trainer_ade20k(
    model,
    gpu,
    logger,
    save_dir,
    train_epochs=50,
    lr=0.01,
    betas=(0.9, 0.99),
    weight_decay=3e-4,
):
    parameters = {
        "auxiliary": False,
        "auxiliary_weight": 0,
        "cutout": False,
        "cutout_length": 0,
        "drop_path_prob": 0.0,
        "grad_clip": 5,
        "train_portion": 0.5,
    }

    auxiliary = parameters["auxiliary"]
    auxiliary_weight = parameters["auxiliary_weight"]
    cutout = parameters["cutout"]
    cutout_length = parameters["cutout_length"]
    drop_path_prob = parameters["drop_path_prob"]
    train_portion = parameters["train_portion"]
    grad_clip = parameters["grad_clip"]
    batch_size = 64

    torch.cuda.set_device(gpu)
    hash_key = model.hashkey
    genotype = model.genotype
    model_test_data = get_ade20k_test_loader(cifar10_path, batch_size=batch_size)

    model_train_data, model_val_data = get_ade20k_train_and_val_loader(
        cifar10_path,
        train_portion=train_portion,
        batch_size=batch_size,
    )
    device = torch.device("cuda:%d" % gpu)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_epochs, 0.000001, -1
    )

    best_val_miou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = []
    val_miou_list = []

    # criterion = MulticlassDiceLoss()
    criterion = SegCrossEntropy(
        count=[
            12193341,
            9028743,
            6656214,
            3663520,
            4222514,
            2431602,
            2465817,
            1914317,
            1767912,
            1265560,
            1390130,
            943939,
            1742748,
            1098604,
            901898,
            987625,
            1494646,
            899286,
            840676,
            877102,
            864075,
            631333,
            619549,
            483137,
            625706,
            699029,
            719618,
            411366,
            261129,
            558366,
            317017,
            383848,
            278271,
            330766,
            352495,
            265812,
            202564,
            178741,
            222856,
            226554,
            231416,
            196972,
            206822,
            177162,
            156319,
            176521,
            183361,
            166920,
            463647,
            149604,
            130485,
            140316,
            139720,
            122235,
            93828,
            174387,
            179030,
            168256,
            120975,
            180318,
            227369,
            155095,
            179504,
            81049,
            122732,
            119923,
            151978,
            151051,
            164419,
            129668,
            88889,
            107944,
            110131,
            91131,
            106622,
            108088,
            86977,
            110595,
            83898,
            90586,
            78242,
            73597,
            49342,
            94537,
            86742,
            62820,
            82298,
            52895,
            79582,
            56143,
            70862,
            50142,
            84733,
            39343,
            80508,
            66112,
            41254,
            43610,
            74548,
            38799,
            53534,
            71937,
            54334,
            44173,
            59127,
            48554,
            57870,
            61739,
            58071,
            72063,
            42979,
            40553,
            41161,
            93975,
            72206,
            33620,
            35497,
            56153,
            37764,
            37683,
            52019,
            41000,
            34291,
            37608,
            37910,
            33411,
            39225,
            31177,
            56687,
            32026,
            50672,
            28084,
            40442,
            27843,
            23810,
            34752,
            23946,
            36861,
            21834,
            17543,
            31248,
            38844,
            29867,
            35777,
            37062,
            33169,
            15257,
            20451,
            19413,
            22029,
        ]
    )
    criterion.to(device)

    for epoch in range(train_epochs):
        model.train()
        model.drop_path_prob = drop_path_prob * epoch / train_epochs
        total_inference_time = 0
        start = time.time()
        losses = AverageMeter()
        mIoUs = AverageMeter()
        mAccs = AverageMeter()
        for i, data in enumerate(model_train_data):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            begin_inference = time.time()
            outputs, outputs_aux = model(input, device)
            loss = criterion(outputs, labels)

            if auxiliary:
                loss_aux = criterion(outputs_aux, labels)
                loss += auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step(epoch)

            miou = mean_iou(outputs, labels)
            macc = mean_accuracy(outputs, labels)
            n = input.size(0)
            losses.update(loss.item(), n)
            mIoUs.update(miou.item(), n)
            mAccs.update(macc.item(), n)
            if i % 100 == 0:
                logger.info(
                    f"Train iter: {i:03d} loss: {losses.avg:.4f} mIoU: {mIoUs.avg:.4f} mAcc: {mAccs.avg:.4f}"
                )
            inference_time = time.time() - begin_inference
            total_inference_time += inference_time

        running_loss_avg = losses.avg
        duration = time.time() - start
        logger.info(
            f"Epoch: {epoch} training loss: {losses.avg:.6f} "
            f"mIoU: {mIoUs.avg:.4f} mAcc: {mAccs.avg:.4f} time duration: {duration:.5f} "
            f"avg inference time: {total_inference_time / (i * 1.0):.5f}"
        )
        loss_list.append(losses.avg)

        # if epoch != 0 and epoch % 5 == 0:
        if True:
            losses = AverageMeter()
            mIoUs = AverageMeter()
            mAccs = AverageMeter()
            model.eval()
            total = 0
            with torch.no_grad():
                for data in model_val_data:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs, _ = model(images, device)
                    loss = criterion(outputs, labels)
                    miou = mean_iou(outputs, labels)
                    macc = mean_accuracy(outputs, labels)
                    n = images.size(0)
                    losses.update(loss.item(), n)
                    mIoUs.update(miou.item(), n)
                    mAccs.update(macc.item(), n)

                    total += labels.size(0)
            val_miou = mIoUs.avg
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"Valid mIoU: {val_miou:.4f} mAcc: {mAccs.avg:.4f}")
            val_miou_list.append(val_miou)

    model.load_state_dict(best_model_wts)
    model.eval()
    miou_test = AverageMeter()
    macc_test = AverageMeter()
    with torch.no_grad():
        for data in model_test_data:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images, device)
            miou = mean_iou(outputs, labels)
            macc = mean_accuracy(outputs, labels)
            n = images.size(0)
            miou_test.update(miou.item(), n)
            macc_test.update(macc.item(), n)
        test_miou = miou_test.avg
    logger.info(f"Test mIoU: {test_miou:.4f} mAcc: {macc_test.avg:.4f}")

    model_save_path = save_dir + "/model_pkl/" + hash_key + ".pkl"

    model_save_dict = {
        "genotype": genotype,
        "model": model.to("cpu"),
        "hash_key": hash_key,
        "running_loss_avg": running_loss_avg,
        "val_miou": val_miou,
        "test_miou": test_miou,
        "best_val_miou": best_val_miou,
        "loss_list": loss_list,
        "val_miou_list": val_miou_list,
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(model_save_dict, f)
    logger.info("#" * 100)

    return hash_key, best_val_miou, test_miou


def model_trainer_voc(
    model,
    gpu,
    logger,
    save_dir,
    train_epochs=100,
    lr=0.005,
    momentum=0.9,
    betas=(0.9, 0.99),
    weight_decay=3e-4,
):
    parameters = {
        "auxiliary": False,
        "auxiliary_weight": 0,
        "cutout": False,
        "cutout_length": 0,
        "drop_path_prob": 0.0,
        "grad_clip": 5,
        "train_portion": 0.5,
    }

    auxiliary = parameters["auxiliary"]
    auxiliary_weight = parameters["auxiliary_weight"]
    cutout = parameters["cutout"]
    cutout_length = parameters["cutout_length"]
    drop_path_prob = parameters["drop_path_prob"]
    train_portion = parameters["train_portion"]
    grad_clip = parameters["grad_clip"]
    batch_size = 64

    torch.cuda.set_device(gpu)
    hash_key = model.hashkey
    genotype = model.genotype
    model_test_data = get_voc_test_loader(cifar10_path, batch_size=batch_size)

    model_train_data, model_val_data = get_voc_train_and_val_loader(
        cifar10_path,
        train_portion=train_portion,
        batch_size=batch_size,
    )
    device = torch.device("cuda:%d" % gpu)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_epochs, 0.000001, -1
    )

    best_val_miou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = []
    val_miou_list = []

    # criterion = MulticlassDiceLoss()
    criterion = SegCrossEntropy(
        count=[
            3429724,
            76379,
            24337,
            82082,
            62182,
            52025,
            138809,
            123080,
            220404,
            78052,
            74686,
            80127,
            146124,
            87545,
            95823,
            374892,
            50849,
            76214,
            91097,
            139486,
            78257,
        ]
    )
    criterion.to(device)

    for epoch in range(train_epochs):
        model.train()
        model.drop_path_prob = drop_path_prob * epoch / train_epochs
        total_inference_time = 0
        start = time.time()
        losses = AverageMeter()
        mIoUs = AverageMeter()
        mAccs = AverageMeter()
        for i, data in enumerate(model_train_data):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            begin_inference = time.time()
            outputs, outputs_aux = model(input, device)
            loss = criterion(outputs, labels)

            if auxiliary:
                loss_aux = criterion(outputs_aux, labels)
                loss += auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step(epoch)

            miou = mean_iou(outputs, labels)
            macc = mean_accuracy(outputs, labels)
            n = input.size(0)
            losses.update(loss.item(), n)
            mIoUs.update(miou.item(), n)
            mAccs.update(macc.item(), n)
            if i % 100 == 0:
                logger.info(
                    f"Train iter: {i:03d} loss: {losses.avg:.4f} mIoU: {mIoUs.avg:.4f} mAcc: {mAccs.avg:.4f}"
                )
            inference_time = time.time() - begin_inference
            total_inference_time += inference_time

        running_loss_avg = losses.avg
        duration = time.time() - start
        logger.info(
            f"Epoch: {epoch} training loss: {losses.avg:.6f} "
            f"mIoU: {mIoUs.avg:.4f} mAcc: {mAccs.avg:.4f} time duration: {duration:.5f} "
            f"avg inference time: {total_inference_time / (i * 1.0):.5f}"
        )
        loss_list.append(losses.avg)

        # if epoch != 0 and epoch % 5 == 0:
        if True:
            losses = AverageMeter()
            mIoUs = AverageMeter()
            mAccs = AverageMeter()
            model.eval()
            total = 0
            with torch.no_grad():
                for data in model_val_data:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs, _ = model(images, device)
                    loss = criterion(outputs, labels)
                    miou = mean_iou(outputs, labels)
                    macc = mean_accuracy(outputs, labels)
                    n = images.size(0)
                    losses.update(loss.item(), n)
                    mIoUs.update(miou.item(), n)
                    mAccs.update(macc.item(), n)

                    total += labels.size(0)
            val_miou = mIoUs.avg
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"Valid mIoU: {val_miou:.4f} mAcc: {mAccs.avg:.4f}")
            val_miou_list.append(val_miou)

    model.load_state_dict(best_model_wts)
    model.eval()
    miou_test = AverageMeter()
    macc_test = AverageMeter()
    with torch.no_grad():
        for data in model_test_data:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images, device)
            miou = mean_iou(outputs, labels)
            macc = mean_accuracy(outputs, labels)
            n = images.size(0)
            miou_test.update(miou.item(), n)
            macc_test.update(macc.item(), n)
        test_miou = miou_test.avg
    logger.info(f"Test mIoU: {test_miou:.4f} mAcc: {macc_test.avg:.4f}")

    model_save_path = save_dir + "/model_pkl/" + hash_key + ".pkl"

    model_save_dict = {
        "genotype": genotype,
        "model": model.to("cpu"),
        "hash_key": hash_key,
        "running_loss_avg": running_loss_avg,
        "val_miou": val_miou,
        "test_miou": test_miou,
        "best_val_miou": best_val_miou,
        "loss_list": loss_list,
        "val_miou_list": val_miou_list,
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(model_save_dict, f)
    logger.info("#" * 100)

    return hash_key, best_val_miou, test_miou
