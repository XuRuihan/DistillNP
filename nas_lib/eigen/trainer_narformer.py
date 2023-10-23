import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import random
from ..models.narformer import NARFormer, NARFormerLoss, tokenizer
from timm.optim import create_optimizer_v2
from ..utils.metric_logger import MetricLogger
from ..utils.utils_solver import get_linear_schedule_with_warmup
import itertools
from tqdm import trange


OPS = {
    "input": 0,
    "none": 1,
    "max_pool_3x3": 2,
    "avg_pool_3x3": 3,
    "skip_connect": 4,
    "sep_conv_3x3": 5,
    "sep_conv_5x5": 6,
    "dil_conv_3x3": 7,
    "dil_conv_5x5": 8,
    "concat": 9,
    "output": 10,
}
NUM_VERTICES = 15


def is_toposort(matrix):
    # a toposort is equivalent to the upper triangle adjacency matrix
    for i in range(len(matrix)):
        for j in range(0, i):
            if matrix[i][j] != 0:
                return False
    return True


def ac_aug_generate(adj, ops, permutations):
    num_vertices = len(ops)
    perms = permutations[num_vertices]
    auged_adjs = [adj]
    auged_opss = [ops]
    adj_array = np.array(adj)
    ops_array = np.array(ops)

    for id, perm in enumerate(perms):
        adj_aug = adj_array[perm][:, perm].astype(int).tolist()
        ops_aug = ops_array[perm].astype(int).tolist()
        if is_toposort(adj_aug) and (
            (adj_aug not in auged_adjs) or (ops_aug not in auged_opss)
        ):
            auged_adjs.append(adj_aug)
            auged_opss.append(ops_aug)
    return auged_adjs[1:], auged_opss[1:]


# Pre-calculate permutation sequences
def permutation_sequences():
    permutations = {}
    for num_vertices in range(15, 16):
        temp = list(range(2, num_vertices - 1))
        temp_list = itertools.permutations(temp)

        perms = []
        for id, perm in enumerate(temp_list):
            # skip the identical permutation
            if id == 0:
                continue
            # Keep the first and the last position fixed
            perm = [0, 1] + list(perm) + [num_vertices - 1]
            perms.append(np.array(perm))

        permutations[num_vertices] = perms
    return permutations


class DARTSDataset(Dataset):
    def __init__(
        self,
        arch_list,
        val_accuracy=None,
        aug_data=False,
    ):
        datas = []
        # permutations = permutation_sequences()
        if val_accuracy is None:
            val_accuracy = [0.0] * len(arch_list)
        for (adj, ops), acc in zip(arch_list, val_accuracy):
            adj = adj[:NUM_VERTICES, :NUM_VERTICES]
            ops = torch.tensor([OPS[op] for op in ops[:NUM_VERTICES]])
            acc = torch.tensor(acc, dtype=torch.float)
            code, rel_pos, c_adj_d, code_depth = tokenizer(ops, adj)
            datas.append(
                [
                    {
                        "index": 0,
                        "adj": adj,
                        "ops": ops,
                        "val_acc": acc,
                        "code": code,
                        "code_rel_pos": rel_pos,
                        "code_adj": c_adj_d,
                        "code_depth": code_depth,
                    }
                ]
            )
            if aug_data:
                # auged_adjs, auged_opss = ac_aug_generate(adj, ops, permutations)
                auged_adjs, auged_opss = [], []
                if len(auged_opss) == 0:
                    datas[-1].append(datas[-1][0])
                    continue
                netcodes = [
                    tokenizer(auged_ops, auged_adj)
                    for auged_ops, auged_adj in zip(auged_opss, auged_adjs)
                ]
                for i in range(len(auged_opss)):
                    datas[-1].append(
                        {  # auged data's key : 1 ~ num_auged
                            "index": i,
                            "adj": auged_adjs[i],
                            "ops": auged_opss[i],
                            "val_acc": acc,
                            "code": netcodes[i][0],
                            "code_rel_pos": netcodes[i][1],
                            "code_adj": netcodes[i][2],
                            "code_depth": netcodes[i][3],
                        }
                    )
        self.data = datas
        self.aug_data = aug_data

    def __getitem__(self, index):
        # data_0, data_1 = random.sample(self.data[index], 2)
        if self.aug_data:
            data_0 = self.data[index][0]
            data_1 = random.sample(self.data[index][1:], 1)[0]
            return data_0, data_1
        else:
            return self.data[index][0]

    def __len__(self):
        return len(self.data)


class FixedLengthBatchSampler(Sampler):
    def __init__(
        self,
        data_source,
        batch_size,
        include_partial=False,
        rng=None,
        maxlen=None,
        length_to_size=None,
    ):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = {0: self.batch_size}
        self.length_map = self.get_length_map()
        self.reset()

    def get_length_map(self):
        """
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.
        """
        # Record the lengths of each example.
        length_map = {}
        # {70:[0, 23, 3332, ...], 110:[3, 421, 555, ...], length:[dataidx_0, dataidx_1, ...]}
        for i in range(len(self.data_source)):
            length = (
                len(self.data_source[i][0]["ops"])
                if len(self.data_source[i]) == 2
                else len(self.data_source[i]["ops"])
            )
            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue
            length_map.setdefault(length, []).append(i)
        return length_map

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start + 1, length + 1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """
        # Shuffle the order.
        for length in self.length_map.keys():
            self.rng.shuffle(self.length_map[length])

        # Initialize state.
        state = {}
        # e.g. {70(length):{'nbatches':3(num_batch), 'surplus':True, 'position':-1}}
        for length, arr in self.length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = len(arr) % batch_size
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []  # [70, 70, 70, 110, ...] length list
        for length, v in state.items():
            order += [length] * v["nbatches"]

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                # if v["surplus"] >= torch.cuda.device_count():
                order += [length]

        self.rng.shuffle(order)

        self.length_map = self.length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1
        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]["position"] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start : start + batch_size]

        self.state[length]["position"] = position
        self.index = index
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self) -> int:
        return len(self.order)


class NarFormerPredictorTrainer:
    def __init__(
        self,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=3000,
        batch_size=128,
        rate=20.0,
        warmup_ratio=0.1,
        aug_data=False,
        out_dim=1,
    ):
        self.device = device
        self.nas_agent = NARFormer(out_dim=out_dim)
        self.nas_agent.to(self.device)

        self.criterion = NARFormerLoss()
        self.criterion.to(self.device)

        self.optimizer = create_optimizer_v2(
            self.nas_agent, opt="adamw", lr=lr, weight_decay=0.01
        )
        nbatches = 1
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_ratio * nbatches * epochs,
            num_training_steps=nbatches * epochs,
            min_ratio=0.01,
        )
        self.batch_size = batch_size
        self.epoch = epochs
        self.rate = rate
        if type(rate) == torch.Tensor:
            self.rate = rate.to(self.device)
        self.aug_data = aug_data

    def fit(self, arch_list, target, logger=None):
        trainset = DARTSDataset(arch_list, target, self.aug_data)
        train_sampler = FixedLengthBatchSampler(
            trainset, self.batch_size, include_partial=True
        )
        train_loader = DataLoader(
            trainset,
            shuffle=(train_sampler is None),
            num_workers=0,
            pin_memory=True,
            batch_sampler=train_sampler,
        )

        meters = MetricLogger(delimiter=" ")
        self.nas_agent.train()
        if logger:
            logger.info("Training predictor")
        for epoch in trange(self.epoch):
            for batch_data in train_loader:
                if self.aug_data:
                    data_0, data_1 = batch_data
                    batch_data = {
                        key: torch.cat([data_0[key], data_1[key]], dim=0)
                        for key in data_0.keys()
                    }
                for k, v in batch_data.items():
                    batch_data[k] = v.to(self.device)
                targ = batch_data["val_acc"]
                pred = (
                    self.nas_agent(
                        batch_data["code"].float(),
                        batch_data["code_rel_pos"].int(),
                        batch_data["code_adj"].float(),
                        batch_data["code_depth"].float(),
                    )
                    * self.rate
                )

                pred = pred.squeeze(1)
                loss = self.criterion(pred, targ)["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                meters.update(loss=loss.item())
        if logger:
            logger.info(f"{meters}")
        return meters.meters["loss"].avg

    @torch.inference_mode()
    def pred(self, arch_list):
        testset = DARTSDataset(arch_list, aug_data=False)
        test_loader = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        pred_list = []
        self.nas_agent.eval()
        for batch_data in test_loader:
            for k, v in batch_data.items():
                batch_data[k] = v.to(self.device)
            pred = (
                self.nas_agent(
                    batch_data["code"],
                    batch_data["code_rel_pos"],
                    batch_data["code_adj"],
                    batch_data["code_depth"],
                )
                * self.rate
            )
            pred = pred.squeeze(1)
            pred_list.append(pred)
        return torch.cat(pred_list, dim=0)
