from nas_lib.utils.utils_darts import init_nasbench_macro_cifar10, convert_to_genotype
from hashlib import sha256
from nas_lib.eigen.trainer_nasbench_open_darts_async import async_macro_model_train
from nas_lib.models_darts.darts_graph import nasbench2graph2
import numpy as np
from nas_lib.eigen.trainer_narformer import NarFormerPredictorTrainer
import torch
import random
import torch.backends.cudnn as cudnn


def narformer_search_open(
    search_space,
    algo_info,
    logger,
    gpus,
    save_dir,
    verbose=True,
    dataset="cifar10",
    seed=111222333,
):
    """
    regularized evolution
    """
    total_queries = algo_info["total_queries"]
    num_init = algo_info["num_init"]
    k_num = algo_info["k"]
    epochs = algo_info["epochs"]
    batch_size = algo_info["batch_size"]
    lr = algo_info["lr"]
    encode_path = algo_info["encode_path"]
    candidate_nums = algo_info["candidate_nums"]
    macro_graph_dict = {}
    model_keys = []
    init_nasbench_macro_cifar10(save_dir)

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    # Train the first group of NNs
    data_dict = search_space.generate_random_dataset(
        num=num_init, encode_paths=encode_path
    )
    data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in data_dict]
    data_dict_keys = [
        sha256(str(k).encode("utf-8")).hexdigest() for k in data_dict_keys
    ]
    model_keys.extend(data_dict_keys)
    for i, d in enumerate(data_dict):
        macro_graph_dict[data_dict_keys[i]] = list(d)  # [arch, encoding]
    darts_neural_dict = search_space.assemble_cifar10_neural_net(data_dict)
    data = async_macro_model_train(
        model_data=darts_neural_dict, gpus=gpus, save_dir=save_dir, dataset=dataset
    )
    for hashkey, v in data.items():
        if hashkey not in macro_graph_dict:
            raise ValueError("model trained acc key should in macro_graph_dict")
        macro_graph_dict[hashkey].extend(v)
    query = num_init + len(data_dict_keys)

    # Start Evolution
    while query <= total_queries:
        train_data = search_space.assemble_graph(macro_graph_dict, model_keys)
        val_errors = np.array([macro_graph_dict[hashkey][2] for hashkey in model_keys])

        # Gather training NNs.
        arch_data_list = []
        for arch in train_data:
            arch_data_list.append(arch)

        # Gather candidate NNs.
        candidate_graph_dict = {}
        candidates = search_space.get_candidates(
            macro_graph_dict,
            model_keys,
            num=candidate_nums,
            encode_paths=encode_path,
        )
        candidate_dict_keys = [
            convert_to_genotype(d[0], verbose=False) for d in candidates
        ]
        candidate_dict_keys = [
            sha256(str(k).encode("utf-8")).hexdigest() for k in candidate_dict_keys
        ]
        for i, d in enumerate(candidates):
            candidate_graph_dict[candidate_dict_keys[i]] = list(d)
        xcandidates = search_space.assemble_graph(
            candidate_graph_dict, candidate_dict_keys
        )
        candiate_list = []
        for cand in xcandidates:
            candiate_list.append(cand)

        # Train Predictor
        meta_neuralnet = NarFormerPredictorTrainer(
            lr=lr, epochs=epochs, batch_size=batch_size, rate=20.0
        )
        meta_neuralnet.fit(arch_data_list, val_errors, logger=logger)
        pred_train = meta_neuralnet.pred(arch_data_list).cpu().numpy()
        # Predict Candidates
        acc_pred = meta_neuralnet.pred(candiate_list).cpu().numpy()
        candidate_np = acc_pred
        sorted_indices = np.argsort(candidate_np)

        # Select Highest Candidates
        temp_candidate_train_arch = []
        for j in sorted_indices[:k_num]:
            model_keys.append(candidate_dict_keys[j])
            macro_graph_dict[candidate_dict_keys[j]] = candidate_graph_dict[
                candidate_dict_keys[j]
            ]
            temp_candidate_train_arch.append(
                candidate_graph_dict[candidate_dict_keys[j]]
            )

        # Train new group
        darts_candidate_neural_dict = search_space.assemble_cifar10_neural_net(
            temp_candidate_train_arch
        )
        darts_candidate_acc = async_macro_model_train(
            model_data=darts_candidate_neural_dict,
            gpus=gpus,
            save_dir=save_dir,
            dataset=dataset,
        )
        for hashkey, v in darts_candidate_acc.items():
            if hashkey not in macro_graph_dict:
                raise ValueError("model trained acc key should in macro_graph_dict")
            macro_graph_dict[hashkey].extend(v)

        # Step Info
        predictor_error = np.mean(np.abs(pred_train - val_errors)).item()
        val_errors = np.array([macro_graph_dict[hashkey][2] for hashkey in model_keys])

        if verbose:
            top_5_loss = sorted(np.round(val_errors, 5))[: min(5, len(val_errors))]
            logger.info(
                f"Query: {query}  "
                f"Predictor error: {predictor_error:.4f}  "
                f"Current Top 5 NNs' val errors: {top_5_loss}"
            )
        query += len(temp_candidate_train_arch)
    return macro_graph_dict, model_keys
