# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
import pickle

from utils import utils
from utils.utils import mem
from utils.constants import *
from utils.metrics import *
from models.BaseModel import BaseModel

# Train/Eval/Predict scripts
class AdaptiveBaseRunnerOracle(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument("--epoch", type=int, default=300, help="Number of epochs.")
        parser.add_argument(
            "--check_epoch",
            type=int,
            default=0,
            help="Check some tensors every check_epoch.",
        )
        parser.add_argument(
            "--test_epoch",
            type=int,
            default=-1,
            help="Print test results every test_epoch (-1 means no print).",
        )
        parser.add_argument(
            "--early_stop",
            type=int,
            default=10,
            help="The number of epochs when valid results drop continuously.",
        )
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
        parser.add_argument(
            "--l2", type=float, default=0, help="Weight decay in optimizer."
        )
        parser.add_argument(
            "--batch_size", type=int, default=2048, help="Batch size during training."
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=2048,
            help="Batch size during testing.",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help="optimizer: SGD, Adam, Adagrad, Adadelta",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=32,
            help="Number of processors when prepare batches in DataLoader",
        )
        parser.add_argument(
            "--pin_memory", type=int, default=1, help="pin_memory in DataLoader"
        )
        parser.add_argument(
            "--topk",
            type=str,
            default="5,10,20,50,100",
            help="The number of items recommended to each user.",
        )
        parser.add_argument(
            "--metric",
            type=str,
            default="NDCG,HR,MRR",
            help="metrics: NDCG, HR, MRR, PRECISION, MAP",
        )
        parser.add_argument(
            "--main_metric",
            type=str,
            default="NDCG@20",
            help="Main metric to determine the best model.",
        )
        parser.add_argument(
            "--clip_grad",
            action="store_true",
            help="Whether to use gradient clip or not.",
        )
        parser.add_argument(
            "--max_norm",
            type=float,
            default=1.0,
            help="Max norm of the gradients for gradient clipping",
        )
        return parser

    @staticmethod
    def evaluate_method(
        predictions: np.ndarray, targets: np.ndarray, topk: list, metrics: list
    ) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param targets: (-1,) shape, the ground-truth item id
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()

        # torch version of topk
        max_topk = max(topk)
        batch_size = 65536
        top_indices_list = []
        for start in range(0, predictions.shape[0], batch_size):
            end = min(start + batch_size, predictions.shape[0])
            batch = predictions[start:end]
            _, topk_idx = torch.topk(batch, k=max_topk, dim=1)
            top_indices_list.append(topk_idx.cpu().numpy())
            # print(top_indices_list)
            del batch, topk_idx
            torch.cuda.empty_cache()
        top_idx = np.vstack(top_indices_list)

        targets = targets.cpu().numpy()
        pos_matrix = top_idx == targets.reshape(-1, 1)
        # print(pos_matrix.shape)
        for m in metrics:
            m_res = eval(m)(pos_matrix, topk)
            for k, res in zip(topk, m_res):
                evaluations[m + "@" + str(k)] = round(res.mean(), 4)

        return evaluations

    def __init__(self, args):
        self.train_models = args.train
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(",")]
        self.metrics = [m.strip().upper() for m in args.metric.split(",")]
        self.main_metric = args.main_metric  # early stop based on main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.clip_grad = args.clip_grad
        self.max_norm = args.max_norm
        self.device = args.device

        self.time = None  # will store [start_time, last_step_time]

        self.log_path = os.path.dirname(args.log_file)  # path to save predictions
        self.save_appendix = args.log_file.split("/")[-1].split(".")[
            0
        ]  # appendix for prediction saving

        print('this is AdaptiveBaseRunner')

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info("Optimizer: " + self.optimizer_name)
        optimizer = eval("torch.optim.{}".format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
        )
        return optimizer

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        model = data_dict["train"].model
        main_metric_results, valid_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict["train"], epoch=epoch + 1)
                if np.isnan(loss):
                    logging.info("Loss is Nan. Stop training at %d." % (epoch + 1))
                    break
                training_time = self._check_time()

                # Observe selected tensors
                if (
                    len(model.check_list) > 0
                    and self.check_epoch > 0
                    and epoch % self.check_epoch == 0
                ):
                    utils.check(model.check_list)

                # Record valid results
                valid_result = self.evaluate(
                    data_dict["valid"], [self.main_topk], self.metrics
                )
                valid_results.append(valid_result)
                main_metric_results.append(valid_result[self.main_metric])
                logging_str = "Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	valid=({})".format(
                    epoch + 1, loss, training_time, utils.format_metric(valid_result)
                )

                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result = self.evaluate(
                        data_dict["test"], self.topk[:1], self.metrics
                    )
                    logging_str += " test=({})".format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += " [{:<.1f} s]".format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or (
                    hasattr(model, "stage") and model.stage == 1
                ):
                    model.save_model()
                    logging_str += " *"
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info(
                        "Early stop at %d based on valid result." % (epoch + 1)
                    )
                    break

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith("y"):
                logging.info(
                    os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
                )
                exit(1)

        # Find the best valid result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(
            os.linesep
            + "Best Iter(valid)={:>5}\t valid=({}) [{:<.1f} s] ".format(
                best_epoch + 1,
                utils.format_metric(valid_results[best_epoch]),
                self.time[1] - self.time[0],
            )
        )
        model.load_model()

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )
        for batch in tqdm(
            dl, leave=False, desc="Epoch {:<3}".format(epoch), ncols=100, mininterval=1
        ):
            batch = utils.batch_to_gpu(batch, model.device)

            model.optimizer.zero_grad()
            out_dict = model(batch, epoch=epoch, stage="train")

            loss = model.loss(out_dict)
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(
            criterion[-self.early_stop :]
        ):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    def predict(self, dataset: BaseModel.Dataset) -> np.ndarray:

        dataset.model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        dataset.model.encode_all_items()  # encode all items before inference
        
        max_topk = max(self.topk)
        all_topk_indices = []
        all_targets = []
        
        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )
        batch_idx = 0
        with torch.no_grad():
            for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc="Predict"):

                batch_gpu = utils.batch_to_gpu(batch, dataset.model.device)
                prediction = dataset.model.inference(batch_gpu)["prediction"]
                
                prediction[:, 0] = -torch.inf
                prediction[:, -1] = -torch.inf
                
                _, topk_indices = torch.topk(prediction, k=max_topk, dim=1)
                
                all_topk_indices.append(topk_indices.cpu().numpy())
                all_targets.append(batch[ITEM_ID].cpu().numpy())
                
                del prediction, batch_gpu, topk_indices
                
                batch_idx += 1
                if batch_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        all_topk_indices = np.vstack(all_topk_indices)
        all_targets = np.concatenate(all_targets)
        
        return all_topk_indices, all_targets
    
    def evaluate(
        self, dataset: BaseModel.Dataset, topks: list, metrics: list
    ) -> Dict[str, float]:

        topk_indices, targets = self.predict(dataset)
        
        # 计算 pos_matrix
        max_topk = topk_indices.shape[1]
        pos_matrix = topk_indices == targets.reshape(-1, 1)
        
        evaluations = dict()
        for m in metrics:
            m_res = eval(m)(pos_matrix, topks)
            for k, res in zip(topks, m_res):
                evaluations[m + "@" + str(k)] = round(res.mean(), 4)
        
        return evaluations

    def predict_test(self, dataset: BaseModel.Dataset, kl_threshold) -> np.ndarray:

        dataset.model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        dataset.model.encode_all_items()  # encode all items before inference
        

        max_topk = max(self.topk)
        all_topk_indices = []
        all_targets = []
        all_stop_t = []
        
        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )
        batch_idx = 0
        with torch.no_grad():
            for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc="Predict"):
                batch_gpu = utils.batch_to_gpu(batch, dataset.model.device)
                infer_out = dataset.model.inference_test(batch_gpu, kl_threshold)
                prediction = infer_out["prediction"]
                stop_t = infer_out["stop_t"]          # [B]
                all_stop_t.append(stop_t.cpu().numpy())
                
                prediction[:, 0] = -torch.inf
                prediction[:, -1] = -torch.inf
                
                _, topk_indices = torch.topk(prediction, k=max_topk, dim=1)
                
                all_topk_indices.append(topk_indices.cpu().numpy())
                all_targets.append(batch[ITEM_ID].cpu().numpy())
                
                del prediction, batch_gpu, topk_indices
                
                batch_idx += 1
                if batch_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        all_topk_indices = np.vstack(all_topk_indices)
        all_targets = np.concatenate(all_targets)
        stop_all = np.concatenate(all_stop_t)
        avg_stop = stop_all.mean()
        
        return all_topk_indices, all_targets, avg_stop, stop_all
    
    def evaluate_test(
        self, dataset: BaseModel.Dataset, topks: list, metrics: list, kl_threshold
    ) -> Dict[str, float]:

        topk_indices, targets, avg_stop, _ = self.predict_test(dataset, kl_threshold)
        
        max_topk = topk_indices.shape[1]
        pos_matrix = topk_indices == targets.reshape(-1, 1)
        
        evaluations = dict()
        for m in metrics:
            m_res = eval(m)(pos_matrix, topks)
            for k, res in zip(topks, m_res):
                evaluations[m + "@" + str(k)] = round(res.mean(), 4)
        
        return evaluations, avg_stop

    @torch.no_grad()
    def evaluate_steps_test(self, dataset, topks, metrics):
        model = dataset.model
        model.eval()
        model.encode_all_items()
        E = model.all_item_embs                      # [N, D]
        max_topk = max(topks)

        all_targets = []
        all_topk_by_step = None  # list[list[np.ndarray]]

        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
        )

        for batch in dl:
            batch_gpu = utils.batch_to_gpu(batch, model.device)
            out = model.forward(batch_gpu, stage="infer")
            seq_output = out["seq_output"]            # [2B, T, D] if noise_factor>0 else [B, T, D]

            B = batch[ITEM_ID].shape[0]
            seq_output = seq_output[:B]              # clean view, shape [B, T, D]
            T = seq_output.shape[1]

            if all_topk_by_step is None:
                all_topk_by_step = [[] for _ in range(T)]

            for t in range(T):
                logits_t = seq_output[:, t, :] @ E.T  # [B, N]
                logits_t[:, 0] = -torch.inf
                logits_t[:, -1] = -torch.inf
                _, topk_t = torch.topk(logits_t, k=max_topk, dim=1)
                all_topk_by_step[t].append(topk_t.cpu().numpy())

            all_targets.append(batch[ITEM_ID].cpu().numpy())
            # break
        targets = np.concatenate(all_targets)
        topk_by_step = [np.vstack(chunks) for chunks in all_topk_by_step]  # each: [n_examples, max_topk]

        # metrics per step
        step_results = []
        for t, topk_idx in enumerate(topk_by_step):
            pos_matrix = topk_idx == targets.reshape(-1, 1)
            evals = {}
            for m in metrics:
                m_res = eval(m)(pos_matrix, topks)
                for k, res in zip(topks, m_res):
                    evals[f"{m}@{k}"] = round(res.mean(), 4)
            step_results.append(evals)

        return step_results

    def _ndcg_at_k_per_sample(self, topk_idx, targets, k=5):
        pos_matrix = topk_idx == targets.reshape(-1, 1)
        return NDCG(pos_matrix, [k])[0]  # per-sample array

    def evaluate_kl_oracle(self, dataset, thresholds, metrics, topks, k=5):
        best_ndcg = None
        best_topk_idx = None
        best_stop_t = None

        for thr in thresholds:
            topk_idx, targets, _, stop_all = self.predict_test(dataset, thr)
            ndcg = self._ndcg_at_k_per_sample(topk_idx, targets, k=k)

            if best_ndcg is None:
                best_ndcg = ndcg
                best_topk_idx = topk_idx.copy()
                best_stop_t = stop_all.copy()
            else:
                mask = ndcg > best_ndcg
                best_ndcg[mask] = ndcg[mask]
                best_topk_idx[mask] = topk_idx[mask]
                best_stop_t[mask] = stop_all[mask]

        result_dict = self._eval_from_topk(best_topk_idx, targets, metrics, topks)
        return result_dict, best_stop_t

    def predict_topk_by_step(self, dataset, topks):
        model = dataset.model
        model.eval()
        model.encode_all_items()
        E = model.all_item_embs
        max_topk = max(topks)

        all_targets = []
        all_topk_by_step = None

        dl = DataLoader(
                dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=dataset.collate_batch,
                pin_memory=self.pin_memory,
            )

        for batch in dl:
            batch_gpu = utils.batch_to_gpu(batch, model.device)
            out = model.forward(batch_gpu, stage="infer")
            seq_output = out["seq_output"]
            B = batch[ITEM_ID].shape[0]
            seq_output = seq_output[:B]
            T = seq_output.shape[1]

            if all_topk_by_step is None:
                all_topk_by_step = [[] for _ in range(T)]

            for t in range(T):
                logits_t = seq_output[:, t, :] @ E.T
                logits_t[:, 0] = -torch.inf
                logits_t[:, -1] = -torch.inf
                _, topk_t = torch.topk(logits_t, k=max_topk, dim=1)
                all_topk_by_step[t].append(topk_t.cpu().numpy())

            all_targets.append(batch[ITEM_ID].cpu().numpy())

        targets = np.concatenate(all_targets)
        topk_by_step = [np.vstack(chunks) for chunks in all_topk_by_step]
        return topk_by_step, targets

    def evaluate_steps_oracle(self, dataset, metrics, topks, k=5):
        topk_by_step, targets = self.predict_topk_by_step(dataset, topks)

        best_ndcg = None
        best_topk_idx = None
        best_step = None

        for t, topk_idx in enumerate(topk_by_step):
            ndcg = self._ndcg_at_k_per_sample(topk_idx, targets, k=k)

            if best_ndcg is None:
                best_ndcg = ndcg
                best_topk_idx = topk_idx.copy()
                best_step = np.full_like(ndcg, t)
            else:
                mask = ndcg > best_ndcg
                best_ndcg[mask] = ndcg[mask]
                best_topk_idx[mask] = topk_idx[mask]
                best_step[mask] = t

        result_dict = self._eval_from_topk(best_topk_idx, targets, metrics, topks)
        return result_dict, best_step

    def _eval_from_topk(self, topk_idx, targets, metrics, topks):
        pos_matrix = topk_idx == targets.reshape(-1, 1)
        evals = {}
        for m in metrics:
            m_res = eval(m)(pos_matrix, topks)
            for k, res in zip(topks, m_res):
                evals[f"{m}@{k}"] = round(res.mean(), 4)
        return evals

    def print_res(self, dataset: BaseModel.Dataset, give_kl_threshold = None) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        res_str = "(" + utils.format_metric(result_dict) + ")"
        if give_kl_threshold is not None:
            result_dict_kl, avg_stop = self.evaluate_test(dataset, self.topk, self.metrics, give_kl_threshold)
            res_str += f'\n kl_threshold: {give_kl_threshold}, avg_stop: {avg_stop}, ({utils.format_metric(result_dict_kl)})'
        else:
            kl_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1]
            for kl_threshold in kl_thresholds:
                result_dict_kl, avg_stop = self.evaluate_test(dataset, self.topk, self.metrics, kl_threshold)
                res_str += f'\n kl_threshold: {kl_threshold}, avg_stop: {avg_stop}, ({utils.format_metric(result_dict_kl)})'

            result_dict_all_steps = self.evaluate_steps_test(dataset, self.topk, self.metrics)
            for idx, result_dict_each_step in enumerate(result_dict_all_steps):
                res_str += f'\n step: {idx}, ({utils.format_metric(result_dict_each_step)})'

            oracle_kl_result, best_stop_t = self.evaluate_kl_oracle(
                dataset, kl_thresholds, self.metrics, self.topk, k=5
            )
            oracle_step_result, best_step = self.evaluate_steps_oracle(
                dataset, self.metrics, self.topk, k=5
            )

            ratio_same = (best_stop_t == best_step).mean()
            abs_diff = np.abs((best_stop_t - best_step)).mean()

            res_str += f"\n oracle_kl_best_by_ndcg5: ({utils.format_metric(oracle_kl_result)})"
            res_str += f"\n oracle_step_best_by_ndcg5: ({utils.format_metric(oracle_step_result)})"
            res_str += f"\n oracle_same_step_ratio: {ratio_same:.4f}, oracle_abs_diff: {abs_diff}"
        
        return res_str
