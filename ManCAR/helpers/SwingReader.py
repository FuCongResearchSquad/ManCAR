# -*- coding: UTF-8 -*-
"""
SwingReader
===========

Loads sequence recommendation data (train/valid/test) and precomputes Swing item–item
neighbors (two variants: simple and Alibaba/“ali”). Outputs tensors ready for PyTorch
datasets plus a dict `swing_topk_neighbours[item] -> List[(nbr, score)]`.

Main steps
----------
1. Read item & interaction CSVs, map original IDs to internal IDs, build feature dicts.
2. Pad / truncate user histories to MAX_ITEM_SEQ_LEN (left padding).
3. Build Swing neighbors on a chosen subset of data (train / train+valid / only history).
4. Provide quick sanity evaluation of Swing neighbors (hit rate over last K triggers).

"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad
import math

from utils import utils
from utils.constants import *
from collections import defaultdict
import heapq
import itertools
from helpers.BaseReader import BaseReader
import time
import random
from tqdm import tqdm
from functools import lru_cache


class SwingReader(BaseReader):
    """
    Data reader + Swing preprocessor.

    Args (from CLI):
        path:            root folder of processed CSVs.
        dataset:         dataset name (subfolder under `path`).
        sep:             CSV separator.
        swing_alpha(…):  smoothing / weighting hyper-params for Swing (simple / ali).
        swing_topk:      keep top-k neighbors per item.
        swing_m:         when many common users, sample at most M pairs for (u,v).
        swing_min_u:     minimum common users required to keep an item-pair.
        swing_data:      which split(s) to use when building Swing tables:
                         {"train_hist", "train", "train_val"}.
        swing_version:   {"ali","simple"} choose algorithm.

    Attributes set:
        data_dict:  dict split->tensor fields (ITEM_ID, ITEM_SEQ_ID, ITEM_SEQ_LEN)
        item_id2feat/item_feat_num: item feature maps and vocab sizes
        swing_topk_neighbours: dict[int] -> List[(nbr_id, weight)]
        n_users, n_items: cardinalities incl. PAD=0
    """

    @staticmethod
    def parse_data_args(parser):
        """Register SwingReader-specific CLI args."""
        parser.add_argument("--path", type=str, default="datasets/processed", help="Input data dir.")
        parser.add_argument("--dataset", type=str, default="CDs_and_Vinyl", help="Choose a dataset.")
        parser.add_argument("--sep", type=str, default=",", help="sep of csv file.")
        parser.add_argument("--swing_alpha", type=float, default=0.1, help="Smoothing constant for swing scorer.")
        parser.add_argument("--swing_data", type=str, default="train_hist", help="train_hist/train/train_val.")
        parser.add_argument("--swing_alpha1", type=float, default=5.0, help="Ali swing: α1.")
        parser.add_argument("--swing_alpha2", type=float, default=1.0, help="Ali swing: α2.")
        parser.add_argument("--swing_beta", type=float, default=0.3, help="Ali swing: β.")
        parser.add_argument("--swing_topk", type=int, default=100, help="Max neighbors per item.")
        parser.add_argument("--swing_m", type=int, default=10, help="Random sample user pairs when huge.")
        parser.add_argument("--swing_min_u", type=int, default=2, help="Min common user count to keep pair.")
        parser.add_argument("--swing_version", type=str, default='ali', help="ali/simple.")
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset

        # swing hyper-params
        self.swing_topk = args.swing_topk
        self.swing_data = args.swing_data
        self.swing_version = args.swing_version
        self.swing_alpha1 = args.swing_alpha1
        self.swing_alpha2 = args.swing_alpha2
        self.swing_beta = args.swing_beta
        self.swing_m = args.swing_m
        self.swing_min_u = args.swing_min_u

        self.swing_topk_neighbors = None
        self._read_data()

    # ---------------------------------------------------------------------- #
    #                          Swing helpers                                 #
    # ---------------------------------------------------------------------- #
    def topk_per_item(self, counter: dict, k: int):
        """
        Convert undirected pair scores into directed top‑k neighbor lists.

        Parameters
        ----------
        counter : dict[(i,j)] -> float
            Accumulated symmetrical scores where i<j.
        k : int
            Keep only top-k neighbors per item.

        Returns
        -------
        dict[item] -> List[(nbr, score)]
        """
        neigh = defaultdict(list)
        for (i, j), w in counter.items():
            neigh[i].append((w, j))
            neigh[j].append((w, i))

        return {
            itm: [(j, w) for w, j in heapq.nlargest(k, lst)]
            for itm, lst in neigh.items()
        }

    def test_swing(self, swing, data_dict, trigger_num=5, swing_topk_test=20):
        """
        Quick-and-dirty hit-rate check: does any of the last `trigger_num` items'
        neighbors contain the true target?

        Parameters
        ----------
        swing : dict[int] -> List[(nbr, score)]
        data_dict : split tensors
        trigger_num : int
        swing_topk_test : int
        """
        test_seqs = data_dict[ITEM_SEQ_ID][:, -trigger_num:]
        targets   = data_dict[ITEM_ID]
        score = 0.0
        not_in = 0

        for idx in range(test_seqs.size(0)):
            seq    = test_seqs[idx]
            target = int(targets[idx])
            hit = False
            for item in seq.tolist():
                if item not in swing:
                    not_in += 1
                    continue
                neighbors = [nbr for nbr, _ in swing[item][:swing_topk_test]]
                if target in neighbors:
                    score += 1
                    hit = True
                    break
        logging.info(f"For swing_topk_test={swing_topk_test}\n Score: {score / test_seqs.size(0)}")
        logging.info(f"No Swing neighbors count: {not_in}")

    def _gen_item2user(self, df):
        """
        Build item->set(users) from a DataFrame containing [USER_ID, ITEM_SEQ_ID].

        Returns
        -------
        dict[item] -> set(users)
        """
        item2user = defaultdict(set)
        for u, items in df.itertuples(index=False, name=None):
            for i in items:
                item2user[i].add(u)
        return item2user

    def _build_swing_simple(self, data, topk):
        """
        Simple Swing (co-occurrence) scoring:
            w_ij += 1 / (m*(m-1)) for every pair (i,j) in a basket of size m.

        data : iterable of lists (user baskets)
        """
        t0 = time.perf_counter()
        counter = defaultdict(float)
        for items in data:
            m = len(items)
            w = 1.0 / (m * (m - 1))
            for i, j in itertools.combinations(items, 2):
                p = (i, j) if i < j else (j, i)
                counter[p] += w
        logging.info(f"_total_ _build_swing_ time: {time.perf_counter()-t0:.2f}s")
        return self.topk_per_item(counter, topk)

    def _build_swing_ali(self, data, topk, item2user, user2item):
        """
        Alibaba Swing variant (paper/formula with α1, α2, β, M sampling).

        data      : iterable of lists (user baskets)
        item2user : dict[item] -> set(users)
        user2item : dict[user] -> set(items)
        """
        logging.info(f"swing ali parameters: a1={self.swing_alpha1}, a2={self.swing_alpha2}, b={self.swing_beta}, M={self.swing_m}, min_common_user_num={self.swing_min_u}")
        counter = defaultdict(float)

        w3_dict = {item: 1.0 / math.sqrt(len(item2user[item])) for item in item2user.keys()}
        _uv_cache = {}

        def _sim(i, j, alpha1=5.0, alpha2=1.0, beta=0.3, M=10, min_common_user_num=2):
            """
            Compute sim(i,j) with truncated user pairs and cached per-(u,v) weights.
            """
            users = list(item2user[i] & item2user[j])
            if len(users) > M:
                users = random.sample(users, M)

            w3 = w3_dict.get(j)
            if w3 is None or len(users) < min_common_user_num:
                return 0.0

            total = 0.0
            for u, v in itertools.combinations(users, 2):
                key = (u, v) if u < v else (v, u)
                w_uv = _uv_cache.get(key)
                if w_uv is None:
                    Iu = user2item[u]; Iv = user2item[v]
                    inter_uv = len(Iu & Iv)
                    if inter_uv == 0:
                        w_uv = 0.0
                    else:
                        w1 = 1.0 / ((len(Iu)+alpha1)**beta * (len(Iv)+alpha1)**beta)
                        w2 = 1.0 / (inter_uv + alpha2)
                        w_uv = w1 * w2
                    _uv_cache[key] = w_uv
                total += w_uv * 2
            return total * w3

        for seq in tqdm(data, desc="Building ali swing, counting co-occur pairs", unit="user"):
            for i, j in itertools.combinations(seq, 2):
                key_ij = (i, j) if i < j else (j, i)
                w = _sim(i, j,
                         alpha1=self.swing_alpha1, alpha2=self.swing_alpha2,
                         beta=self.swing_beta, M=self.swing_m,
                         min_common_user_num=self.swing_min_u)
                counter[key_ij] += w

        return self.topk_per_item(counter, topk)


    def _read_data(self):
        """
        Read item/interaction CSVs, build tensors and Swing tables.
        """
        
        if self.dataset in ["CDs_and_Vinyl", "Software", "Baby_Products", "Video_Games", "Musical_Instruments"]:
            ORIG_ITEM_ID = "parent_asin"
            ITEM_DATA_COLUMN = [
                "parent_asin", "item_id", "text_emb",
            ]
            ITEM_FEAT_COLUMN = []

        self.item_feat_column = ITEM_FEAT_COLUMN

        logging.info(f'Reading data from "{self.prefix}", dataset = "{self.dataset}"')

        # -------- items --------
        item_df = pd.read_csv(
            os.path.join(self.prefix, self.dataset, f"{self.dataset}.item.csv"),
            sep=self.sep,
            usecols=ITEM_DATA_COLUMN,
        ).reset_index(drop=True)

        orig_item_id2item_id = item_df.set_index(ORIG_ITEM_ID)[ITEM_ID].to_dict()

        item_df.drop(columns=[ORIG_ITEM_ID], inplace=True)
        self.item_id2text_emb = item_df.set_index(ITEM_ID)["text_emb"].to_dict()
        item_df.drop(columns=["text_emb"], inplace=True)

        for feat in ITEM_FEAT_COLUMN:
            if feat.endswith("seq_id"):
                item_df[feat] = item_df[feat].apply(eval)

        self.item_id2feat = item_df.set_index(ITEM_ID).to_dict(orient="index")
        self.item_feat_num = {
            feat: len(set(item_df[feat].dropna().explode())) for feat in ITEM_FEAT_COLUMN
        }
        logging.info(f"item_feat_num: {self.item_feat_num}")

        # -------- interactions --------
        inter_df = dict()
        user_set = set()
        n_entry = 0
        for key in ["train", "valid", "test"]:
            split_df = pd.read_csv(
                os.path.join(self.prefix, self.dataset, f"{self.dataset}.{key}.csv"),
                sep=self.sep,
            ).reset_index(drop=True)
            split_df = utils.eval_list_columns(split_df)
            split_df[ITEM_SEQ] = split_df[ITEM_SEQ].str.split()
            split_df[ITEM_SEQ_LEN] = split_df[ITEM_SEQ].apply(len)
            split_df[ITEM_ID] = split_df[ORIG_ITEM_ID].map(orig_item_id2item_id)
            split_df[ITEM_SEQ_ID] = split_df[ITEM_SEQ].apply(
                lambda x: [orig_item_id2item_id[iid] for iid in x]
            )

            user_set.update(split_df[USER_ID].tolist())
            n_entry += len(split_df)
            inter_df[key] = split_df

        logging.info("Counting dataset statistics...")
        self.n_users = len(user_set) + 1
        self.n_items = item_df[ITEM_ID].max() + 1
        del user_set

        logging.info(f'"# user": {self.n_users-1}, "# item": {self.n_items-1}, "# entry": {n_entry}')

        # -------- tensors --------
        data_dict = {key: dict() for key in ["train", "valid", "test"]}
        for split in ["train", "valid", "test"]:
            split_df = inter_df[split]
            split_df[ITEM_SEQ] = split_df[ITEM_SEQ].apply(lambda x: x[-MAX_ITEM_SEQ_LEN:])

            data_dict[split][ITEM_ID] = torch.from_numpy(split_df[ITEM_ID].values).long()

            item_seq = [torch.from_numpy(np.array(x)).long() for x in split_df[ITEM_SEQ_ID].values]
            left_padded = [pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=0) for seq in item_seq]
            data_dict[split][ITEM_SEQ_ID] = torch.stack(left_padded)

            data_dict[split][ITEM_SEQ_LEN] = torch.from_numpy(split_df[ITEM_SEQ_LEN].values).long()

        self.data_dict = data_dict
        del data_dict

        logging.info(f"size of train: {len(self.data_dict['train'][ITEM_ID])}")
        logging.info(f"size of valid: {len(self.data_dict['valid'][ITEM_ID])}")
        logging.info(f"size of test: {len(self.data_dict['test'][ITEM_ID])}")
        logging.info("Finish reading data.")

        # -------------------- Swing preprocessing --------------------
        train_df = inter_df["train"]

        tmp_df = pd.DataFrame({
            USER_ID:        train_df[USER_ID].to_numpy(),
            ITEM_SEQ_LEN:   train_df[ITEM_SEQ_LEN].to_numpy(),
            ITEM_ID:        train_df[ITEM_ID].to_numpy(),
            ITEM_SEQ_ID:    train_df[ITEM_SEQ_ID].tolist(),
        })

        # best row per user (longest history)
        idx = tmp_df.groupby(USER_ID)[ITEM_SEQ_LEN].idxmax()
        longest = tmp_df.loc[idx, [USER_ID, ITEM_SEQ_ID, ITEM_ID]]

        # choose swing_data source
        if self.swing_data == "train_hist":
            swing_data = longest[ITEM_SEQ_ID]
        elif self.swing_data == "train":
            longest['swing_seq'] = longest.apply(lambda r: r[ITEM_SEQ_ID] + [int(r[ITEM_ID])], axis=1)
            swing_data = longest['swing_seq'].tolist()
        elif self.swing_data == "train_val":
            val_df = inter_df["valid"]
            swing_union_df = pd.concat([train_df, val_df], ignore_index=True)
            useful_columns = [USER_ID, ITEM_SEQ_LEN, ITEM_SEQ_ID, ITEM_ID]
            swing_union_df = swing_union_df[useful_columns]
            union_longest = swing_union_df.loc[idx, [USER_ID, ITEM_SEQ_ID, ITEM_ID]]
            union_longest['swing_seq'] = longest.apply(lambda r: r[ITEM_SEQ_ID] + [int(r[ITEM_ID])], axis=1)
            swing_data = union_longest["swing_seq"].tolist()
        else:
            logging.error(f"Unknown swing_data: {self.swing_data}")
            return

        # build Swing
        if self.swing_version == "ali":
            if self.swing_data == "train_hist":
                tmp = longest[[USER_ID, ITEM_SEQ_ID]]
                item2user = self._gen_item2user(tmp)
                user2item = {u: set(lst) for u, lst in tmp.set_index(USER_ID)[ITEM_SEQ_ID].to_dict().items()}
            elif self.swing_data == "train":
                tmp = longest[[USER_ID, 'swing_seq']]
                item2user = self._gen_item2user(tmp)
                user2item = {u: set(lst) for u, lst in tmp.set_index(USER_ID)['swing_seq'].to_dict().items()}
            else:
                tmp = union_longest[[USER_ID, 'swing_seq']]
                item2user = self._gen_item2user(tmp)
                user2item = {u: set(lst) for u, lst in tmp.set_index(USER_ID)['swing_seq'].to_dict().items()}

            self.swing_topk_neighbours = self._build_swing_ali(swing_data, self.swing_topk, item2user, user2item)
        else:
            self.swing_topk_neighbours = self._build_swing_simple(swing_data, self.swing_topk)

        # quick tests
        logging.info("Test swing tables on train set. ")
        self.test_swing(self.swing_topk_neighbours, self.data_dict['train'], trigger_num=5, swing_topk_test=20)
        self.test_swing(self.swing_topk_neighbours, self.data_dict['train'], trigger_num=5, swing_topk_test=100)
        logging.info("Test swing tables on test set.")
        self.test_swing(self.swing_topk_neighbours, self.data_dict['test'], trigger_num=5, swing_topk_test=20)
        self.test_swing(self.swing_topk_neighbours, self.data_dict['test'], trigger_num=5, swing_topk_test=100)
        logging.info("----------Finish swing preprocessing")