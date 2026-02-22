# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad

from utils import utils
from utils.constants import *


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument(
            "--path",
            type=str,
            default="datasets/processed",
            help="Input data dir.",
        )
        parser.add_argument(
            "--dataset", type=str, default="CDs_and_Vinyl", help="Choose a dataset."
        )
        parser.add_argument("--sep", type=str, default=",", help="sep of csv file.")
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

    def _count_inter_num(self, all_df):

        df = all_df.copy()
        df[ITEM_SEQ_LEN] = df[ITEM_SEQ].apply(len)
        user_max_history_df = df.loc[df.groupby(USER_ID)[ITEM_SEQ_LEN].idxmax()]
        total_inter_num = (user_max_history_df[ITEM_SEQ_LEN] + 1).sum()

        return total_inter_num

    def _read_data(self):
        if self.dataset in ["CDs_and_Vinyl"]:
            ORIG_ITEM_ID = "parent_asin"
            ITEM_DATA_COLUMN = [
                "parent_asin",
                "item_id",
                # "second_cate_id",
                # "third_cate_id",
                # "store_id",
                "text_emb",
            ]
            ITEM_FEAT_COLUMN = [
                # "second_cate_id",
                # "third_cate_id",
                # "store_id",
            ]

        self.item_feat_column = ITEM_FEAT_COLUMN

        logging.info(
            'Reading data from "{}", dataset = "{}" '.format(self.prefix, self.dataset)
        )
        inter_df = dict()
        for key in ["train", "valid", "test"]:
            inter_df[key] = pd.read_csv(
                os.path.join(self.prefix, self.dataset, f"{self.dataset}.{key}.csv"),
                sep=self.sep,
            ).reset_index(drop=True)
            inter_df[key] = utils.eval_list_columns(inter_df[key])

        logging.info("Counting dataset statistics...")
        key_columns = [USER_ID, ORIG_ITEM_ID, ITEM_SEQ]
        all_df = pd.concat(
            [inter_df[key][key_columns] for key in ["train", "valid", "test"]],
            ignore_index=True,
        )
        all_df[ITEM_SEQ] = all_df[ITEM_SEQ].apply(lambda x: x.split())
        item_df = pd.read_csv(
            os.path.join(self.prefix, self.dataset, f"{self.dataset}.item.csv"),
            sep=self.sep,
            usecols=ITEM_DATA_COLUMN,
        ).reset_index(drop=True)

        self.n_users = all_df[USER_ID].nunique() + 1  # including [PAD]
        self.n_items = item_df[ITEM_ID].max() + 1  # including [PAD]
        inter_num = self._count_inter_num(all_df)
        logging.info(
            '"# user": {}, "# item": {}, "# entry": {}'.format(
                self.n_users - 1, self.n_items - 1, inter_num
            )
        )
        all_df[ITEM_SEQ] = all_df[ITEM_SEQ].apply(lambda x: x[-MAX_ITEM_SEQ_LEN:])
        all_df[ITEM_SEQ_LEN] = all_df[ITEM_SEQ].apply(len)

        # 获取ORIG_ITEM_ID对应的ITEM_ID，得到 dict
        orig_item_id2item_id = item_df.set_index(ORIG_ITEM_ID)[ITEM_ID].to_dict()

        # 将 ORIG_ITEM_ID 从 item_id2feat 中删除
        item_df.drop(columns=[ORIG_ITEM_ID], inplace=True)
        # 提取 item_df 中的 item_id to text_emb 映射
        self.item_id2text_emb = item_df.set_index(ITEM_ID)["text_emb"].to_dict()
        item_df.drop(columns=["text_emb"], inplace=True)
        for feat in ITEM_FEAT_COLUMN:
            if feat.endswith("seq_id"):  # 特征是 list
                item_df[feat] = item_df[feat].apply(eval)  # 将字符串转换为列表
        # 获取 ITEM_ID 到 feat 的映射
        self.item_id2feat = item_df.set_index(ITEM_ID).to_dict(orient="index")
        # 获取每个特征对应的数量，注意有的特征是 int，有的是 list
        self.item_feat_num = {
            feat: len(set(item_df[feat].dropna().explode()))
            for feat in ITEM_FEAT_COLUMN
        }
        logging.info(f"item_feat_num: {self.item_feat_num}")

        # 设置 [PAD] 的特征
        self.item_id2feat[0] = {
            feat: (
                0
                if isinstance(self.item_id2feat[1][feat], int)
                else [0] * len(self.item_id2feat[1][feat])
            )
            for feat in ITEM_FEAT_COLUMN
        }

        # 将 all_df 中的 ORIG_ITEM_ID 映射到 ITEM_ID
        all_df[ITEM_ID] = all_df[ORIG_ITEM_ID].map(orig_item_id2item_id)

        # 将 all_df 中的 ITEM_SEQ 映射到 ITEM_ID
        all_df[ITEM_SEQ_ID] = all_df[ITEM_SEQ].apply(
            lambda x: [orig_item_id2item_id[iid] for iid in x]
        )

        all_data_dict = dict()
        all_data_dict[ITEM_ID] = torch.from_numpy(all_df[ITEM_ID].values).long()
        item_seq = [
            torch.from_numpy(np.array(x)).long() for x in all_df[ITEM_SEQ_ID].values
        ]

        # Left Padding
        left_padded_seqs = [
            pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=0) for seq in item_seq
        ]
        all_data_dict[ITEM_SEQ_ID] = torch.stack(left_padded_seqs)

        all_data_dict[ITEM_SEQ_LEN] = torch.from_numpy(
            all_df[ITEM_SEQ_LEN].values
        ).long()

        self.data_dict = {key: dict() for key in ["train", "valid", "test"]}
        start, end = 0, 0
        for key in ["train", "valid", "test"]:
            end = start + len(inter_df[key])
            for c in all_data_dict:
                self.data_dict[key][c] = all_data_dict[c][start:end]
            start = end

        logging.info(f"size of train: {len(self.data_dict['train'][ITEM_ID])}")
        logging.info(f"size of valid: {len(self.data_dict['valid'][ITEM_ID])}")
        logging.info(f"size of test: {len(self.data_dict['test'][ITEM_ID])}")
        logging.info("Finish reading data.")
