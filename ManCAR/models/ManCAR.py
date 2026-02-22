
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset

from utils import utils
from helpers.BaseReader import BaseReader
from helpers.SwingReader import SwingReader
from models.BaseModel import AdaptivePromptingModel
from utils.constants import *
import utils.layers as layers
import utils.rearec_layers as rearec_layers
from collections import OrderedDict

import math

import torch
import torch.nn.functional as F

class ManCAR(AdaptivePromptingModel):


    @staticmethod
    def parse_model_args(parser):
        parser = AdaptivePromptingModel.parse_model_args(parser)
        parser.add_argument(
            "--emb_size", type=int, default=256, help="Size of embeddings"
        )
        parser.add_argument(
            "--num_layers", type=int, default=2, help="Number of layers"
        )
        parser.add_argument("--num_heads", type=int, default=2, help="Number of heads")
        parser.add_argument(
            "--inner_size", type=int, default=300, help="Size of inner hidden layers"
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout probability for each deep layer",
        )

        parser.add_argument(
            "--temp_scale",
            type=float,
            default=5,
            help="Epoch steps for progressive learning",
        )
        parser.add_argument(
            "--hidden_act",
            type=str,
            default="gelu",
            help="Activation function of hidden layers",
        )
        parser.add_argument(
            "--rms_norm_eps",
            type=float,
            default=1e-12,
            help="RMS normalization epsilon",
        )
        parser.add_argument(
            "--initializer_range",
            type=float,
            default=0.02,
            help="Initializer range for parameters",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Temperature for softmax in prediction",
        )
        parser.add_argument(
            "--ori_weight",
            type=float,
            default=1.0,
            help="Weight for seq embedding loss",
        )

        parser.add_argument(
            "--swing_weight",
            type=float,
            default=0.0,
            help="Weight for swing loss",
        )

        parser.add_argument(
            "--pl_weight",
            type=float,
            default=1.0,
            help="Weight for swing loss",
        )

        parser.add_argument(
            "--noise_factor",
            type=float,
            default=0.1,
            help="Weight for swing loss",
        )

        parser.add_argument(
            "--cl_weight",
            type=float,
            default=0.0,
            help="Weight for swing loss",
        )

        parser.add_argument(
            "--reason_step", type=int, default=2, help="Reasoning steps"
        )
        return parser

    def __init__(self, args, corpus: SwingReader):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.hidden_size = self.emb_size  # same as emb_size
        self.inner_size = args.inner_size
        self.dropout = args.dropout
        self.hidden_act = args.hidden_act
        self.rms_norm_eps = args.rms_norm_eps
        self.initializer_range = args.initializer_range
        self.temperature = args.temperature
        self.w1 = args.ori_weight
        self.w2 = args.swing_weight

        self.pl_weight = args.pl_weight
        self.cl_weight = args.cl_weight
        self.noise = args.noise_factor
        self.temp_scale = args.temp_scale
        self.reason_step = args.reason_step

        self._define_params(corpus)
        self.apply(self.init_weights)
        
    class Dataset(AdaptivePromptingModel.Dataset):
        def __init__(self, model, corpus, phase: str):
            super().__init__(model, corpus, phase)
            self.swing_neighbors = model.swing_neighbors
            self.trigger_num = model.trigger_num
            self.item_per_trigger = model.item_per_trigger
            self.swing_id_maxlen = model.swing_id_maxlen

        def _get_feed_dict(self, index):
            item_seq, target_item, item_seq_len = (
                self.data[ITEM_SEQ_ID][index],
                self.data[ITEM_ID][index],
                self.data[ITEM_SEQ_LEN][index],
            )

            feed_dict = {
                ITEM_SEQ_ID: item_seq,
                ITEM_ID: target_item,
                ITEM_SEQ_LEN: item_seq_len,
            }
            target_item = target_item.item()
            item_seq = item_seq.tolist()
            feed_dict.update(
                **{
                    feat: torch.tensor(
                        self.item_id2feat[target_item][feat], dtype=torch.long
                    )
                    for feat in self.item_id2feat[target_item]
                },
                **{
                    f"seq_{feat}": torch.tensor(
                        [self.item_id2feat[iid][feat] for iid in item_seq],
                        dtype=torch.long,
                    )
                    for feat in self.item_id2feat[target_item]
                }
            )

            trigger_items = item_seq[-self.trigger_num:]

            prompt_ids = []
            for itm in trigger_items:
                neigh = [nbr for nbr, _ in self.swing_neighbors.get(itm, [])]
                # truncate to item_per_trigger
                neigh = neigh[:self.item_per_trigger]
                # pad up to item_per_trigger
                if len(neigh) < self.item_per_trigger:
                    neigh = neigh + [0] * (self.item_per_trigger - len(neigh))
                
                prompt_ids.extend(neigh)

            tensor_ids  = torch.tensor(prompt_ids, dtype=torch.long)
            trigger_items  = torch.tensor(trigger_items, dtype=torch.long)


            one_hop_list = [nbr for nbr, _ in self.swing_neighbors.get(item_seq[-1], [])]

                        

            final_list =one_hop_list[:10] + [0] * (10 - len( one_hop_list[:10]))


            neighbor_item_i2i = torch.tensor(final_list, dtype=torch.long)
            feed_dict["neighbor_item_i2i"] = neighbor_item_i2i


            feed_dict["prompt_ids"]  = tensor_ids        # [prompt_len]
            feed_dict["trigger_ids"] = trigger_items
            # --------------------------------------------------------------

            feed_dict.update(
                {
                    f"prompt_{feat}": torch.tensor(
                        [self.item_id2feat[iid][feat] for iid in tensor_ids],
                        dtype=torch.long,
                    )
                    for feat in self.item_id2feat[target_item]
                }
            )

            return feed_dict
    
    def _define_params(self, corpus):
        self.item_id_emb = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_ITEM_SEQ_LEN + 2, self.emb_size, padding_idx=0)
        self.feat_emb = nn.ModuleDict()
        for feat in corpus.item_feat_column:
            self.feat_emb[feat] = nn.Embedding(
                corpus.item_feat_num[feat] + 1, self.emb_size, padding_idx=0
            )

        self.trm_encoder = layers.TransformerBlockv2(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps,
        )
        self.model = rearec_layers.PromptReaRecBertWrapper(
            self.trm_encoder, self.hidden_size, self.reason_step, self.noise
        )


        self.loss_fct = nn.CrossEntropyLoss()

    @torch.no_grad()
    def encode_all_items(self, batch_size=8192):
        """
        Encode all items in the dataset.
        """
        self.eval()

        all_item_embs = []
        total_batches = (self.item_num + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Encoding items", unit="batch") as pbar:
            for start_idx in range(0, self.item_num, batch_size):
                end_idx = min(start_idx + batch_size, self.item_num)

                item_ids = torch.arange(
                    start_idx, end_idx, dtype=torch.long, device=self.device
                )

                batch_item_embs = [self.item_id_emb(item_ids)]

                for feat in self.feat_emb:
                    item_feat_ids = torch.tensor(
                        [
                            self.item_id2feat[item_id.item()][feat]
                            for item_id in item_ids
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                    item_feat_embs = self.feat_emb[feat](item_feat_ids)
                    item_feat_embs = self.avg_feat_emb(
                        item_feat_embs, item_feat_ids, is_seq=False
                    )
                    batch_item_embs.append(item_feat_embs)

                batch_final_embs = torch.sum(torch.stack(batch_item_embs, dim=1), dim=1)
                # stacked = torch.cat(batch_item_embs, dim=1)             # [B, L, 4*emb]
                # batch_final_embs = self._down_proj(stacked)
                all_item_embs.append(batch_final_embs)

                pbar.set_postfix(
                    {
                        "Items": f"{end_idx}/{self.item_num}",
                        "Batch_size": end_idx - start_idx,
                    }
                )
                pbar.update(1)

        self.all_item_embs = torch.cat(all_item_embs, dim=0)
        
    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        item_seq_ids, item_seq_len = feed_dict[ITEM_SEQ_ID], feed_dict[ITEM_SEQ_LEN]
        B = item_seq_ids.size(0)

        # Left Padding for item_seq_ids
        hist_padding_mask = item_seq_ids != 0
        valid_pos_ids = torch.cumsum(hist_padding_mask.long(), dim=1)
        pos_ids = torch.where(hist_padding_mask, valid_pos_ids, 0)
        pos_embs = self.pos_emb(pos_ids)

        # prompt_id and mask
        prompt_ids = feed_dict["prompt_ids"]

        item_feat_embs = [self.item_id_emb(item_seq_ids)]
        
        for feat in self.feat_emb:
            feat_ids = feed_dict[f"seq_{feat}"]
            feat_emb = self.feat_emb[feat](feat_ids)
            feat_emb = self.avg_feat_emb(feat_emb, feat_ids)
            item_feat_embs.append(feat_emb)
        item_embs = torch.sum(torch.stack(item_feat_embs, dim=2), dim=2)

        item_embs = item_embs + pos_embs
        # print(prompt_ids)
        prompt_feat_embs = [self.item_id_emb(prompt_ids)]
        for feat in self.feat_emb:
            feat_ids = feed_dict[f"seq_{feat}"]
            feat_emb = self.feat_emb[feat](feat_ids)
            feat_emb = self.avg_feat_emb(feat_emb, feat_ids)
            prompt_feat_embs.append(feat_emb)
        prompt_embs = torch.sum(torch.stack(prompt_feat_embs, dim=1), dim=1)
        # print(prompt_embs.shape)
        input_embs = torch.cat([prompt_embs, item_embs], dim=1)
        
        # Calculate sequence embeddings
        output, prompt_latent = self.model(
            input_embs,
            item_seq_len,
            prompt_ids,
            self.all_item_embs,
            epoch,
            feed_dict['neighbor_item_i2i'],
            feed_dict[ITEM_ID],
            stage,
            noise_factor=(
                self.noise
            ),
        )
        seq_embs = output[:B, -1, :]
        if stage == "all_steps":
            seq_embs = output
        
        feed_dict["seq_embs"] = seq_embs
        feed_dict["seq_output"] = output
        feed_dict["prompt_latent"] = prompt_latent

        feed_dict['epoch'] = epoch
        return feed_dict
    
    def kl_rank_softlabel_full_vocab(
        self,
        r_t: torch.Tensor,              # [B, D]
        pos_ids: torch.Tensor,          # [B, K]  (K<=10)
        target_ids: torch.Tensor,       # [B]
        temperature: float,
        gamma: float = 1.0,             # rank weight temperature
        use_cosine: bool = False,      
    ):

        E = self.all_item_embs  # [N, D]
        B = r_t.size(0)
        device = r_t.device
        K = pos_ids.size(1)

        if use_cosine:
            r = F.normalize(r_t, dim=-1)
            En = F.normalize(E, dim=-1)
            scores = (r @ En.T) / temperature          # [B, N]
        else:
            scores = (r_t @ E.T) / temperature         # [B, N]

        # logZ: [B]
        logZ = torch.logsumexp(scores, dim=-1)         # [B]

        sup_ids = torch.cat([target_ids.unsqueeze(1), pos_ids], dim=1)  # [B, 1+K]

        sup_scores = scores.gather(1, sup_ids)

        logp_sup = sup_scores - logZ.unsqueeze(1)      # [B, 1+K]
        rank = torch.arange(K+1, device=device).float()          # 0..K-1
        q_sup = F.softmax(-rank / gamma, dim=0)                    # [K]
        q_sup = q_sup.unsqueeze(0).expand(B, -1)                       # [B, K]


        logq_sup = torch.log(q_sup.clamp_min(1e-12))
        kl = (q_sup * (logq_sup - logp_sup)).sum(dim=1).mean()
        return kl

    
    def loss(self, out_dict: dict) -> torch.Tensor:
        # self.encode_all_items()
        """
        Compute the loss for the model output.
        :param out_dict: Dictionary containing model outputs
        :return: Loss value
        """
        seq_embs = out_dict["seq_embs"]
        target_item_ids = out_dict[ITEM_ID]
        batch_size = len(seq_embs)
        
        logits = torch.matmul(seq_embs, self.all_item_embs.T)

        trigger_ids, prompt_latent = out_dict["trigger_ids"], out_dict["prompt_latent"] # [B, K], # [B, K * P]
        thinking_embs = out_dict["seq_output"]  # (B, T, D)
        T = thinking_embs.shape[1]
    
    
        thinking_sup_loss = 0.0
        if out_dict['epoch'] >= 0:
            for i in range(T):
                thinking_item = out_dict['neighbor_item_i2i'][:,:10]
                thinking_emb = thinking_embs[:batch_size, i, :]
                thinking_sup_loss += self.kl_rank_softlabel_full_vocab(
                                            r_t=thinking_emb,
                                            pos_ids=thinking_item,
                                            target_ids=target_item_ids,
                                            temperature=1.0,
                                            gamma=1.0 * (T - i),
                                        )
        


        all_logits = torch.einsum(
            "btd,nd->btn", thinking_embs[:batch_size], self.all_item_embs
        )
        temp_scales = self.temperature * (torch.arange(1, T+1).to(logits.device) ** 2)
        scaled_logits = all_logits / temp_scales.view(1, T, 1)
        pl_loss = self.loss_fct(
            scaled_logits.view(-1, self.all_item_embs.size(0)),
            target_item_ids.repeat_interleave(T)
        )

        return  pl_loss * self.pl_weight  + thinking_sup_loss

    @torch.no_grad()
    def inference(self, feed_dict: dict) -> dict:
        """
        Inference method for the model.
        :param feed_dict: Dictionary containing input data
        :return: Dictionary with inference results
        """
        seq_embs = self.forward(feed_dict, stage="infer")["seq_embs"]
        if self.all_item_embs is None:
            self.encode_all_items()

        logits = torch.matmul(seq_embs, self.all_item_embs.transpose(0, 1))
        return {"prediction": logits}

    def analyze_reasoning_dynamics(self, logits_all, target_ids, k=10):
        """
        logits_all: [B, T, N] 
        target_ids: [B] 
        """
        B, T, N = logits_all.shape
        device = logits_all.device

        probs_all = torch.softmax(logits_all, dim=-1)

        print(f"\n{'=' * 20} Reasoning Dynamics Analysis (Batch Avg) {'=' * 20}")
        print(
            f"{'Step':<5} | {'Entropy':<8} | {'MaxProb':<8} | {'KL(t, t-1)':<10} | {'TargetRank':<10} | {'Hit@10':<8}")
        print("-" * 75)

        last_probs = None

        stats = {"entropy": [], "kl": []}

        for t in range(T):
            probs_t = probs_all[:, t, :]  # [B, N]
            logits_t = logits_all[:, t, :]

            log_probs = torch.log(probs_t + 1e-9)
            entropy = -(probs_t * log_probs).sum(dim=-1).mean().item()
            max_prob = probs_t.max(dim=-1)[0].mean().item()

            if last_probs is not None:
    
                kl_dist = F.kl_div(log_probs, last_probs, reduction='none').sum(dim=-1).mean().item()
            else:
                kl_dist = 0.0

            last_probs = probs_t  

            target_logits = logits_t.gather(1, target_ids.unsqueeze(1))  # [B, 1]
            ranks = (logits_t > target_logits).sum(dim=-1) + 1  # [B]
            avg_rank = ranks.float().mean().item()

            # Hit Rate @ K
            hit_rate = (ranks <= k).float().mean().item()

            stats["entropy"].append(entropy)
            stats["kl"].append(kl_dist)

            print(
                f"{t:<5} | {entropy:.4f}   | {max_prob:.4f}   | {kl_dist:.6f}   | {avg_rank:.1f}      | {hit_rate:.4f}")

        print("=" * 75 + "\n")
        return stats

    @torch.no_grad()
    def inference_test(self, feed_dict: dict, kl_threshold) -> dict:
        """
        Reasoning early-stop inference:
        stop at earliest t where DKL(p_{t-1} || p_t) < kl_threshold.

        Returns:
        - prediction: [B_act, N]
        - stop_t:     [B_act]
        - kl_all:     [B_act, Tmax]  (kl_all[:,0]=+inf)
        """
        self.eval()

        out = self.forward(feed_dict, stage="infer")
        seq_output = out["seq_output"]  # [B_full, S, D]
        B_full, S, D = seq_output.shape

        if self.all_item_embs is None:
            self.encode_all_items()
        E = self.all_item_embs  # [N, D]

        Tmax = min(getattr(self, "infer_Tmax", S), S)
        min_steps = getattr(self, "infer_min_steps", 1)
        # kl_threshold = getattr(self, "infer_kl_threshold", 0.03)

        logits_all = torch.einsum("btd,nd->btn", seq_output[: B_full // 2, :Tmax, :], E)  # [B_act, Tmax, N]
        B_act = logits_all.size(0)

        self.analyze_reasoning_dynamics(logits_all, feed_dict[ITEM_ID])

        kl_all = torch.full((B_act, Tmax), float("inf"), device=logits_all.device)

        prev_probs = None
        for t in range(Tmax):
            logits_t = logits_all[:, t, :]                       # [B_act, N]
            probs_t = torch.softmax(logits_t, dim=-1)            # [B_act, N]
            log_probs_t = torch.log(probs_t + 1e-9)              # [B_act, N]

            if prev_probs is not None:
                # F.kl_div(input=log p_t, target=p_{t-1}) == KL(p_{t-1} || p_t)
                kl_t = torch.nn.functional.kl_div(
                    log_probs_t, prev_probs, reduction="none"
                ).sum(dim=-1)                                    # [B_act]
                kl_all[:, t] = kl_t

            prev_probs = probs_t

        stop_mask = (kl_all < kl_threshold)                      # [B_act, Tmax]
        if min_steps > 0:
            stop_mask[:, :min_steps] = False

        first_stop = stop_mask.float().argmax(dim=1)             # [B_act]
        has_stop = stop_mask.any(dim=1)                          # [B_act]
        stop_t = torch.where(has_stop, first_stop, torch.full_like(first_stop, Tmax - 1))
        print(f'stop_t: {stop_t}')

        final_logits = logits_all[torch.arange(B_act, device=logits_all.device), stop_t]  # [B_act, N]
        return {"prediction": final_logits, "stop_t": stop_t, "kl_all": kl_all}