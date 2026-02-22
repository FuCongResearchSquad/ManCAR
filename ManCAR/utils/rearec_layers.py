# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import math
import copy
from typing import Tuple, Optional
from torch.nn import functional as F
from torch import Tensor
import logging

from utils.constants import *
from utils.layers import BaseBERTEncoder

class ERLBERTEncoder(BaseBERTEncoder):
    """
    ERLBERTEncoder is a wrapper for ERL-based BERT4Rec model to generate sequence.

    Args:
        transformer (nn.Module): the BERT model
        step_num (int): the number of steps in the ReaRec model
        hidden_size (int): the hidden size of the BERT model
        dropout (float): dropout probability
        layer_norm_eps (float): epsilon for Layer normalization
    Returns:
        all_outputs (torch.Tensor): the output of the sequence
    """

    def __init__(self, transformer, step_num, hidden_size, layer_norm_eps=1e-12):
        super(ERLBERTEncoder, self).__init__(
            transformer, hidden_size, layer_norm_eps
        )
        self.step_num = step_num
        self.reason_pos_emb = nn.Embedding(step_num, hidden_size)

    def _prepare_padding_mask(self, input_lens, step_num, device):
        batch_size = len(input_lens)
        input_lens = input_lens + step_num
        max_item_seq_len = MAX_ITEM_SEQ_LEN + step_num
        # Create a padding mask where True indicates padding positions
        padding_mask = (
            torch.arange(max_item_seq_len)
            .unsqueeze(0)
            .expand(batch_size, max_item_seq_len)
            .to(device)
        )
        padding_mask = padding_mask < (max_item_seq_len - input_lens.unsqueeze(1))
        mask = torch.zeros_like(padding_mask, device=device)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask.unsqueeze(1).unsqueeze(2)

    def _prepare_attention_mask(self, seq_len, step_num, padding_mask, device):
        batch_size = padding_mask.size(0)
        mask = torch.ones(
            (batch_size, 1, seq_len + step_num, seq_len + step_num), device=device
        )
        mask[..., :seq_len, :seq_len] = 1
        # prefix masking
        if step_num > 0:
            mask[..., -step_num:, -step_num:] = torch.tril(
                torch.ones((step_num, step_num), device=device)
            )
            mask[seq_len:, :seq_len] = 1
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def forward(self, input_embs, input_lens):
        device = input_embs.device
        batch_size, seq_len, _ = input_embs.size()

        kv_caches = None
        all_step_outputs = []

        for cur_step in range(self.step_num + 1):
            padding_mask = self._prepare_padding_mask(input_lens, cur_step, device)
            attention_mask = self._prepare_attention_mask(
                seq_len, cur_step, padding_mask, device
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)

            outputs, kv_caches = self.transformer(
                input_embs,
                attention_mask,
                output_all_encoded_layers=True,
                kv_caches=kv_caches,
            )
            all_step_outputs.append(outputs[-1][:, -1, :])

            input_embs = outputs[-1]

        final_outputs = torch.stack(all_step_outputs, dim=1)

        return final_outputs  # [batch_size, step_num, hidden_size]

class AutoRegressiveWrapper(nn.Module):
    """
    AutoRegressiveWrapper is a wrapper for transformer model to generate auto-regressive sequence.

    Args:
        transformer (nn.Module): the transformer model
        reason_steps (int): the number of steps to generate auto-regressive sequence
        hidden_size (int): the hidden size of the transformer model

    Returns:
        all_outputs (torch.Tensor): the output of the auto-regressive sequence
    """

    def __init__(self, transformer, hidden_size, reason_step=0, dropout=0.5, layer_norm_eps=1e-12):
        super(AutoRegressiveWrapper, self).__init__()
        self.transformer = transformer
        self.n_layers = len(transformer.layers)
        self.reason_step = reason_step
        self.hidden_size = hidden_size
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

    def _prepare_attention_mask(self, batch_size, seq_len, device, padding_mask):
        mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def _prepare_padding_mask(self, input_lens, device, step):
        input_lens = input_lens + step
        batch_size = len(input_lens)
        max_item_seq_len = MAX_ITEM_SEQ_LEN + step
        padding_mask = (
            torch.arange(max_item_seq_len)
            .unsqueeze(0)
            .expand(batch_size, max_item_seq_len)
            .to(device)
        )
        padding_mask = padding_mask < (max_item_seq_len - input_lens.unsqueeze(1))
        return padding_mask.unsqueeze(1).unsqueeze(2)

    def forward(self, input_embs, input_lens):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        past_key_values = [None] * self.n_layers
        all_outputs = []

        for step in range(self.reason_step + 1):
            curr_seq_len = seq_len + step

            # 
            padding_mask = self._prepare_padding_mask(input_lens, device, step)
            attention_mask = self._prepare_attention_mask(
                batch_size, curr_seq_len, device, padding_mask
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            # Outputs (all encoder layers)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            last_hidden_states = outputs[0][-1][:, -1:, :]
            all_outputs.append(last_hidden_states)
            past_key_values = outputs[1]

            if step == self.reason_step:
                break

            input_embs = last_hidden_states

        if self.reason_step == 0:
            return all_outputs[0]

        return torch.cat(
            all_outputs, dim=1
        )  # [batch_size, reason_steps+1, hidden_size]


class ReaRecAutoRegressiveWrapper(AutoRegressiveWrapper):
    def __init__(self, transformer, hidden_size, reason_step):
        super(ReaRecAutoRegressiveWrapper, self).__init__(
            transformer,
            hidden_size,
            reason_step,
        )
        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)

    def forward(self, input_embs, input_lens, noise_factor=0.0, reason_step=None):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        # Duplicate batch, B: onwards is the noisy view
        repeat_batch_size_factor = (noise_factor > 0.0) + 1
        input_embs = input_embs.repeat(repeat_batch_size_factor, 1, 1)
        input_lens = input_lens.repeat(repeat_batch_size_factor)

        past_key_values = [None] * self.n_layers
        all_outputs = []
        all_noise_outputs = []

        reason_step = reason_step if reason_step is not None else self.reason_step
        for step in range(reason_step + 1):
            curr_seq_len = seq_len + step

            padding_mask = self._prepare_padding_mask(input_lens, device, step)
            attention_mask = self._prepare_attention_mask(
                batch_size * repeat_batch_size_factor,
                curr_seq_len,
                device,
                padding_mask,
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            # Get non noisy view's state and add it to outputs
            last_hidden_states = outputs[0][-1][:batch_size, -1:, :]
            all_outputs.append(last_hidden_states)
            if noise_factor > 0:
                # Add noisy view state
                all_noise_outputs.append(outputs[0][-1][batch_size:, -1:, :])

            if step == reason_step:
                break

            past_key_values = outputs[1]

            # w/ positional embedding
            new_pos_emb = self.reason_pos_emb(
                torch.tensor([step], device=device)
            ).expand(batch_size, 1, -1)
            input_embs = last_hidden_states + new_pos_emb

            # w/o positional embedding
            # input_embs = last_hidden_states

            if noise_factor > 0.0:
                # Add gaussian noise to noisy view
                noise = (torch.randn_like(input_embs) * noise_factor).to(device)
                noise_input_embs = input_embs + noise
                input_embs = torch.cat([input_embs, noise_input_embs], dim=0)

        all_outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            all_noise_outputs = torch.cat(all_noise_outputs, dim=1)
            all_outputs = torch.cat([all_outputs, all_noise_outputs], dim=0)

        return all_outputs  # [batch_size, reason_steps+1, hidden_size]


class BertAutoRegressiveWrapper(AutoRegressiveWrapper):
    def __init__(self, transformer, hidden_size, reason_step=0):
        super(BertAutoRegressiveWrapper, self).__init__(
            transformer, hidden_size, reason_step
        )

    def _prepare_attention_mask(
        self, batch_size, seq_len, think_len, device, padding_mask
    ):
        mask = torch.zeros((total_len, total_len), device=device)
        mask[:seq_len, :seq_len] = 1
        if think_len > 0:
            mask[-think_len:, -think_len:] = torch.tril(
                torch.ones((think_len, think_len), device=device)
            )
            mask[seq_len:, :seq_len] = 1
        mask = (
            mask.unsqueeze(0)
            .unsqueeze(1)
            .expand(batch_size, 1, total_len, total_len)
        )
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def forward(self, input_embs, input_lens):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        past_key_values = [None] * self.n_layers
        all_outputs = []

        for step in range(self.reason_step + 1):
            padding_mask = self._prepare_padding_mask(input_lens, device, step)
            attention_mask = self._prepare_attention_mask(
                batch_size, seq_len, step, device, padding_mask
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            last_hidden_states = outputs[0][-1][:, -1:, :]
            all_outputs.append(last_hidden_states)
            past_key_values = outputs[1]

            if step == self.reason_step:
                break

            input_embs = last_hidden_states

        if self.reason_step == 0:
            return all_outputs[0]

        return torch.cat(
            all_outputs, dim=1
        )  # [batch_size, reason_steps+1, hidden_size]


class ReaRecBertWrapper(BertAutoRegressiveWrapper):
    def __init__(self, transformer, hidden_size, reason_step):
        super(ReaRecBertWrapper, self).__init__(
            transformer,
            hidden_size,
            reason_step,
        )
        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)

    def forward(self, input_embs, input_lens, noise_factor=0.0, reason_step=None):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        repeat_batch_size_factor = (noise_factor > 0.0) + 1
        input_embs = input_embs.repeat(repeat_batch_size_factor, 1, 1)
        input_lens = input_lens.repeat(repeat_batch_size_factor)

        past_key_values = [None] * self.n_layers
        all_outputs = []
        all_noise_outputs = []

        reason_step = reason_step if reason_step is not None else self.reason_step
        for step in range(reason_step + 1):
            padding_mask = self._prepare_padding_mask(input_lens, device, step)
            attention_mask = self._prepare_attention_mask(
                batch_size * repeat_batch_size_factor,
                seq_len,
                step,
                device,
                padding_mask,
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            last_hidden_states = outputs[0][-1][:batch_size, -1:, :]
            all_outputs.append(last_hidden_states)
            if noise_factor > 0:
                all_noise_outputs.append(outputs[0][-1][batch_size:, -1:, :])

            if step == reason_step:
                break

            past_key_values = outputs[1]

            # w/ positional embedding
            new_pos_emb = self.reason_pos_emb(
                torch.tensor([step], device=device)
            ).expand(batch_size, 1, -1)
            input_embs = last_hidden_states + new_pos_emb

            # w/o positional embedding
            # input_embs = last_hidden_states

            if noise_factor > 0.0:
                noise = (torch.randn_like(input_embs) * noise_factor).to(device)
                noise_input_embs = input_embs + noise
                input_embs = torch.cat([input_embs, noise_input_embs], dim=0)

        all_outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            all_noise_outputs = torch.cat(all_noise_outputs, dim=1)
            all_outputs = torch.cat([all_outputs, all_noise_outputs], dim=0)

        return all_outputs  # [batch_size, reason_steps+1, hidden_size]


class PromptBertAutoRegressiveWrapper(AutoRegressiveWrapper):
    def __init__(self, transformer, hidden_size, reason_step=0):
        super(PromptBertAutoRegressiveWrapper, self).__init__(
            transformer, hidden_size, reason_step
        )

    def _prepare_attention_mask(
        self, batch_size, seq_len, think_len, device, padding_mask
    ):
        mask = torch.zeros((total_len, total_len), device=device)
        mask[:seq_len, :seq_len] = 1
        if think_len > 0:
            mask[-think_len:, -think_len:] = torch.tril(
                torch.ones((think_len, think_len), device=device)
            )
            mask[seq_len:, :seq_len] = 1
        mask = (
            mask.unsqueeze(0)
            .unsqueeze(1)
            .expand(batch_size, 1, total_len, total_len)
        )
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def forward(self, input_embs, padding_mask, input_lens):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        past_key_values = [None] * self.n_layers
        all_outputs = []

        for step in range(self.reason_step + 1):
            attention_mask = self._prepare_attention_mask(
                batch_size, seq_len, step, device, padding_mask
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            last_hidden_states = outputs[0][-1][:, -1:, :]
            all_outputs.append(last_hidden_states)
            past_key_values = outputs[1]

            if step == self.reason_step:
                break

            input_embs = last_hidden_states

        if self.reason_step == 0:
            return all_outputs[0]

        return torch.cat(
            all_outputs, dim=1
        )  # [batch_size, reason_steps+1, hidden_size]


class PromptReaRecBertWrapper(PromptBertAutoRegressiveWrapper):
    """
    Wrapper that adds (optional) autoregressive “reasoning” steps on top of a base Transformer.

    • Inputs: prompt+history embeddings (already concatenated), their lengths, and prompt IDs.
    • If `reason_step == 0`: run the whole sequence once (no AR).  
      Return the last token embedding and the prompt segment embedding.
    • If `reason_step > 0`: iteratively append `reason_step` new tokens.
        - At each step: build masks, reuse KV cache, take the newest token, (optionally) add noise,
          add a positional embedding, and feed it back.
    • Padding mask = right‑padded prompt  + left‑padded history.  
      Attention mask = causal for “thinking” tokens, full for prefix, then padded with -1e10.
    • Noise path duplicates the batch (clean + noisy) to compute CL later.

    """

    def __init__(self, transformer, hidden_size, reason_step, noise_factor):
        super(PromptReaRecBertWrapper, self).__init__(transformer, hidden_size, reason_step)
        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)  # pos emb for each thinking step
        self.reason_step = reason_step

        self.adpter_norm = ThoughtAdapter(256)

        
    def _prepare_padding_mask(self, input_lens, prompt_ids, step, device):
        """
        Build a padding mask (True = pad) for prompt (right-pad) + history (left-pad) at current step.

        input_lens: [B]  real history lengths (without left-pad)
        prompt_ids : [B, P]
        step       : int, number of generated thinking tokens so far
        return     : [B,1,1,P+L] bool mask (broadcastable to attention scores)
        """
        B = len(input_lens)
        L = MAX_ITEM_SEQ_LEN + step

        prompt_pad = prompt_ids.eq(0)  # [B,P]
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        hist_start = L - input_lens.unsqueeze(1)
        hist_pad = idx < hist_start     # [B,L]  left side is pad

        combined_pad = torch.cat([prompt_pad, hist_pad], dim=1)  # [B,P+L]
        return combined_pad.unsqueeze(1).unsqueeze(2)

    def _prepare_attention_mask(self, batch_size, seq_len, think_len, device, padding_mask):
        """
        Build the full attention mask (float) with causal part for thinking tokens.

        seq_len   : prompt+history length
        think_len : current number of generated tokens
        padding_mask: [B,1,1,P+L]
        return    : [B,1,total,total] float mask with -1e10 on disallowed/pad, 0 on allowed
        """
        total_len = seq_len + think_len
        mask = torch.zeros((total_len, total_len), device=device)
        mask[:seq_len, :seq_len] = 1  # full visibility for prefix
        if think_len > 0:
            mask[-think_len:, -think_len:] = torch.tril(torch.ones((think_len, think_len), device=device))
            mask[seq_len:, :seq_len] = 1  # thinking tokens can see prefix
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def forward(self, input_embs, input_lens, prompt_ids, item_embs, epoch, i2i_label=None, target_id=None, stage='infer', noise_factor=0.01, reason_step=None):
        """
        input_embs : [B, P+L, D]  prompt + history embeddings
        input_lens : [B]         history lengths
        prompt_ids : [B, P]
        noise_factor: float      >0 duplicates batch with gaussian noise
        reason_step : override default steps if not None

        returns:
            outputs      : [B*(1 or 2), steps+1, D]  (or [B,1,D] if reason_step==0)
            prompt_latent: [B, P, D]
        """
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device
        _, prompt_len = prompt_ids.size()

        # ----- fast path: no reasoning -----
        if self.reason_step == 0:
            repeat_batch = (noise_factor > 0.0) + 1
            input_embs = input_embs.repeat(repeat_batch, 1, 1)
            prompt_ids = prompt_ids.repeat(repeat_batch, 1)
            input_lens = input_lens.repeat(repeat_batch)
            pad_mask = self._prepare_padding_mask(input_lens, prompt_ids, 0, device)
            attn_mask = self._prepare_attention_mask(batch_size * repeat_batch, seq_len, 0, device, pad_mask).bool()

            inp = self.dropout(self.LayerNorm(input_embs))
            hs, _, attention_scores, attention_mask = self.transformer(inp, attn_mask, kv_caches=None)

            # Prompt Latents used for swing loss, final for classification
            prompt_latent = hs[:batch_size, :prompt_len, :]
            final_token   = hs[:, -1:, :]
            return final_token, prompt_latent

        # ----- AR path -----
        repeat_batch = (noise_factor > 0.0) + 1
        input_embs = input_embs.repeat(repeat_batch, 1, 1)
        prompt_ids = prompt_ids.repeat(repeat_batch, 1)
        input_lens = input_lens.repeat(repeat_batch)

        past_kv = [None] * self.n_layers
        all_outputs, all_noise_outputs = [], []

        reason_step = self.reason_step if reason_step is None else reason_step
        for step in range(reason_step + 1):
            pad_mask = self._prepare_padding_mask(input_lens, prompt_ids, step, device)
            attn_mask = self._prepare_attention_mask(batch_size * repeat_batch,
                                                     seq_len, step, device, pad_mask).bool()

            input_embs = self.dropout(self.LayerNorm(input_embs))
            hs, past_kv, attention_scores, attention_mask = self.transformer(input_embs, attn_mask, kv_caches=past_kv)

            last_h = hs[:batch_size, -1:, :]
            
            all_outputs.append(last_h)
            
            last_h = self.adpter_norm(last_h, item_embs)
            

            if noise_factor > 0:
                all_noise_outputs.append(hs[batch_size:, -1:, :])

            if step == reason_step:
                break

            new_pos = self.reason_pos_emb(torch.tensor([step], device=device)).expand(batch_size, 1, -1)
            input_embs = last_h + new_pos
            if noise_factor > 0.0:
                noise = torch.randn_like(input_embs) * noise_factor
                input_embs = torch.cat([input_embs, input_embs + noise], dim=0)

        prompt_latent = hs[:batch_size, :prompt_len, :]
        outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            outputs = torch.cat([outputs, torch.cat(all_noise_outputs, dim=1)], dim=0)

        return outputs, prompt_latent


class PromptReaRecSupBertWrapper(PromptBertAutoRegressiveWrapper):
    """
    Wrapper that adds (optional) autoregressive “reasoning” steps on top of a base Transformer.

    English
    -------
    • Inputs: prompt+history embeddings (already concatenated), their lengths, and prompt IDs.
    • If `reason_step == 0`: run the whole sequence once (no AR).  
      Return the last token embedding and the prompt segment embedding.
    • If `reason_step > 0`: iteratively append `reason_step` new tokens.
        - At each step: build masks, reuse KV cache, take the newest token, (optionally) add noise,
          add a positional embedding, and feed it back.
    • Padding mask = right‑padded prompt  + left‑padded history.  
      Attention mask = causal for “thinking” tokens, full for prefix, then padded with -1e10.
    • Noise path duplicates the batch (clean + noisy) to compute CL later.

    """

    def __init__(self, transformer, hidden_size, reason_step, noise_factor):
        super(PromptReaRecSupBertWrapper, self).__init__(transformer, hidden_size, reason_step)
        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)  # pos emb for each thinking step
        self.reason_step = reason_step

    def _prepare_padding_mask(self, input_lens, prompt_ids, step, device):
        """
        Build a padding mask (True = pad) for prompt (right-pad) + history (left-pad) at current step.

        input_lens: [B]  real history lengths (without left-pad)
        prompt_ids : [B, P]
        step       : int, number of generated thinking tokens so far
        return     : [B,1,1,P+L] bool mask (broadcastable to attention scores)
        """
        B = len(input_lens)
        L = MAX_ITEM_SEQ_LEN + step

        prompt_pad = prompt_ids.eq(0)  # [B,P]
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        hist_start = L - input_lens.unsqueeze(1)
        hist_pad = idx < hist_start     # [B,L]  left side is pad

        combined_pad = torch.cat([prompt_pad, hist_pad], dim=1)  # [B,P+L]
        return combined_pad.unsqueeze(1).unsqueeze(2)

    def _prepare_attention_mask(self, batch_size, seq_len, think_len, device, padding_mask):
        """
        Build the full attention mask (float) with causal part for thinking tokens.

        seq_len   : prompt+history length
        think_len : current number of generated tokens
        padding_mask: [B,1,1,P+L]
        return    : [B,1,total,total] float mask with -1e10 on disallowed/pad, 0 on allowed
        """
        total_len = seq_len + think_len
        mask = torch.zeros((total_len, total_len), device=device)
        mask[:seq_len, :seq_len] = 1  # full visibility for prefix
        if think_len > 0:
            mask[-think_len:, -think_len:] = torch.tril(torch.ones((think_len, think_len), device=device))
            mask[seq_len:, :seq_len] = 1  # thinking tokens can see prefix
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def forward(self, input_embs, input_lens, prompt_ids, item_emb, noise_factor=0.01, reason_step=None):
        """
        input_embs : [B, P+L, D]  prompt + history embeddings
        input_lens : [B]         history lengths
        prompt_ids : [B, P]
        noise_factor: float      >0 duplicates batch with gaussian noise
        reason_step : override default steps if not None

        returns:
            outputs      : [B*(1 or 2), steps+1, D]  (or [B,1,D] if reason_step==0)
            prompt_latent: [B, P, D]
        """
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device
        _, prompt_len = prompt_ids.size()

        # ----- fast path: no reasoning -----
        if self.reason_step == 0:
            repeat_batch = (noise_factor > 0.0) + 1
            input_embs = input_embs.repeat(repeat_batch, 1, 1)
            prompt_ids = prompt_ids.repeat(repeat_batch, 1)
            input_lens = input_lens.repeat(repeat_batch)
            pad_mask = self._prepare_padding_mask(input_lens, prompt_ids, 0, device)
            attn_mask = self._prepare_attention_mask(batch_size * repeat_batch, seq_len, 0, device, pad_mask).bool()

            inp = self.dropout(self.LayerNorm(input_embs))
            hs, _ = self.transformer(inp, attn_mask, kv_caches=None)

            # Prompt Latents used for swing loss, final for classification
            prompt_latent = hs[:batch_size, :prompt_len, :]
            final_token   = hs[:, -1:, :]
            return final_token, prompt_latent

        # ----- AR path -----
        repeat_batch = (noise_factor > 0.0) + 1
        input_embs = input_embs.repeat(repeat_batch, 1, 1)
        prompt_ids = prompt_ids.repeat(repeat_batch, 1)
        input_lens = input_lens.repeat(repeat_batch)

        past_kv = [None] * self.n_layers
        all_outputs, all_noise_outputs = [], []

        reason_step = self.reason_step if reason_step is None else reason_step
        for step in range(reason_step + 1):
            pad_mask = self._prepare_padding_mask(input_lens, prompt_ids, step, device)
            attn_mask = self._prepare_attention_mask(batch_size * repeat_batch,
                                                     seq_len, step, device, pad_mask).bool()

            input_embs = self.dropout(self.LayerNorm(input_embs))
            hs, past_kv = self.transformer(input_embs, attn_mask, kv_caches=past_kv)

            last_h = hs[:batch_size, -1:, :]
            all_outputs.append(last_h)

            if noise_factor > 0:
                all_noise_outputs.append(hs[batch_size:, -1:, :])

            if step == reason_step:
                break

            # prepare next-step token
            new_pos = self.reason_pos_emb(torch.tensor([step], device=device)).expand(batch_size, 1, -1)
            input_embs = last_h + new_pos

            if noise_factor > 0.0:
                noise = torch.randn_like(input_embs) * noise_factor
                input_embs = torch.cat([input_embs, input_embs + noise], dim=0)

        prompt_latent = hs[:batch_size, :prompt_len, :]
        outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            outputs = torch.cat([outputs, torch.cat(all_noise_outputs, dim=1)], dim=0)

        return outputs, prompt_latent
    

class ThoughtAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    @torch.no_grad()
    def target_norm(self, emb_weight):
        return emb_weight.norm(dim=-1).mean()

    def forward(self, h, emb_weight):

        tn = self.target_norm(emb_weight).detach()
        hn = h.norm(dim=-1, keepdim=True)
        h = h * (tn / hn) * self.scale
        return h
    
    