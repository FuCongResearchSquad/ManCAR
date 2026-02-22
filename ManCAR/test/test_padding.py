import torch

# ----------------------------------------------------------------------
# 1)  Dummy constants & helpers
# ----------------------------------------------------------------------
MAX_ITEM_SEQ_LEN = 50          # left‑padded user history length
BATCH            = 3           # mini‑batch size
PROMPT_LEN       = 15          # number of prompt tokens

device = "cuda" if torch.cuda.is_available() else "cpu"

# random history lengths in [1 .. MAX_ITEM_SEQ_LEN]
hist_lens = torch.randint(1, MAX_ITEM_SEQ_LEN + 1, (BATCH,), device=device)

# batch × prompt_len  (0 = pad, 1 – 999 = valid prompt ids)
prompt_ids = torch.randint(0, 1000, (BATCH, PROMPT_LEN), device=device)
prompt_ids[0, 3:] = 0         # add some right‑pad
prompt_ids[1, 10:] = 0
prompt_ids[2, 6:] = 0

print("history lens:", hist_lens.tolist())
print("prompt ids (0 = pad):\n", prompt_ids.cpu().numpy())

# ----------------------------------------------------------------------
# 2)  Copy / paste your two implementations
# ----------------------------------------------------------------------
def pad_mask_v1(input_lens, prompt_ids, device):
    batch_size = len(input_lens)
    max_item_seq_len = MAX_ITEM_SEQ_LEN
    base = torch.arange(max_item_seq_len, device=device)           # [L]
    # [B,L] True → will be padded
    hist_pad = base.unsqueeze(0) < (max_item_seq_len - input_lens.unsqueeze(1))
    prompt_pad = prompt_ids.eq(0)                                  # [B,P]
    comb = torch.cat([prompt_pad, hist_pad], 1)                    # [B,P+L]
    return comb.unsqueeze(1).unsqueeze(2)                          # [B,1,1,P+L]

def pad_mask_v2(input_lens, prompt_ids, step, device):
    B = len(input_lens)
    L = MAX_ITEM_SEQ_LEN + step
    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    hist_pad = idx < (L - input_lens.unsqueeze(1))
    prompt_pad = prompt_ids.eq(0)
    comb = torch.cat([prompt_pad, hist_pad], 1)
    return comb.unsqueeze(1).unsqueeze(2)

def attn_mask(batch, seq_len, think_len, device, pad_mask):
    tot = seq_len + think_len
    m = torch.zeros((tot, tot), device=device)
    m[:seq_len, :seq_len] = 1
    if think_len > 0:
        m[-think_len:, -think_len:] = torch.tril(torch.ones((think_len, think_len), device=device))
        m[seq_len:, :seq_len] = 1
    m = m.unsqueeze(0).unsqueeze(1).expand(batch, 1, tot, tot)
    m = m.masked_fill(m == 0, -1e10).masked_fill(m == 1, 0.0)
    return m.masked_fill(pad_mask, -1e10)

# ----------------------------------------------------------------------
# 3)  Checks
# ----------------------------------------------------------------------
for step in (0, 2):
    m1 = pad_mask_v1(hist_lens, prompt_ids, device)
    m2 = pad_mask_v2(hist_lens, prompt_ids, step, device)  # step only changes history part
    same_shape   = m1.shape == m2.shape
    same_content = torch.equal(m1, m2) if step == 0 else "n/a"

    print(f"\n=== step = {step} ===")
    print("shape v1 :", list(m1.shape))
    print("shape v2 :", list(m2.shape))
    if step == 0:
        print("identical :", same_content)
    else:
        print("v2 extra col/row for step =", step)

    # build attention mask to be sure it broadcasts without error
    am = attn_mask(BATCH, PROMPT_LEN + MAX_ITEM_SEQ_LEN, step, device, m2)
    print("attn mask OK :", am.shape)

