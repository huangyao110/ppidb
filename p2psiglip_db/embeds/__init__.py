"""Per-PLM feature extractors. Each submodule is independently runnable
via `python -m p2psiglip_db.embeds.<plm>`; the root-level get_embeddings.py
dispatcher delegates here based on --plm.

Pooling:
  Most extractors accept --pool {mean,max,cls,residue}. Pooled outputs are
  (D,) fp32 arrays; residue outputs are (L,D) fp16 arrays. ProtT5 has no
  CLS/BOS token, so use mean, max, or residue there.

Embedding dims:
  esmc          960      — ESMC-300M
  esm2          1280     — facebook/esm2_t33_650M_UR50D
  prott5        1024     — Rostlab/prot_t5_xl_uniref50 encoder
  prostt5       1024     — Rostlab/ProstT5 encoder, AA mode
  prostt5_3di   1024     — Rostlab/ProstT5 encoder, 3Di-input mode
  saprot        1280     — westlake-repl/SaProt_650M_AF2, paired AA+3Di
  profam        D        — ProFam-1 pfLM, default pool is residue
  prosst_2048   768      — AI4Protein/ProSST-2048, sequence + structure tokens
"""

PLMS = ['esmc', 'esm2', 'prott5', 'prostt5', 'prostt5_3di', 'saprot', 'profam', 'prosst_2048']
