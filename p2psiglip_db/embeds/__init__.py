"""Per-PLM feature extractors. Each submodule is independently runnable
via `python -m p2psiglip_db.embeds.<plm>`; the root-level get_embeddings.py
dispatcher delegates here based on --plm.

Embedding dims:
  esmc          (960,)    — ESMC-300M, mean over residues, BOS/EOS stripped
  esm2          (1280,)   — facebook/esm2_t33_650M_UR50D
  prott5        (1024,)   — Rostlab/prot_t5_xl_uniref50 encoder
  prostt5       (1024,)   — Rostlab/ProstT5 encoder, AA mode
  prostt5_3di   (1024,)   — Rostlab/ProstT5 encoder, 3Di-input mode
  saprot        (1280,)   — westlake-repl/SaProt_650M_AF2, paired AA+3Di
                         pass --per-residue for (L,1280) fp16
  profam        (D,)      — ProFam-1 pfLM, pass --mean-pool for pooled
                         default output is per-residue (L,D) fp16
  prosst_2048   (768,)    — AI4Protein/ProSST-2048, sequence + structure tokens
"""

PLMS = ['esmc', 'esm2', 'prott5', 'prostt5', 'prostt5_3di', 'saprot', 'profam', 'prosst_2048']
