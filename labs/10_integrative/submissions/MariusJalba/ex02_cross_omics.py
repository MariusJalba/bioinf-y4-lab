"""
Exercise 10.2 — Identify top SNP–Gene correlations

TODO:
- încărcați matricea integrată multi-omics
- împărțiți rândurile în SNPs vs gene (după indice sau după nume)
- calculați corelații între fiecare SNP și fiecare genă
- filtrați |r| > 0.5
- exportați snp_gene_pairs_<handle>.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

HANDLE = "MariusJalba"
JOINT_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/multiomics_concat_{HANDLE}.csv")

OUT_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/snp_gene_pairs_{HANDLE}.csv")


df = pd.read_csv(JOINT_CSV, index_col=0)
snp_df = df[df.index.str.startswith("rs")]
gene_df = df[~df.index.str.startswith("rs")]

#excludem liniile cu variatie 0
snp_df = snp_df.loc[snp_df.std(axis=1) > 0]
gene_df = gene_df.loc[gene_df.std(axis=1) > 0]

X = snp_df.to_numpy(dtype=np.float32)
y = gene_df.to_numpy(dtype=np.float32)

n = X.shape[1]

R = (X @ y.T) / n

rows, cols = np.where(np.abs(R) > 0.5)
snp_ids = snp_df.index.to_numpy()
gene_ids = gene_df.index.to_numpy()

df_pairs = pd.DataFrame({"SNP": snp_ids[rows],"Gene": gene_ids[cols], "Correlation": R[rows, cols].astype(float)})

df_pairs["abs_r"] = df_pairs["Correlation"].abs()
df_pairs = df_pairs.sort_values("abs_r", ascending=False).drop(columns=["abs_r"])
df_pairs.to_csv(OUT_CSV, index=False)
