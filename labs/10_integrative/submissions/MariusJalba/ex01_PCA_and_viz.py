"""
Exercise 10 — PCA Single-Omics vs Joint

TODO:
- încărcați SNP și Expression
- normalizați fiecare strat (z-score)
- rulați PCA pe:
    1) strat SNP
    2) strat Expression
    3) strat Joint (concat)
- generați 3 figuri PNG
- comparați vizual distribuția probelor
"""

from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

HANDLE = "MariusJalba"

SNP_CSV = Path(f"data/work/{HANDLE}/lab10/snp_matrix_{HANDLE}.csv")
EXP_CSV = Path(f"data/work/{HANDLE}/lab10/expression_matrix_{HANDLE}.csv")

OUT_DIR = Path(f"labs/10_integrative/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load(snp_path: Path, exp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_snp = pd.read_csv(snp_path, index_col=0)
    df_exp = pd.read_csv(exp_path, index_col=0)
    return df_snp, df_exp

def alignemnt(df_snp: pd.DataFrame, df_exp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = df_snp.columns.intersection(df_exp.columns)
    if len(common) == 0:
        raise ValueError("No common samples between SNP and EXP")
    
    df_snp = df_snp[common]
    df_exp = df_exp[common]
    return df_snp, df_exp

def numeric_to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.apply(pd.to_numeric, errors="coerce")
    df_num = df_num.dropna(axis=0, how="all")
    df_num = df_num.dropna(axis=0, how="any")
    return df_num

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0,1)
    return df.sub(mu,axis=0).div(sigma,axis=0)

def run_PCA(df: pd.DataFrame, title: str, out_png: Path):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(df.T)
    plt.scatter(proj[:, 0], proj[:, 1], c="blue")
    plt.title("PCA on Integrated Multi-Omics")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

if __name__ == "__main__":
    df_snp, df_exp = load(SNP_CSV, EXP_CSV)
    df_snp, df_exp = alignemnt(df_snp, df_exp)
    df_snp = numeric_to_matrix(df_snp)
    df_exp = numeric_to_matrix(df_exp)
    df_snp_norm = normalize(df_snp)
    df_exp_norm = normalize(df_exp)
    df_joint = pd.concat([df_snp_norm, df_exp_norm], axis=0)
    joint_csv = OUT_DIR / f"multiomics_concat_MariusJalba.csv"
    df_joint.to_csv(joint_csv)
    run_PCA(df_snp_norm, "PCA SNP Only", OUT_DIR / "pca_snp_MariusJalba.png")
    run_PCA(df_exp_norm, "PCA EXP Only", OUT_DIR / "pca_exp_MariusJalba.png")
    run_PCA(df_joint, "PCA Joint", OUT_DIR / "pca_joint_MariusJalba.png")


