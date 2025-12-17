"""
Exercise 9.1 — Drug–Gene Bipartite Network & Drug Similarity Network

Scop:
- să construiți o rețea bipartită drug–gene plecând de la un CSV
- să proiectați layer-ul de medicamente folosind similaritatea dintre seturile de gene
- să exportați un fișier cu muchiile de similaritate între medicamente

TODO:
- încărcați datele drug–gene
- construiți dict-ul drug -> set de gene țintă
- construiți graful bipartit drug–gene (NetworkX)
- calculați similaritatea dintre medicamente (ex. Jaccard)
- construiți graful drug similarity
- exportați tabelul cu muchii: drug1, drug2, weight
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List

import matplotlib.pyplot as plt
import itertools
import networkx as nx
import pickle
import pandas as pd
import pickle


# --------------------------
# Config — adaptați pentru handle-ul vostru
# --------------------------
HANDLE = "MariusJalba"

# Input: fișier cu coloane cel puțin: drug, gene
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Output directory & files
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_SUMMARY = OUT_DIR / f"drug_summary_{HANDLE}.csv"
OUT_DRUG_SIMILARITY = OUT_DIR / f"drug_similarity_{HANDLE}.csv"
OUT_GRAPH_DRUG_GENE = OUT_DIR / f"network_drug_gene_{HANDLE}.gpickle"


def ensure_exists(path: Path) -> None:
   if not path.exists():
    raise FileNotFoundError(f"Fisierul {Path} nu exista")


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"drug","gene"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Fisierul trebuie sa contina coloanele {required_columns}")
    return df

def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    return df.groupby("drug")["gene"].apply(set).to_dict()


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    B = nx.Graph()
    for drug, genes in drug2genes.items():
        B.add_node(drug,bipartite ="drug")
        for gene in genes:
            B.add_node(gene, bipartite = "gene")
            B.add_edge(drug, gene)
    return B

def summarize_drugs(drug2genes: Dict[str, Set[str]]) -> pd.DataFrame:
    data = [(drug, len(genes)) for drug, genes in drug2genes.items()]
    return pd.DataFrame(data, columns=["drug","num_targets"])


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def compute_drug_similarity_edges(
    drug2genes: Dict[str, Set[str]],
    min_sim: float = 0.0,
) -> List[Tuple[str, str, float]]:
   edges = []
   for d1, d2 in itertools.combinations(drug2genes.keys(), 2):
    sim = jaccard_similarity(drug2genes[d1], drug2genes[d2])
    if sim >= min_sim:
        edges.append((d1, d2, sim))
   return edges


def edges_to_dataframe(edges: List[Tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(edges, columns = ["drug1","drug2","similarity"])


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DRUG_GENE_CSV)

    df = load_drug_gene_table(DRUG_GENE_CSV)

    drug2genes = build_drug2genes(df)

    B = build_bipartite_graph(drug2genes)
    with open(OUT_GRAPH_DRUG_GENE, "wb") as f:
        pickle.dump(B, f, protocol = pickle.HIGHEST_PROTOCOL)

    summary_df = summarize_drugs(drug2genes)
    summary_df.to_csv(OUT_DRUG_SUMMARY, index=False)

    edges = compute_drug_similarity_edges(drug2genes, min_sim = 0.1)
    edges_df = edges_to_dataframe(edges)
    edges_df.to_csv(OUT_DRUG_SIMILARITY, index=False)
