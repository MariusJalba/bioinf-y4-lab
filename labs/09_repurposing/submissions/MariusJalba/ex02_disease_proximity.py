"""
Exercise 9.2 — Disease Proximity and Drug Ranking

Scop:
- să calculați distanța medie dintre fiecare medicament și un set de gene asociate unei boli
- să ordonați medicamentele în funcție de proximitate (network-based prioritization)

TODO-uri principale:
- încărcați graful bipartit drug–gene (din exercițiul 9.1) sau reconstruiți-l
- încărcați lista de disease genes
- pentru fiecare medicament, calculați distanța minimă / medie până la genele bolii
- exportați un tabel cu medicamente și scorul lor de proximitate
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pickle
import networkx as nx
import pandas as pd

# --------------------------
# Config
# --------------------------
HANDLE = "MariusJalba"

# Input: graful bipartit (salvat anterior) SAU tabelul drug-gene
GRAPH_DRUG_GENE = Path(f"labs/09_repurposing/submissions/{HANDLE}/network_drug_gene_{HANDLE}.gpickle")
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Input: lista genelor bolii
DISEASE_GENES_TXT = Path(f"data/work/{HANDLE}/lab09/disease_genes_{HANDLE}.txt")

# Output directory & file
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_PRIORITY = OUT_DIR / f"drug_priority_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Fisierul {Path} nu exista")


def load_bipartite_graph_or_build() -> nx.Graph:
    if GRAPH_DRUG_GENE.exists():
        with open(GRAPH_DRUG_GENE, "rb") as f:
            G = pickle.load(f)
        return G
    else:
        df = pd.read_csv(DRUG_GENE_CSV)
        required_columns = {"drug","gene"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Fisierul trebuie sa contina coloanele {required_columns}")
        df.groupby("drug")["gene"].apply(set).to_dict()
        G = nx.Graph()
        for drug, genes in drug2genes.items():
            G.add_node(drug,bipartite ="drug")
            for gene in genes:
                G.add_node(gene, bipartite = "gene")
                G.add_edge(drug, gene)
        return G



def load_disease_genes(path: Path) -> Set[str]:
    genes = set()
    with open(path, "r") as f:
        for line in f:
            gene = line.strip()
            if gene:
                genes.add(gene) 
    return genes


def get_drug_nodes(B: nx.Graph) -> List[str]:
    drugs = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "drug"]
    return drugs

def compute_drug_disease_distance(
    B: nx.Graph,
    drug: str,
    disease_genes: Set[str],
    mode: str = "mean",
    max_dist: int = 5,
) -> float:
    distances = []
    for gene in disease_genes:
        if gene in B:
            try:
                d = nx.shortest_path_length(B, source=drug, target=gene)
                distances.append(d)
            except nx.NetworkXNoPath:
                distances.append(max_dist + 1)
        else:
            distances.append(max_dist + 1)

    if mode == "min":
        return min(distances)
    if mode == "mean":
        return sum(distances) / len(distances)


def rank_drugs_by_proximity(
    B: nx.Graph,
    disease_genes: Set[str],
    mode: str = "mean",
) -> pd.DataFrame:
    drugs = get_drug_nodes(B)
    results = []
    for drug in drugs:
        dist = compute_drug_disease_distance(B, drug, disease_genes, mode=mode)
        results.append({"drug": drug, "distance":dist})
    df = pd.DataFrame(results)
    df = df.sort_values("distance", ascending=True).reset_index(drop=True)
    return df

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DRUG_GENE_CSV)
    ensure_exists(DISEASE_GENES_TXT)

    B = load_bipartite_graph_or_build()

    disease_genes = load_disease_genes(DISEASE_GENES_TXT)

    df_priority = rank_drugs_by_proximity(B, disease_genes, mode = "mean")
    
    df_priority.to_csv(OUT_DRUG_PRIORITY, index=False) 