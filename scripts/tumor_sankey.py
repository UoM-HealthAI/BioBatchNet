"""
Tumor Sankey diagram: visualize tumor cell clustering flow across batch correction strengths.

Flow: raw → disc_0.05 → disc_0.1 → disc_0.3
- Only Tumor cells are tracked
- Tumor cluster defined by majority rule (>50% Tumor cells)
- Edges colored by Patient/BATCH to show batch effects
"""
import scanpy as sc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Configuration - reuse from tumor_biomarker_analysis.py
DISC_DIRS = {
    0.05: '20260204_150407_disc_0.05',
    0.1: '20260204_151101_disc_0.1',
    0.3: '20260204_151803_disc_0.3',
}

BASE_DIR = Path('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/BatchEffect/BioBatchNet/src/saved/immucan')
RAW_DATA_PATH = Path('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/BatchEffect/BioBatchNet/DATA/IMC/IMMUcan.h5ad')
OUTPUT_DIR = Path('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/BatchEffect/BioBatchNet/DATA/IMC/tumor_analysis_disc_sweep')

THRESHOLD = 0.5
SEED = 42
CONDITIONS = ['raw', 'disc0.05', 'disc0.1', 'disc0.3']


def best_leiden_by_nmi(adata: sc.AnnData, label_key: str, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0), random_state=None):
    """Find best leiden resolution by NMI with ground truth labels."""
    best = (-1.0, -1.0, None)
    random_state = random_state if random_state is not None else SEED
    y = adata.obs[label_key].values
    for r in resolutions:
        sc.tl.leiden(adata, key_added='cluster', resolution=r, random_state=random_state)
        pred = adata.obs['cluster'].values
        nmi = normalized_mutual_info_score(y, pred)
        ari = adjusted_rand_score(y, pred)
        if nmi > best[0]:
            best = (nmi, ari, r)
    print(f"  best_leiden_by_nmi: resolution={best[2]}, NMI={best[0]:.4f}, ARI={best[1]:.4f}")
    return best


def identify_tumor_clusters(adata, threshold=THRESHOLD):
    """Identify clusters where Tumor cells are majority (>threshold)."""
    ct_cluster = pd.crosstab(adata.obs['leiden'], adata.obs['celltype'], normalize='index')
    if 'Tumor' not in ct_cluster.columns:
        return []
    tumor_clusters = ct_cluster[ct_cluster['Tumor'] > threshold].index.tolist()
    return tumor_clusters


def process_condition(raw_adata, condition, corrected_path=None):
    """Process a single condition: cluster and assign tumor labels."""
    print(f"Processing: {condition}")
    if corrected_path is None:
        adata = raw_adata.copy()
        sc.pp.neighbors(adata, use_rep='X', random_state=SEED)
    else:
        adata = sc.read_h5ad(corrected_path)
        sc.pp.neighbors(adata, use_rep='X_biobatchnet', random_state=SEED)

    _, _, best_res = best_leiden_by_nmi(adata, 'celltype', random_state=SEED)
    sc.tl.leiden(adata, key_added='leiden', resolution=best_res, random_state=SEED)

    tumor_clusters = identify_tumor_clusters(adata)
    print(f"  Tumor clusters (>{THRESHOLD*100:.0f}%): {tumor_clusters}")

    # Assign labels for tumor cells
    # Tumor cells in tumor clusters: "{condition}_{cluster_id}"
    # Tumor cells in non-tumor clusters: "{condition}_undefined"
    labels = []
    for idx, row in adata.obs.iterrows():
        if row['celltype'] == 'Tumor':
            if row['leiden'] in tumor_clusters:
                labels.append(f"{condition}_{row['leiden']}")
            else:
                labels.append(f"{condition}_undefined")
        else:
            labels.append(None)  # Non-tumor cells

    return adata, labels


def build_sankey_data(tumor_df):
    """Build sankey diagram data from tumor cell flow dataframe."""
    all_nodes = []
    for cond in CONDITIONS:
        all_nodes.extend(tumor_df[cond].unique())
    all_nodes = list(dict.fromkeys(all_nodes))  # Preserve order, remove duplicates
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # Collect flows between consecutive conditions
    sources = []
    targets = []
    values = []
    colors = []

    # Color palette for patients
    patients = tumor_df['patient'].unique()
    n_patients = len(patients)
    # Generate distinct colors for each patient
    colorscale = [
        f'rgba({int(255 * (i / n_patients))}, {int(100 + 100 * ((i * 3) % n_patients) / n_patients)}, {int(255 * (1 - i / n_patients))}, 0.5)'
        for i in range(n_patients)
    ]
    patient_to_color = {p: colorscale[i] for i, p in enumerate(patients)}

    for i in range(len(CONDITIONS) - 1):
        src_col = CONDITIONS[i]
        tgt_col = CONDITIONS[i + 1]

        # Group by source, target, and patient
        flow_counts = tumor_df.groupby([src_col, tgt_col, 'patient']).size().reset_index(name='count')

        for _, row in flow_counts.iterrows():
            sources.append(node_to_idx[row[src_col]])
            targets.append(node_to_idx[row[tgt_col]])
            values.append(row['count'])
            colors.append(patient_to_color[row['patient']])

    return all_nodes, sources, targets, values, colors, patient_to_color


def create_sankey_figure(all_nodes, sources, targets, values, colors, patient_to_color):
    """Create plotly sankey diagram."""
    # Node positions: arrange by condition
    node_x = []
    node_y = []
    for node in all_nodes:
        cond = node.rsplit('_', 1)[0]
        cond_idx = CONDITIONS.index(cond) if cond in CONDITIONS else 0
        node_x.append(cond_idx / (len(CONDITIONS) - 1))
        node_y.append(0.5)  # Let plotly handle y positioning

    # Node colors: gray for undefined, blue for tumor clusters
    node_colors = []
    for node in all_nodes:
        if 'undefined' in node:
            node_colors.append('rgba(150, 150, 150, 0.8)')
        else:
            node_colors.append('rgba(100, 149, 237, 0.8)')  # Cornflower blue

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=all_nodes,
            color=node_colors,
            x=node_x,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
        )
    ))

    # Add legend for patients
    # Create invisible scatter traces for legend
    for patient, color in patient_to_color.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color.replace('0.5)', '1)')),
            name=patient,
            showlegend=True
        ))

    fig.update_layout(
        title_text="Tumor Cell Flow Across Batch Correction Strengths<br><sub>Edges colored by Patient (BATCH)</sub>",
        font_size=12,
        width=1200,
        height=800,
    )

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data: {RAW_DATA_PATH}")
    raw_adata = sc.read_h5ad(RAW_DATA_PATH)
    print(f"  n_obs={raw_adata.n_obs}, n_vars={raw_adata.n_vars}")

    # Process all conditions
    condition_labels = {}

    # Raw
    _, labels = process_condition(raw_adata, 'raw')
    condition_labels['raw'] = labels

    # Corrected conditions
    for disc, disc_dir in sorted(DISC_DIRS.items()):
        path = BASE_DIR / disc_dir / 'seed_42' / 'biobatchnet.h5ad'
        cond_name = f'disc{disc}'
        _, labels = process_condition(raw_adata, cond_name, path)
        condition_labels[cond_name] = labels

    # Build dataframe for tumor cells only
    tumor_mask = raw_adata.obs['celltype'] == 'Tumor'
    tumor_indices = raw_adata.obs.index[tumor_mask]

    tumor_df = pd.DataFrame({
        'cell_id': tumor_indices,
        'patient': raw_adata.obs.loc[tumor_indices, 'BATCH'].values,
    })

    for cond in CONDITIONS:
        # Extract labels for tumor cells
        cond_labels = condition_labels[cond]
        tumor_df[cond] = [cond_labels[raw_adata.obs.index.get_loc(idx)] for idx in tumor_indices]

    print(f"\nTumor cells: {len(tumor_df)}")
    print(f"Patients: {tumor_df['patient'].nunique()}")

    # Build sankey data
    all_nodes, sources, targets, values, colors, patient_to_color = build_sankey_data(tumor_df)

    print(f"\nSankey nodes: {len(all_nodes)}")
    print(f"Sankey links: {len(sources)}")

    # Create and save figure
    fig = create_sankey_figure(all_nodes, sources, targets, values, colors, patient_to_color)

    output_html = OUTPUT_DIR / 'tumor_sankey.html'
    fig.write_html(str(output_html))
    print(f"\nSaved: {output_html}")

    # Also save as PNG
    try:
        output_png = OUTPUT_DIR / 'tumor_sankey.png'
        fig.write_image(str(output_png), scale=2)
        print(f"Saved: {output_png}")
    except Exception as e:
        print(f"Could not save PNG (kaleido may not be installed): {e}")


if __name__ == '__main__':
    main()
