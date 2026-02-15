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
    0.05: '20260129_201004_disc0.05',
    0.1: '20260129_185122_disc0.1',
    0.3: '20260129_185122_disc0.3',
}

BASE_DIR = Path('/home/w29632hl/code/BatchEffect/BioBatchNet/src/saved/immucan')
RAW_DATA_PATH = Path('/home/w29632hl/code/BatchEffect/BioBatchNet/BBNDATA/IMC/IMMUcan.h5ad')
OUTPUT_DIR = Path('/home/w29632hl/code/BatchEffect/BioBatchNet/BBNDATA/Results/IMC/tumor_analysis')

THRESHOLD = 0.5
RESOLUTION = 0.8
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
    # best_res = RESOLUTION
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


MIN_EDGE_CELLS = 80  
TOPK = 2


def build_sankey_data(tumor_df):
    """Build sankey diagram data from tumor cell flow dataframe.

    For each source cluster, keep only top-K flow destinations.
    Edges with fewer than MIN_EDGE_CELLS cells are dropped (no 'others' node).
    """
    # Collect real cluster nodes only (no others)
    all_nodes = []
    for cond in CONDITIONS:
        all_nodes.extend(tumor_df[cond].unique())
    all_nodes = list(dict.fromkeys(all_nodes))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    sources, targets, values, colors = [], [], [], []

    # Color palette for patients
    patients = tumor_df['patient'].unique()
    n_patients = len(patients)
    colorscale = [
        f'rgba({int(255 * (i / n_patients))}, {int(100 + 100 * ((i * 3) % n_patients) / n_patients)}, {int(255 * (1 - i / n_patients))}, 0.5)'
        for i in range(n_patients)
    ]
    patient_to_color = {p: colorscale[i] for i, p in enumerate(patients)}

    for i in range(len(CONDITIONS) - 1):
        src_col = CONDITIONS[i]
        tgt_col = CONDITIONS[i + 1]

        flow_counts = tumor_df.groupby([src_col, tgt_col, 'patient']).size().reset_index(name='count')
        if flow_counts.empty:
            continue

        pair_counts = flow_counts.groupby([src_col, tgt_col], as_index=False)['count'].sum()

        n_kept, n_dropped = 0, 0
        for src_node, src_pairs in pair_counts.groupby(src_col):
            src_pairs = src_pairs.sort_values('count', ascending=False)
            top = src_pairs.head(TOPK)

            for _, row in top.iterrows():
                tgt = row[tgt_col]
                val = int(row['count'])
                if val < MIN_EDGE_CELLS:
                    n_dropped += 1
                    continue  # 直接忽略小边

                # 按 patient 拆边，Plotly 会自动堆叠成一条粗线分段染色
                patient_rows = flow_counts[(flow_counts[src_col] == src_node) & (flow_counts[tgt_col] == tgt)]
                for patient, pcount in patient_rows.groupby('patient')['count'].sum().items():
                    if pcount > 0:
                        sources.append(node_to_idx[src_node])
                        targets.append(node_to_idx[tgt])
                        values.append(int(pcount))
                        colors.append(patient_to_color[patient])
                n_kept += 1

        print(f"  {src_col} -> {tgt_col}: kept {n_kept} edges, dropped {n_dropped} (< {MIN_EDGE_CELLS})")

    # --- compress nodes: keep only nodes that appear in links ---
    used = set(sources) | set(targets)
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(sorted(used))}
    all_nodes = [all_nodes[old_i] for old_i in sorted(used)]
    sources = [old_to_new[s] for s in sources]
    targets = [old_to_new[t] for t in targets]

    return all_nodes, sources, targets, values, colors, patient_to_color


def create_sankey_figure(all_nodes, sources, targets, values, colors, patient_to_color):
    """Create plotly sankey diagram."""
    # Node positions: arrange by condition, undefined at bottom
    from collections import defaultdict
    cond_nodes = defaultdict(list)
    for i, node in enumerate(all_nodes):
        cond = node.rsplit('_', 1)[0]
        cond_nodes[cond].append((i, node))

    node_x = [0.0] * len(all_nodes)
    node_y = [0.0] * len(all_nodes)
    for cond, nodes in cond_nodes.items():
        cond_idx = CONDITIONS.index(cond) if cond in CONDITIONS else 0
        # Sort: regular clusters first, undefined last
        nodes.sort(key=lambda x: (1 if 'undefined' in x[1] else 0, x[1]))
        n = len(nodes)
        for rank, (idx, _node) in enumerate(nodes):
            node_x[idx] = cond_idx / (len(CONDITIONS) - 1)
            node_y[idx] = 0.01 + rank * 0.98 / max(n - 1, 1)

    # Node colors: gray for tumor clusters, dark red for undefined
    CLUSTER_COLOR = 'rgba(180, 160, 210, 0.8)'
    UNDEF_COLOR = 'rgba(200, 80, 80, 0.8)'
    node_colors = [UNDEF_COLOR if 'undefined' in node else CLUSTER_COLOR for node in all_nodes]

    # Node labels: hide
    node_labels = ['' for _ in all_nodes]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=node_labels,
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

    # Add condition names as column headers
    for i, cond in enumerate(CONDITIONS):
        fig.add_annotation(
            x=i / (len(CONDITIONS) - 1),
            y=1.05,
            text=f"<b>{cond.replace('disc', 'disc ')}</b>",
            showarrow=False,
            font=dict(size=14),
            xref='x',
            yref='paper',
        )

    # Add legend: node types (use opacity 1.0 to match Sankey rendering)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color=CLUSTER_COLOR.replace('0.8)', '1)'), symbol='square'),
        name='Tumor cluster', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color=UNDEF_COLOR.replace('0.8)', '1)'), symbol='square'),
        name='Undefined (non-tumor-majority)', showlegend=True
    ))

    # Add legend: patients
    for patient, color in patient_to_color.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color.replace('0.5)', '1)')),
            name=patient,
            showlegend=True
        ))

    fig.update_layout(
        title_text="",
        font_size=12,
        width=1200,
        height=800,
        xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
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
        path = BASE_DIR / disc_dir / 'seed_42' / 'adata.h5ad'
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
