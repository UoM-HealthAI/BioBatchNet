import scib
import scanpy as sc

# 生成一个简单的 AnnData 结构
adata = sc.datasets.paul15()  # 载入示例数据集
adata_raw = adata.copy()

# 添加模拟的 batch 和 label 信息
adata.obs['batch'] = ['batch1' if i % 2 == 0 else 'batch2' for i in range(adata.n_obs)]
adata.obs['celltype'] = ['type1' if i % 3 == 0 else 'type2' for i in range(adata.n_obs)]

# 计算 metrics
batch_key = 'batch'
label_key = 'celltype'
embed = 'X_pca'  # 假设有 PCA 结果
type_ = 'full'

# 执行验证
try:
    ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_=type_)
    print("ilisi_graph 可用:", ilisi)
except Exception as e:
    print("ilisi_graph 失败:", e)

try:
    pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key)
    print("pcr_comparison 可用:", pcr)
except Exception as e:
    print("pcr_comparison 失败:", e)

try:
    graph_connectivity = scib.me.graph_connectivity(adata, label_key=label_key)
    print("graph_connectivity 可用:", graph_connectivity)
except Exception as e:
    print("graph_connectivity 失败:", e)

try:
    asw_batch = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embed)
    print("silhouette_batch 可用:", asw_batch)
except Exception as e:
    print("silhouette_batch 失败:", e)

try:
    asw_cell = scib.me.silhouette(adata, label_key=label_key, embed=embed)
    print("silhouette 可用:", asw_cell)
except Exception as e:
    print("silhouette 失败:", e)

try:
    scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
    print("cluster_optimal_resolution 可用")
except Exception as e:
    print("cluster_optimal_resolution 失败:", e)

try:
    ari = scib.me.ari(adata, cluster_key="cluster", label_key="celltype")
    print("ARI 可用:", ari)
except Exception as e:
    print("ARI 失败:", e)

try:
    nmi = scib.me.nmi(adata, cluster_key="cluster", label_key="celltype")
    print("NMI 可用:", nmi)
except Exception as e:
    print("NMI 失败:", e)
