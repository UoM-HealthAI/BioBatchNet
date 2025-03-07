import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from utils import visualize, target_distribution

def pretrain_autoencoder(model, data_loader, device, epochs=50, lr=1e-3):
    print("begin to train AutoEncoder")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_data in tqdm(data_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}"):
            x, _ = batch_data
            x = x.to(device)
            optimizer.zero_grad()
            z, out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Pretrain Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    print("Pretrain stage finished\n")

def initialize_clusters(model, data_loader, device):
    print("Useing K-means get cluster centers...")
    model.autoencoder.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="get embedding features"):
            x, y = batch_data
            x = x.to(device)
            z, _ = model.autoencoder(x)
            embeddings.append(z.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels, axis=0) 
    
    kmeans = KMeans(n_clusters=model.n_clusters, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(embeddings)
    
    model.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    visualize(embeddings, labels, kmeans.cluster_centers_)
    visualize(embeddings, y_pred, kmeans.cluster_centers_)
    print("cluster centers are found\n")

def train_DEC(model, data_loader, device, epochs=100, lr=1e-3, update_interval=10, tol=1e-3):
    print("begin to train DEC")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    previous_loss = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_data in tqdm(data_loader, desc=f"DEC train Epoch {epoch+1}/{epochs}"):
            x, _ = batch_data
            x = x.to(device)
            
            # 前向传播
            q = model(x)  # 软分配
            
            # 计算目标分布
            p = target_distribution(q).detach()
            
            # 计算损失
            loss = criterion(torch.log(q), p)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        # print(f"DEC Epoch {epoch+1}, 损失: {avg_loss:.4f}")
        
        # 每隔update_interval个epoch检查一次收敛
        if (epoch + 1) % update_interval == 0:
            model.eval()
            total_loss_val = 0
            with torch.no_grad():
                for batch_data in data_loader:
                    x, _ = batch_data
                    x = x.to(device)
                    q = model(x)
                    p = model.target_distribution(q).detach()
                    loss = criterion(torch.log(q), p)
                    total_loss_val += loss.item()
            avg_loss_val = total_loss_val / len(data_loader)
            print(f"验证损失: {avg_loss_val:.4f}")
            
            # 判断是否收敛
            if np.abs(previous_loss - avg_loss_val) < tol:
                print("收敛判定：训练停止。")
                break
            previous_loss = avg_loss_val
            model.train()
    print("DEC训练完成。\n")

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors

def clustering_training(
    model, 
    clustering_model, 
    X_tensor,
    dataloader, 
    optimizer, 
    device,
    num_cluster,
    num_epochs, 
    ratio_start=0.1,
    ratio_end=0.9,
    iter_start=0,
    iter_end=500,
    center_ratio=0.5,
):

    X_tensor = X_tensor.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    loss_fn_cls = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # === E-step: Prototype Pseudo-Labeling ===
        all_embeddings = []
        all_logits = []
        model.eval() 
        with torch.no_grad():
            for data in tqdm(dataloader, desc="E-step: Extracting embeddings"):
                inputs = data[0].to(device)
                embeddings = model.encoder(inputs)  
                logits = clustering_model(embeddings)  
                all_embeddings.append(embeddings)
                all_logits.append(logits)

        all_embeddings = torch.cat(all_embeddings)  # Shape: (N, D)
        all_logits = torch.cat(all_logits)          # Shape: (N, C)
        probabilities = torch.softmax(all_logits, dim=1)  # Shape: (N, C)
        
        # === 计算当前选择比例 ===
        if iter_end != iter_start:
            current_ratio = ratio_start + (ratio_end - ratio_start) * max(0, min(epoch - iter_start, iter_end - iter_start)) / (iter_end - iter_start)
            current_ratio = min(max(current_ratio, ratio_start), ratio_end)
        else:
            current_ratio = ratio_start
        print(f'Current selection ratio: {current_ratio:.4f}')
         
        # === 使用 select_samples 选择高置信度样本并分配伪标签 ===
        centers, idx_select, labels_select = select_samples(
            embeddings=all_embeddings,
            num_cluster=num_cluster,
            center_ratio=center_ratio,
            scores=probabilities,
            ratio=current_ratio
        )
        
        # === 分配伪标签 ===
        # 由于一个样本可能被多个簇分配标签，我们需要重复这些样本
        unique_indices, inverse_indices = torch.unique(idx_select, return_inverse=True)
        pseudo_labels_final = labels_select  # Shape: (num_selected,)
        
        # 获取选中的样本
        selected_embeddings = all_embeddings[idx_select]  # Shape: (num_selected, D)
        selected_labels = pseudo_labels_final  # Shape: (num_selected,)
        selected_labels = selected_labels.to(device)
        
        # === M-step: Training Clustering Head === 
        model.train()
        clustering_model.train()
        optimizer.zero_grad()
        
        # 前向传播
        logits_final = clustering_model(selected_embeddings)
        loss = loss_fn_cls(logits_final, selected_labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        print(f'Loss: {loss.item():.4f}')
    
    return centers

def select_samples(embeddings, num_cluster, center_ratio, scores, ratio):

    _, idx_max = torch.sort(scores, dim=0, descending=True)
    idx_max = idx_max.cpu()
    num_per_cluster = idx_max.shape[0] // num_cluster
    # print(ratio_select)
    k = int(center_ratio * num_per_cluster * ratio)
    # print(k, len(idx_max))
    idx_max = idx_max[0:k, :]

    centers = []
    for c in range(num_cluster):
        centers.append(embeddings[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

    centers = torch.cat(centers, dim=0)

    num_select_c = int(num_per_cluster * ratio)

    dis = torch.einsum('cd,nd->cn', [centers, embeddings])
    idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
    labels_select = torch.arange(0, num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()

    return centers, idx_select, labels_select