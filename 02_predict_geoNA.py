# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import os
import numpy as np
from collections import defaultdict
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# from luca_train_pro_geo_ES import VirusHostClassifier, encode_categorical_onehot, collate_fn_gpu_cache

class VirusHostClassifier(nn.Module):
    def __init__(self, embedding_dim=1280, reduced_dim=256, major_classes=[], func_list={},
                 host_cat_dims=None, virus_cat_dims=None, dtype=torch.float32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.major_classes = major_classes
        self.total_classes = len(major_classes)
        self.dtype = dtype
        self.func_weights = nn.Parameter(torch.zeros(len(func_list), dtype=self.dtype))
        self.func_to_idx = {func: idx for idx, func in enumerate(func_list.keys())}
        self.dim_reduction = nn.Linear(embedding_dim, reduced_dim, dtype=self.dtype)
        num_heads = 0
        self.class_queries = nn.ParameterDict({
            cls: nn.Parameter(torch.randn(num_heads, embedding_dim, dtype=self.dtype))
            for cls in major_classes
        })
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True,
            activation='relu', dtype=self.dtype
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.host_onehot_dim = sum(host_cat_dims)
        self.virus_onehot_dim = sum(virus_cat_dims)
        self.cls_head = nn.Sequential(
            nn.Linear(self.total_classes * reduced_dim + self.host_onehot_dim + self.virus_onehot_dim + 3, 128, dtype=self.dtype),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=self.dtype)
        )
    def aggregate_simple(self, embs, indices, device):
        if len(embs) == 0:
            return torch.zeros(self.embedding_dim, device=device, dtype=self.dtype)
        raw_weights = self.func_weights[indices]
        attn = F.softmax(raw_weights, dim=0)
        return (embs * attn.unsqueeze(1)).sum(dim=0)
    def forward(self, host_map_batch, host_indices_batch, virus_map_batch, viurs_indices_batch, host_cat_batch, virus_cat_batch, extra_meta_tensor, device):
        batch_agg_vectors = []
        for host_map, host_indices, virus_map, virus_indices in zip(host_map_batch, host_indices_batch, virus_map_batch, viurs_indices_batch):
            agg_vectors = []
            for cls in self.major_classes:
                embeddings = host_map.get(cls, []) if cls.startswith('KEGG') else virus_map.get(cls, [])
                indices = host_indices[cls] if cls.startswith('KEGG') else virus_indices[cls]
                agg = self.aggregate_simple(embeddings, indices, device)
                reduced = self.dim_reduction(agg)
                agg_vectors.append(reduced)
            batch_agg_vectors.append(torch.stack(agg_vectors, dim=0))
        batch_agg_vectors = torch.stack(batch_agg_vectors, dim=0)
        encoded = self.transformer(batch_agg_vectors)
        host_meta_feat = host_cat_batch
        virus_meta_feat = virus_cat_batch
        x = torch.cat([encoded.reshape(encoded.size(0), -1), host_meta_feat, virus_meta_feat, extra_meta_tensor], dim=1)
        logits = self.cls_head(x).squeeze(-1)
        return logits


def encode_categorical_onehot(row, cats, cat2id):
    onehot_vecs = []
    for col in cats:
        v = (row or {}).get(col, "NA")
        idx = cat2id[col].get(v, cat2id[col]["NA"])
        dim = len(cat2id[col])
        onehot = np.zeros(dim, dtype=np.float32)
        onehot[idx] = 1.0
        onehot_vecs.append(onehot)
    return np.concatenate(onehot_vecs, axis=0)

def collate_fn_gpu_cache(batch, device, host_cat_dims=None, virus_cat_dims=None):
    host_embs_list, host_idx_list, virus_embs_list, virus_idx_list, host_catfeats, virus_catfeats, labels, extra_meta_features = zip(*batch)
    all_emb_ids = set()
    emb_map = {}
    for m in host_embs_list + virus_embs_list:
        for k, embs in m.items():
            eid = id(embs)
            all_emb_ids.add(eid)
            emb_map[eid] = embs
    cpu2gpu = {eid: emb_map[eid].to(device, non_blocking=True) for eid in all_emb_ids}
    def replace(m):
        out = defaultdict(list)
        for k, v in m.items():
            out[k] = cpu2gpu[id(v)]
        return out
    host_embs_gpu = [replace(m) for m in host_embs_list]
    virus_embs_gpu = [replace(m) for m in virus_embs_list]
    host_idx_gpu = [ {k: idx_list.to(device) for k, idx_list in m.items()} for m in host_idx_list ]
    virus_idx_gpu = [ {k: idx_list.to(device) for k, idx_list in m.items()} for m in virus_idx_list ]
    host_catfeats_tensor = torch.tensor(np.stack(host_catfeats), dtype=torch.float32, device=device)
    virus_catfeats_tensor = torch.tensor(np.stack(virus_catfeats), dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    extra_meta_tensor = torch.tensor(np.stack(extra_meta_features), dtype=torch.float32, device=device)
    return host_embs_gpu, host_idx_gpu, virus_embs_gpu, virus_idx_gpu, host_catfeats_tensor, virus_catfeats_tensor, labels_tensor, extra_meta_tensor


def load_pred_virus_embeddings(vector_dir):
    """加载预测病毒的embeddings"""
    embed_dict = dict()
    for fn in os.listdir(vector_dir):
        if fn.endswith(".pt"):
            key = fn.replace("vector_", "").replace(".pt", "")
            embed = torch.load(os.path.join(vector_dir, fn), map_location="cpu")
            embed_dict[key] = embed
    logger.info(f"Loaded {len(embed_dict)} virus embeddings from {vector_dir}")
    return embed_dict

def build_host_proteins_by_file(kegg_host_csv, host_embed_dict):
    """构建宿主蛋白映射"""
    kegg_host_df = pd.read_csv(kegg_host_csv)
    kegg_host_df = kegg_host_df[kegg_host_df['longest_KO'] == 1]
    host_proteins_by_file = defaultdict(list)
    for _, row in kegg_host_df.iterrows():
        fname = row['file_name']
        seq = row['sequence']
        if seq in host_embed_dict:
            host_proteins_by_file[fname].append(seq)
    logger.info(f"Built host mappings for {len(host_proteins_by_file)} files")
    return host_proteins_by_file

def get_host_meta_dict(kegg_host_csv, host_info_csv):
    """获取宿主元数据映射"""
    kegg_host_df = pd.read_csv(kegg_host_csv)
    kegg_host_df = kegg_host_df[kegg_host_df['longest_KO'] == 1]
    host_info_df = pd.read_csv(host_info_csv, dtype=str).fillna("NA")
    seqname_to_meta = host_info_df.set_index('SeqName').to_dict(orient='index')
    file_name_to_seqname = kegg_host_df.groupby('file_name')['sequence'].first().to_dict()
    file_name_to_meta = {}
    for f, seq in file_name_to_seqname.items():
        meta = seqname_to_meta.get(seq, None)
        file_name_to_meta[f] = meta
    logger.info(f"Built host metadata for {len(file_name_to_meta)} files")
    return file_name_to_meta

def load_host_taxonomy_info(taxonomy_csv):
    logger.info(f"Loading host taxonomy information from {taxonomy_csv}...")
    
    if not os.path.exists(taxonomy_csv):
        logger.warning(f"Taxonomy file not found: {taxonomy_csv}")
        return pd.DataFrame()
    
    try:
        taxonomy_df = pd.read_csv(taxonomy_csv)
        taxonomy_cols = [
            'host_filename', 'host_taxid', 'host.species.name', 
            'host.genus', 'host.family', 'host.order', 
            'host.class', 'host.group'
        ]

        available_cols = [col for col in taxonomy_cols if col in taxonomy_df.columns]
        missing_cols = [col for col in taxonomy_cols if col not in taxonomy_df.columns]
        
        if missing_cols:
            logger.warning(f"⚠ Missing columns in taxonomy file: {missing_cols}")
        
        if 'host_filename' not in available_cols:
            logger.error("Required column 'host_filename' not found!")
            return pd.DataFrame()
        
        taxonomy_df = taxonomy_df[available_cols].copy()
        
        original_count = len(taxonomy_df)
        taxonomy_df = taxonomy_df.drop_duplicates(subset=['host_filename'], keep='first')
        deduplicated_count = len(taxonomy_df)
        
        logger.info(f"Original entries: {original_count}")
        logger.info(f"Unique host_filename entries: {deduplicated_count}")
        if original_count > deduplicated_count:
            logger.info(f"Removed {original_count - deduplicated_count} duplicates")
        
        logger.info(f"Available taxonomy columns: {available_cols}")
        
        return taxonomy_df
        
    except Exception as e:
        logger.error(f"Failed to load taxonomy file: {e}")
        return pd.DataFrame()

class VirusHostPredDataset(torch.utils.data.Dataset):
    """预测数据集类"""
    def __init__(self, pairs, host_embed_dict, virus_embed_dict, host_func_map, virus_func_map,
                 func_list, file_name_to_meta, virus_info_dict, host_cats, virus_cats, 
                 host_cat2id, virus_cat2id, host_cat_dims, virus_cat_dims, embedding_dim):
        self.pairs = pairs
        self.host_embed_dict = host_embed_dict
        self.virus_embed_dict = virus_embed_dict
        self.host_func_map = host_func_map
        self.virus_func_map = virus_func_map
        self.func_list = func_list
        self.file_name_to_meta = file_name_to_meta
        self.virus_info_dict = virus_info_dict
        self.host_cats = host_cats
        self.virus_cats = virus_cats
        self.host_cat2id = host_cat2id
        self.virus_cat2id = virus_cat2id
        self.host_cat_dims = host_cat_dims
        self.virus_cat_dims = virus_cat_dims
        self.embedding_dim = embedding_dim
        
        # 构建 func_to_idx 映射
        self.func_to_idx = {func: idx for idx, func in enumerate(func_list.keys())}
        
        # 获取所有 major_classes
        all_classes = set()
        for func_categories in func_list.values():
            if isinstance(func_categories, list):
                all_classes.update(func_categories)
            else:
                all_classes.add(func_categories)
        self.major_classes = sorted(all_classes)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        virus_ids, host_ids, label, host_file_name, virus_filename = self.pairs[idx]

        virus_map = defaultdict(list)
        for pid in virus_ids:
            emb = self.virus_embed_dict.get(pid)
            func = self.virus_func_map.get(pid, "NA")
            major_class_raw = self.func_list.get(func, None)
            
            if emb is not None and major_class_raw is not None:

                if isinstance(major_class_raw, list):
                    major_classes = major_class_raw
                else:
                    major_classes = [major_class_raw]
                
                for major_class in major_classes:
                    virus_map[major_class].append((emb, func))
        
        host_map = defaultdict(list)
        for pid in host_ids:
            emb = self.host_embed_dict.get(pid)
            func = self.host_func_map.get(pid, "NA")
            major_class_raw = self.func_list.get(func, None)
            
            if emb is not None and major_class_raw is not None:
     
                if isinstance(major_class_raw, list):
                    major_classes = major_class_raw
                else:
                    major_classes = [major_class_raw]
                
                for major_class in major_classes:
                    host_map[major_class].append((emb, func))
        
        virus_embs = {}
        virus_indices = {}
        for cat in self.major_classes:
            if not cat.startswith('KEGG'):  # 病毒类别
                if cat in virus_map and len(virus_map[cat]) > 0:
                    embs_list = []
                    indices_list = []
                    for emb, func in virus_map[cat]:
                        if not torch.is_tensor(emb):
                            emb = torch.tensor(emb, dtype=torch.float32)
                        else:
                            emb = emb.clone().detach().to(torch.float32)
                        embs_list.append(emb)
                        indices_list.append(self.func_to_idx[func])
                    
                    virus_embs[cat] = torch.stack(embs_list)
                    virus_indices[cat] = torch.tensor(indices_list, dtype=torch.long)
                else:
                    virus_embs[cat] = torch.zeros((0, self.embedding_dim), dtype=torch.float32)
                    virus_indices[cat] = torch.zeros((0,), dtype=torch.long)
        
        host_embs = {}
        host_indices = {}
        for cat in self.major_classes:
            if cat.startswith('KEGG'):  # 宿主类别
                if cat in host_map and len(host_map[cat]) > 0:
                    embs_list = []
                    indices_list = []
                    for emb, func in host_map[cat]:
                        if not torch.is_tensor(emb):
                            emb = torch.tensor(emb, dtype=torch.float32)
                        else:
                            emb = emb.clone().detach().to(torch.float32)
                        embs_list.append(emb)
                        indices_list.append(self.func_to_idx[func])
                    
                    host_embs[cat] = torch.stack(embs_list)
                    host_indices[cat] = torch.tensor(indices_list, dtype=torch.long)
                else:
                    host_embs[cat] = torch.zeros((0, self.embedding_dim), dtype=torch.float32)
                    host_indices[cat] = torch.zeros((0,), dtype=torch.long)
        
        # === Categorical features ===
        host_meta = self.file_name_to_meta.get(host_file_name, None)
        host_catfeat = encode_categorical_onehot(host_meta, self.host_cats, self.host_cat2id)
        
        # 病毒的 categorical features
        virus_meta = self.virus_info_dict.get(virus_filename, None)
        if virus_meta is not None:
            virus_catfeat = encode_categorical_onehot(virus_meta, self.virus_cats, self.virus_cat2id)
        else:
            # 使用全零向量
            virus_catfeat = np.zeros(sum(self.virus_cat_dims), dtype=np.float32)
        
        extra_meta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        return host_embs, host_indices, virus_embs, virus_indices, host_catfeat, virus_catfeat, label, extra_meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="../UniVH-model-main/test",
                        help="Prediction working directory containing virus_host.csv, embedding/, HMM/ etc.")
    args = parser.parse_args()

    save_dir = "../UniVH-model-main/model_train"
    embed_dir = "../UniVH-model-main"
    timestamp = "20251205_095504"
    # pred_dir = "../UniVH-model-main/test"
    pred_dir = args.pred_dir
    data_dir = "../UniVH-model-main"
    pred_pair_csv = f"{pred_dir}/virus_host.csv"
    pred_virus_csv = f"{pred_dir}/HMM/combined_hmmscan_results_with_func.csv"
    vector_dir = f"{pred_dir}/embedding"

    out_csv = f"{pred_dir}/virus_host_pred_result_with_taxonomy.csv"
    
    embed_file = f"{embed_dir}/protein_embed_dict_20250912_010034.pkl"
    func_map_file = f"{embed_dir}/protein_func_map_20250912_010034.pkl"
    kegg_host_csv = f"{data_dir}/kegg_host_final.csv"
    host_info_file = f"{data_dir}/host_info_geo_env_0.9.csv"
    
    host_taxonomy_csv = f"{save_dir}/dataset.csv"
    
    possible_virus_info_files = [
        f"{pred_dir}/virus_info.csv"
    ]

    host_cats = ['AA','AN','AT','IM','NA','NT','OC','PA','Freshwater','Marine','Terrestrial']
    virus_cats = ['AA','AN','AT','IM','NA','NT','OC','PA','Freshwater','Marine','Terrestrial']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"{'Virus-Host Interaction Prediction':^80}")

    # ===== 加载模型配置 =====
    logger.info("\nLoading model configuration...")
    with open(os.path.join(save_dir, f"func_list_{timestamp}.pkl"), "rb") as f:
        func_list = pickle.load(f)
    with open(os.path.join(save_dir, f"major_classes_{timestamp}.pkl"), "rb") as f:
        major_classes = pickle.load(f)
    with open(os.path.join(save_dir, f"host_cat_dims_{timestamp}.pkl"), "rb") as f:
        host_cat_dims = pickle.load(f)
    with open(os.path.join(save_dir, f"virus_cat_dims_{timestamp}.pkl"), "rb") as f:
        virus_cat_dims = pickle.load(f)
    logger.info(f"Loaded func_list with {len(func_list)} functions")
    logger.info(f"Major classes: {len(major_classes)}")

    model_path = os.path.join(save_dir, f"best_model.pt")

    # ===== 加载宿主embeddings和function映射 =====
    logger.info("\nLoading host embeddings and function mappings...")
    with open(embed_file, 'rb') as f:
        host_embed_dict = pickle.load(f)
    with open(func_map_file, 'rb') as f:
        host_func_map = pickle.load(f)
    embedding_dim = next(iter(host_embed_dict.values())).shape[0]
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Host embeddings: {len(host_embed_dict)}")
    logger.info(f"Host function mappings: {len(host_func_map)}")

    # ===== 加载模型 =====
    logger.info("\nLoading model...")
    model = VirusHostClassifier(
        embedding_dim=embedding_dim,
        reduced_dim=256,
        major_classes=major_classes,
        func_list=func_list,
        host_cat_dims=host_cat_dims,
        virus_cat_dims=virus_cat_dims,
        dtype=torch.float32
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully")

    # ===== 加载宿主信息和categorical映射 =====
    logger.info("\nLoading host metadata...")
    host_info_df = pd.read_csv(host_info_file, dtype=str).fillna("NA")
    host_cat_values = {col: sorted(set(host_info_df[col].dropna().tolist() + ["NA"])) for col in host_cats}
    host_cat2id = {col: {v: i for i, v in enumerate(vals)} for col, vals in host_cat_values.items()}
    logger.info(f"Host metadata loaded: {len(host_info_df)} entries")

    # ===== 加载病毒信息和categorical映射 =====
    logger.info("\nAttempting to load virus metadata...")
    virus_info_dict = {}
    virus_cat2id = {col: {"NA": 0} for col in virus_cats}
    virus_info_loaded = False
    
    for virus_info_file in possible_virus_info_files:
        if os.path.exists(virus_info_file):
            try:
                logger.info(f"  Found virus info file: {virus_info_file}")
                virus_info_df = pd.read_csv(virus_info_file, dtype=str).fillna("NA")
                virus_cat_values = {col: sorted(set(virus_info_df[col].dropna().tolist() + ["NA"])) for col in virus_cats}
                virus_cat2id = {col: {v: i for i, v in enumerate(vals)} for col, vals in virus_cat_values.items()}
                
                if 'filename' in virus_info_df.columns:
                    virus_info_dict = virus_info_df.set_index('filename').to_dict(orient='index')
                elif 'SeqName' in virus_info_df.columns:
                    virus_info_dict = virus_info_df.set_index('SeqName').to_dict(orient='index')
                else:
                    virus_info_dict = virus_info_df.set_index(virus_info_df.columns[0]).to_dict(orient='index')
                
                logger.info(f"Virus metadata loaded: {len(virus_info_dict)} entries")
                virus_info_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {virus_info_file}: {e}")
                continue
    
    if not virus_info_loaded:
        logger.warning("No virus metadata file found")
        logger.warning("Will use ZERO VECTORS for virus categorical features")

    # ===== 加载病毒embeddings =====
    logger.info("\nLoading virus embeddings...")
    virus_embed_dict = load_pred_virus_embeddings(vector_dir)

    # ===== 加载病毒function映射 =====
    logger.info("\nLoading virus function mappings...")
    pred_virus_df = pd.read_csv(pred_virus_csv)
    virus_func_map = {}
    for _, row in pred_virus_df.iterrows():
        virus_func_map[row['query_name']] = row['target_name']
    filename2queries = pred_virus_df.groupby('filename')['query_name'].apply(list).to_dict()
    logger.info(f"Virus function mappings: {len(virus_func_map)}")

    # ===== 构建宿主映射 =====
    logger.info("\nBuilding host mappings...")
    host_proteins_by_file = build_host_proteins_by_file(kegg_host_csv, host_embed_dict)
    file_name_to_meta = get_host_meta_dict(kegg_host_csv, host_info_file)

    # ===== 读取预测对 =====
    logger.info(f"\nReading prediction pairs from {pred_pair_csv}...")
    pred_pairs_df = pd.read_csv(pred_pair_csv, sep=None, engine="python")
    logger.info(f"Total pairs to process: {len(pred_pairs_df)}")
    
    pairs = []
    virus_found, host_found = 0, 0
    
    for _, row in pred_pairs_df.iterrows():
        virus_field = row['SeqName.virus']
        host_field = row['SeqName.host']
        
        # 获取病毒蛋白IDs
        virus_query_names = filename2queries.get(virus_field, [])
        virus_protein_ids = [k for k in virus_embed_dict if k in virus_query_names]
        if not virus_protein_ids:
            logger.warning(f"No embeddings found for virus: {virus_field}")
            continue
        virus_found += 1

        # 获取宿主蛋白IDs
        host_protein_ids = host_proteins_by_file.get(host_field, [])
        if not host_protein_ids:
            logger.warning(f"No embeddings found for host: {host_field}")
            continue
        host_found += 1

        pairs.append((virus_protein_ids, host_protein_ids, 0, host_field, virus_field))

    logger.info(f"Valid virus samples: {virus_found}")
    logger.info(f"Valid host samples: {host_found}")
    logger.info(f"Final valid pairs: {len(pairs)}")
    
    if len(pairs) == 0:
        raise ValueError("No valid prediction pairs found! Please check your embedding files and input pair names.")

    # ===== 构建数据集 =====
    logger.info("\nBuilding prediction dataset...")
    pred_dataset = VirusHostPredDataset(
        pairs, host_embed_dict, virus_embed_dict, host_func_map, virus_func_map,
        func_list, file_name_to_meta, virus_info_dict, host_cats, virus_cats, 
        host_cat2id, virus_cat2id, host_cat_dims, virus_cat_dims, embedding_dim
    )

    # ===== 构建DataLoader =====
    def collate_fn_pred(batch):
        return collate_fn_gpu_cache(batch, device, host_cat_dims, virus_cat_dims)
    
    batch_size = 64
    pred_loader = torch.utils.data.DataLoader(
        pred_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_pred,
        pin_memory=False, 
        num_workers=0
    )
    logger.info(f"DataLoader created with batch_size={batch_size}")

    # ===== 预测 =====
    logger.info("\n" + "=" * 80)
    logger.info("Starting prediction...")
    logger.info("=" * 80)
    pred_probs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(pred_loader):
            host_embs, host_indices, virus_embs, virus_indices, host_cats_batch, virus_cats_batch, labels, extra_meta_tensor = batch

            logits = model(host_embs, host_indices, virus_embs, virus_indices, 
                          host_cats_batch, virus_cats_batch, extra_meta_tensor, device)
            
            probs = torch.sigmoid(logits)
            pred_probs.extend(probs.cpu().tolist())
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(pred_loader):
                logger.info(f"  Progress: {batch_idx + 1}/{len(pred_loader)} batches ({100*(batch_idx+1)/len(pred_loader):.1f}%)")
    
    logger.info("=" * 80)
    logger.info(f"✓ Prediction completed! Total predictions: {len(pred_probs)}")

    # ===== 构建预测结果 DataFrame =====
    logger.info("\nBuilding prediction results...")
    pred_pairs_valid_idx = []
    for idx, row in pred_pairs_df.iterrows():
        virus_field = row['SeqName.virus']
        host_field = row['SeqName.host']
        virus_query_names = filename2queries.get(virus_field, [])
        virus_protein_ids = [k for k in virus_embed_dict if k in virus_query_names]
        host_protein_ids = host_proteins_by_file.get(host_field, [])
        if virus_protein_ids and host_protein_ids:
            pred_pairs_valid_idx.append(idx)
    
    pred_pairs_valid = pred_pairs_df.iloc[pred_pairs_valid_idx].copy()
    pred_pairs_valid['pred_prob'] = pred_probs[:len(pred_pairs_valid)]
    logger.info(f"Total valid predictions: {len(pred_pairs_valid)}")
    
    # ===== 加载宿主分类信息并合并 =====
    logger.info("\n" + "=" * 80)
    logger.info("Merging host taxonomy information...")
    logger.info("=" * 80)
    
    host_taxonomy_df = load_host_taxonomy_info(host_taxonomy_csv)
    
    if not host_taxonomy_df.empty:
        logger.info(f"\nPerforming left join on 'SeqName.host' = 'host_filename'...")
        logger.info(f"  Prediction results: {len(pred_pairs_valid)} rows")
        logger.info(f"  Taxonomy data: {len(host_taxonomy_df)} unique host_filename entries")

        final_results = pred_pairs_valid.merge(
            host_taxonomy_df,
            left_on='SeqName.host',
            right_on='host_filename',
            how='left'
        )

        matched_count = final_results['host_taxid'].notna().sum() if 'host_taxid' in final_results.columns else 0
        unmatched_count = len(final_results) - matched_count
        
        logger.info(f"\nJoin results:")
        logger.info(f"Total rows: {len(final_results)}")
        logger.info(f"Matched with taxonomy: {matched_count} ({100*matched_count/len(final_results):.1f}%)")
        if unmatched_count > 0:
            logger.info(f"Unmatched: {unmatched_count} ({100*unmatched_count/len(final_results):.1f}%)")
        
    else:
        logger.warning("⚠ Taxonomy data not loaded. Saving results without taxonomy information.")
        final_results = pred_pairs_valid
    
    logger.info(f"\nSaving final results to: {out_csv}")
    final_results.to_csv(out_csv, index=False)
    logger.info(f"Results saved successfully!")
    logger.info(f"Total rows: {len(final_results)}")
    logger.info(f"Columns: {list(final_results.columns)}")

    logger.info(f"\nSample of final results (first 3 rows):")
    sample_cols = ['SeqName.virus', 'SeqName.host', 'pred_prob', 'host.species.name', 'host.genus', 'host.family']
    available_sample_cols = [col for col in sample_cols if col in final_results.columns]
    if available_sample_cols:
        logger.info(f"\n{final_results[available_sample_cols].head(3).to_string(index=False)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()

