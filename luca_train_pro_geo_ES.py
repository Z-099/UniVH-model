import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import pandas as pd
import numpy as np
import pickle
import time
import logging
import gc
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_curve, precision_recall_curve
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
mp.set_start_method('spawn', force=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def load_cat_dicts_and_lists(host_info_path, virus_info_path, host_cats, virus_cats):
    host_df = pd.read_csv(host_info_path, dtype=str).fillna("NA")
    virus_df = pd.read_csv(virus_info_path, dtype=str).fillna("NA")
    host_cat_values = {col: sorted(set(host_df[col].dropna().tolist() + ["NA"])) for col in host_cats}
    virus_cat_values = {col: sorted(set(virus_df[col].dropna().tolist() + ["NA"])) for col in virus_cats}
    host_cat2id = {col: {v: i for i, v in enumerate(vals)} for col, vals in host_cat_values.items()}
    virus_cat2id = {col: {v: i for i, v in enumerate(vals)} for col, vals in virus_cat_values.items()}
    host_info = host_df.set_index('SeqName').to_dict(orient='index')
    virus_info = virus_df.set_index('SeqName').to_dict(orient='index')
    host_cat_dims = [len(host_cat2id[col]) for col in host_cats]
    virus_cat_dims = [len(virus_cat2id[col]) for col in virus_cats]
    return host_info, virus_info, host_cat2id, virus_cat2id, host_cat_dims, virus_cat_dims

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

class VirusHostDataset_V1(Dataset):
    def __init__(
        self, pair_list, protein_embed_dict, protein_func_map, func_list,
        host_info_dict, virus_info_dict, host_cats, virus_cats, host_cat2id, virus_cat2id,
        host_emb_cache,
        metadata_list=None
    ):
        self.pair_list = pair_list
        self.protein_embed_dict = protein_embed_dict
        self.protein_func_map = protein_func_map
        self.func_list = func_list
        self.func_to_idx = {func: idx for idx, func in enumerate(func_list.keys())}
        self.host_cats = host_cats
        self.virus_cats = virus_cats
        self.host_cat2id = host_cat2id
        self.virus_cat2id = virus_cat2id
        self.host_info_dict = host_info_dict
        self.virus_info_dict = virus_info_dict
        self.metadata = metadata_list if metadata_list is not None else [None] * len(pair_list)

        embedding_dim = next(iter(self.protein_embed_dict.values())).shape[-1]
        self.major_classes = sorted([c for c in set().union(*[func_list[f] for f in func_list])])
        self.embedding_dim = embedding_dim

        self.host_emb_cache = host_emb_cache
        self.host_precomputed = {}
        for i, (_, host_ids, _) in enumerate(pair_list):
            key = tuple(sorted(set(host_ids)))
            if key not in self.host_emb_cache:
                logger.error(f"Missing host_emb_cache for key: {key}")
                host_embs = {cat: torch.zeros((0, self.embedding_dim), dtype=torch.float32) for cat in self.major_classes if cat.startswith('KEGG')}
                host_indices = {cat: torch.zeros((0,), dtype=torch.long) for cat in self.major_classes if cat.startswith('KEGG')}
                self.host_precomputed[i] = (host_embs, host_indices)
            else:
                self.host_precomputed[i] = self.host_emb_cache[key]

        self.samples = []
        for i, (virus_ids, host_ids, label) in enumerate(pair_list):
            virus_map = defaultdict(list)
            virus_indices_map = defaultdict(list)
            for pid in virus_ids:
                if pid in self.protein_embed_dict and pid in self.protein_func_map:
                    func = self.protein_func_map[pid]
                    if func not in self.func_list:
                        continue
                    emb = self.protein_embed_dict[pid]
                    categories = self.func_list[func] if isinstance(self.func_list[func], (list, set)) else [self.func_list[func]]
                    for major_class in categories:
                        if isinstance(emb, torch.Tensor):
                            virus_map[major_class].append(emb.clone().detach().to(torch.float32))
                        else:
                            virus_map[major_class].append(torch.tensor(emb, dtype=torch.float32))
                        virus_indices_map[major_class].append(self.func_to_idx[func])
            virus_embs = {}
            virus_indices = {}
            for cat in self.major_classes:
                if not cat.startswith('KEGG'):
                    embs = virus_map[cat]
                    virus_embs[cat] = torch.stack(embs) if len(embs) > 0 else torch.zeros((0, embedding_dim), dtype=torch.float32)
                    virus_indices[cat] = torch.tensor(virus_indices_map[cat], dtype=torch.long) if len(embs) > 0 else torch.zeros((0,), dtype=torch.long)

            meta_row = self.metadata[i] if self.metadata else None
            host_taxid = str(meta_row["host_taxid"]) if meta_row and "host_taxid" in meta_row else None
            virus_taxid = str(meta_row["virus_taxid"]) if meta_row and "virus_taxid" in meta_row else None
            host_filename = meta_row['host_filename'] if meta_row and 'host_filename' in meta_row else None
            virus_filename = meta_row['virus_filename'] if meta_row and 'virus_filename' in meta_row else None

            host_row = self.host_info_dict.get(host_taxid, None)
            virus_row = self.virus_info_dict.get(virus_taxid, None)
            host_row = self.host_info_dict.get(host_filename, host_row)
            virus_row = self.virus_info_dict.get(virus_filename, virus_row)

            host_catfeat = encode_categorical_onehot(host_row, self.host_cats, self.host_cat2id)
            virus_catfeat = encode_categorical_onehot(virus_row, self.virus_cats, self.virus_cat2id)
            host_embs, host_indices = self.host_precomputed[i]

            mean_distance = float(meta_row.get("mean_distance", 0.0)) if meta_row else 0.0
            overlap_virus = float(meta_row.get("overlap_virus", 0.0)) if meta_row else 0.0
            overlap_host = float(meta_row.get("overlap_host", 0.0)) if meta_row else 0.0

            self.samples.append((
                host_embs, host_indices, virus_embs, virus_indices, host_catfeat, virus_catfeat, label, np.array([mean_distance, overlap_virus, overlap_host], dtype=np.float32)
            ))

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

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

def evaluate_dataset_full_metrics(model, dataloader, criterion, device, split_name="Validation", plot_dir=None):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    raw_preds, binary_preds, gts = [], [], []
    with torch.no_grad():
        for host_maps, host_indices, virus_maps, virus_indices, host_cats, virus_cats, labels_tensor, extra_meta_tensor in dataloader:
            logits = model(host_maps, host_indices, virus_maps, virus_indices, host_cats, virus_cats, extra_meta_tensor, device)
            loss = criterion(logits, labels_tensor)
            total_loss += loss.item() * len(labels_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            correct += (preds == labels_tensor.long()).sum().item()
            n += len(labels_tensor)
            raw_preds.extend(probs.cpu().tolist())
            binary_preds.extend(preds.cpu().tolist())
            gts.extend([int(l) for l in labels_tensor])
    avg_loss = (total_loss / n) if n else 0
    acc = (100 * correct / n) if n else 0
    auc = roc_auc_score(gts, raw_preds) if len(set(gts)) > 1 else float('nan')
    f1 = f1_score(gts, binary_preds, zero_division=0)
    precision = precision_score(gts, binary_preds, zero_division=0)
    recall = recall_score(gts, binary_preds, zero_division=0)
    bal_acc = balanced_accuracy_score(gts, binary_preds)
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        if len(set(gts)) > 1:
            fpr, tpr, _ = roc_curve(gts, raw_preds)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{split_name} ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plot_dir, f'{split_name}_roc_curve.png'))
            plt.close()
        precision_arr, recall_arr, _ = precision_recall_curve(gts, raw_preds)
        plt.figure()
        plt.plot(recall_arr, precision_arr, label=f'PR curve (F1 = {f1:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{split_name} Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(plot_dir, f'{split_name}_pr_curve.png'))
        plt.close()
    return avg_loss, acc, auc, f1, precision, recall, bal_acc

def train_model(model, train_dataset, val_dataset=None, test_dataset=None, epochs=50, batch_size=256, lr=1e-4,
                host_cat_dims=None, virus_cat_dims=None, save_dir=None,
                early_stop_patience=5, early_stop_metric='val_acc', early_stop_mode='max', min_delta=0.0001):
    device = next(model.parameters()).device
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_fn_gpu_cache(b, device, host_cat_dims, virus_cat_dims),
                                  pin_memory=False, num_workers=0, prefetch_factor=2)
    val_dataloader = (DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda b: collate_fn_gpu_cache(b, device, host_cat_dims, virus_cat_dims),
                                 pin_memory=False, num_workers=0, prefetch_factor=2)
                      if val_dataset else None)
    test_dataloader = (DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=lambda b: collate_fn_gpu_cache(b, device, host_cat_dims, virus_cat_dims),
                                  pin_memory=False, num_workers=0, prefetch_factor=2)
                       if test_dataset else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_metric = None
    counter = 0
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for host_maps, host_indices, virus_maps, virus_indices, host_cats, virus_cats, labels_tensor, extra_meta_tensor in train_dataloader:
            optimizer.zero_grad()
            logits = model(host_maps, host_indices, virus_maps, virus_indices, host_cats, virus_cats, extra_meta_tensor, device)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels_tensor)
            pred = (torch.sigmoid(logits) > 0.5).long()
            correct += (pred == labels_tensor.long()).sum().item()
            n += len(labels_tensor)
        train_loss = total_loss / n if n else 0
        train_acc = 100 * correct / n if n else 0
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        if val_dataloader:
            val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_bal_acc = evaluate_dataset_full_metrics(
                model, val_dataloader, criterion, device, split_name="Validation", plot_dir=save_dir)
            logger.info(
                f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | "
                f"Balanced Acc: {val_bal_acc:.4f}"
            )

            # Early Stopping logic, log best_epoch, best_metric
            if early_stop_metric == 'val_loss':
                current_metric = val_loss
            elif early_stop_metric == 'val_auc':
                current_metric = val_auc
            elif early_stop_metric == 'val_acc':
                current_metric = val_acc
            else:
                current_metric = val_loss

            if best_metric is None or \
                ((early_stop_mode == 'min' and current_metric < best_metric - min_delta) or
                 (early_stop_mode == 'max' and current_metric > best_metric + min_delta)):
                best_metric = current_metric
                counter = 0
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                logger.info(f"Best {early_stop_metric} improved to {best_metric:.6f} at epoch {best_epoch}, model saved.")
            else:
                counter += 1
                logger.info(f"Early stop counter: {counter}/{early_stop_patience}")
            if counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best {early_stop_metric}: {best_metric} at epoch {best_epoch}")
                break

        if test_dataloader:
            test_loss, test_acc, test_auc, test_f1, test_precision, test_recall, test_bal_acc = evaluate_dataset_full_metrics(
                model, test_dataloader, criterion, device, split_name="Test", plot_dir=save_dir)
            logger.info(
                f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
                f"AUC: {test_auc:.4f} | F1: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | "
                f"Balanced Acc: {test_bal_acc:.4f}"
            )
        gc.collect()
        torch.cuda.empty_cache()

def save_val_predictions_v1(val_dataset, model, batch_size, host_cat_dims, virus_cat_dims, output_file):
    device = next(model.parameters()).device
    dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=lambda b: collate_fn_gpu_cache(b, device, host_cat_dims, virus_cat_dims),
                            pin_memory=False, num_workers=0)
    model.eval()
    probs = []
    with torch.no_grad():
        for host_maps, host_indices, virus_maps, virus_indices, host_cats, virus_cats, labels, extra_meta_tensor in dataloader:
            logits = model(host_maps, host_indices, virus_maps, virus_indices, host_cats, virus_cats, extra_meta_tensor, device)
            batch_probs = torch.sigmoid(logits).cpu().tolist()
            probs.extend(batch_probs)
    df = pd.DataFrame(val_dataset.metadata)
    if len(probs) != len(df):
        print(f"Warning: prediction count {len(probs)} != metadata count {len(df)}")
        df = df.iloc[:len(probs)].copy()
    df["prob"] = probs[:len(df)]
    df[["virus_filename","host_filename","virus_taxid","host_taxid","label","mean_distance","overlap_virus","overlap_host","prob"]].to_csv(output_file, index=False)
    print(f"Saved val predictions with probs to {output_file}")

def build_func_list_and_major_classes(kegg_host_df, virus_df):
    func_list = defaultdict(set)
    for _, row in kegg_host_df.iterrows():
        func_list[row['KO']].add(row['host_category'])
    for _, row in virus_df.iterrows():
        func_list[row['target_name']].add(row['virus_category'])
    func_list_final = {k: list(v) for k, v in func_list.items()}
    host_categories = set()
    virus_categories = set()
    for cats in func_list.values():
        for cat in cats:
            if cat.startswith('KEGG'):
                host_categories.add(cat)
            else:
                virus_categories.add(cat)
    major_classes = sorted(host_categories) + sorted(virus_categories)
    return func_list_final, major_classes

if __name__ == "__main__":
    batch_name='1016_allcluster_geo_seed45_ratio3'
    data_dir = '/data/zd/model_luca/data'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    data_paths = {
        'kegg_host': '%s/kegg_host_final.csv' % data_dir,
        'virus': '%s/combined_hmmscan_results_with_func.csv' % data_dir,
        'labeled_data': '%s/1016_allcluster_model_dataset_ratio.csv'%data_dir,
        'host_info': '%s/host_info_geo_env_0.9.csv' % data_dir,
        'virus_info': '%s/virus_fea_realm_env.csv' % data_dir
    }
    save_dir = "/data/zd/model_luca/20250911/%s" % batch_name
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(45)
    torch.manual_seed(45)
    np.random.seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(45)

    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    host_cats = ['AA','AN','AT','IM','NA','NT','OC','PA','Freshwater','Marine','Terrestrial']
    virus_cats = ['AA','AN','AT','IM','NA','NT','OC','PA','Freshwater','Marine','Terrestrial']

    host_info_dict, virus_info_dict, host_cat2id, virus_cat2id, host_cat_dims, virus_cat_dims = load_cat_dicts_and_lists(
        data_paths['host_info'], data_paths['virus_info'], host_cats, virus_cats
    )

    kegg_host_df = pd.read_csv(data_paths['kegg_host'])
    kegg_host_df = kegg_host_df[kegg_host_df['longest_KO'] == 1]
    virus_df = pd.read_csv(data_paths['virus'])
    labeled_df = pd.read_csv(data_paths['labeled_data'])

    labeled_df = labeled_df.replace('NA', pd.NA)
    labeled_df = labeled_df.dropna(subset=['mean_distance', 'overlap_virus', 'overlap_host'])
    labeled_df['mean_distance'] = labeled_df['mean_distance'].astype(float)
    labeled_df['overlap_virus'] = labeled_df['overlap_virus'].astype(float)
    labeled_df['overlap_host'] = labeled_df['overlap_host'].astype(float)

    scaler = MinMaxScaler()
    labeled_df[['mean_distance', 'overlap_virus', 'overlap_host']] = scaler.fit_transform(
        labeled_df[['mean_distance', 'overlap_virus', 'overlap_host']]
    )

    logger.info(f"Loaded {len(labeled_df)} labeled samples after filtering NA and normalization.")

    func_list, major_classes = build_func_list_and_major_classes(kegg_host_df, virus_df)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(save_dir, f"func_list_{timestamp}.pkl"), "wb") as f:
        pickle.dump(func_list, f)
    with open(os.path.join(save_dir, f"major_classes_{timestamp}.pkl"), "wb") as f:
        pickle.dump(major_classes, f)
    with open(os.path.join(save_dir, f"host_cat_dims_{timestamp}.pkl"), "wb") as f:
        pickle.dump(host_cat_dims, f)
    with open(os.path.join(save_dir, f"virus_cat_dims_{timestamp}.pkl"), "wb") as f:
        pickle.dump(virus_cat_dims, f)

    embed_file = "/data/zd/model_luca/data/protein_embed_dict_20250912_010034.pkl"
    func_map_file = "/data/zd/model_luca/data/protein_func_map_20250912_010034.pkl"
    if not os.path.exists(embed_file) or not os.path.exists(func_map_file):
        logger.error("Missing embedding or function mapping file.")
        exit()
    with open(embed_file, 'rb') as f:
        protein_embed_dict = pickle.load(f)
    with open(func_map_file, 'rb') as f:
        protein_func_map = pickle.load(f)
    logger.info(f"Loaded {len(protein_embed_dict)} protein embeddings.")

    virus_proteins_by_file = defaultdict(list)
    host_proteins_by_file = defaultdict(list)
    for _, row in virus_df.iterrows():
        if row['query_name'] in protein_embed_dict:
            virus_proteins_by_file[(row['filename'], row['species_id'])].append(row['query_name'])
    for _, row in kegg_host_df.iterrows():
        if row['sequence'] in protein_embed_dict:
            host_proteins_by_file[(row['file_name'], row['taxid'])].append(row['sequence'])
    del kegg_host_df, virus_df
    gc.collect()

    all_data = labeled_df

    train_df = all_data[all_data['dataset'] == 'train']
    val_df   = all_data[all_data['dataset'] == 'val']
    test_df  = all_data[all_data['dataset'] == 'test']

    def read_pairs_df_with_metadata(df, virus_proteins_by_file, host_proteins_by_file):
        pairs = []
        metadata_list = []
        for _, row in df.iterrows():
            virus_ids = []
            host_ids = []
            virus_filename, virus_taxid = row.get('virus_filename', None), row.get('virus_taxid', None)
            host_filename, host_taxid = row.get('host_filename', None), row.get('host_taxid', None)
            for k, v in virus_proteins_by_file.items():
                if str(k[0]) == str(virus_filename) and str(k[1]) == str(virus_taxid):
                    virus_ids = v
                    break
            for k, v in host_proteins_by_file.items():
                if str(k[0]) == str(host_filename) and str(k[1]) == str(host_taxid):
                    host_ids = v
                    break
            label = int(row['label'])
            if virus_ids and host_ids:
                pairs.append((virus_ids, host_ids, label))
                metadata_list.append({
                    "virus_filename": virus_filename,
                    "host_filename": host_filename,
                    "virus_taxid": virus_taxid,
                    "host_taxid": host_taxid,
                    "label": label,
                    "mean_distance": float(row.get("mean_distance", 0.0)),
                    "overlap_virus": float(row.get("overlap_virus", 0.0)),
                    "overlap_host": float(row.get("overlap_host", 0.0))
                })
        return pairs, metadata_list

    train_pairs, train_metadata = read_pairs_df_with_metadata(train_df, virus_proteins_by_file, host_proteins_by_file)
    val_pairs, val_metadata     = read_pairs_df_with_metadata(val_df, virus_proteins_by_file, host_proteins_by_file)
    test_pairs, test_metadata   = read_pairs_df_with_metadata(test_df, virus_proteins_by_file, host_proteins_by_file)

    random.shuffle(train_pairs)
    random.shuffle(train_metadata)
    gc.collect()

    all_pairs = train_pairs + val_pairs + test_pairs
    unique_host_ids = {}
    for _, host_ids, _ in all_pairs:
        key = tuple(sorted(set(host_ids)))
        if key not in unique_host_ids:
            unique_host_ids[key] = host_ids

    host_emb_cache_path = "/data/zd/model_luca/data/host_precomputed_all.pkl"
    if os.path.exists(host_emb_cache_path):
        with open(host_emb_cache_path, "rb") as f:
            host_emb_cache = pickle.load(f)
    else:
        host_emb_cache = {}
        embedding_dim = next(iter(protein_embed_dict.values())).shape[-1]
        func_to_idx = {func: idx for idx, func in enumerate(func_list.keys())}
        for key, host_ids in unique_host_ids.items():
            host_map = defaultdict(list)
            host_indices_map = defaultdict(list)
            for pid in host_ids:
                if pid in protein_embed_dict and pid in protein_func_map:
                    func = protein_func_map[pid]
                    if func not in func_list:
                        continue
                    emb = protein_embed_dict[pid]
                    categories = func_list[func] if isinstance(func_list[func], (list, set)) else [func_list[func]]
                    for major_class in categories:
                        if isinstance(emb, torch.Tensor):
                            host_map[major_class].append(emb.clone().detach().to(torch.float32))
                        else:
                            host_map[major_class].append(torch.tensor(emb, dtype=torch.float32))
                        host_indices_map[major_class].append(func_to_idx[func])
            host_embs = {}
            host_indices = {}
            for cat in major_classes:
                if cat.startswith('KEGG'):
                    embs = host_map[cat]
                    host_embs[cat] = torch.stack(embs) if len(embs) > 0 else torch.zeros((0, embedding_dim), dtype=torch.float32)
                    host_indices[cat] = torch.tensor(host_indices_map[cat], dtype=torch.long) if len(embs) > 0 else torch.zeros((0,), dtype=torch.long)
            host_emb_cache[key] = (host_embs, host_indices)
        with open(host_emb_cache_path, "wb") as f:
            pickle.dump(host_emb_cache, f)

    if train_pairs and val_pairs and test_pairs:
        embedding_dim = next(iter(protein_embed_dict.values())).shape[0]
        model = VirusHostClassifier(
            embedding_dim=embedding_dim,
            reduced_dim=256,
            major_classes=major_classes,
            func_list=func_list,
            host_cat_dims=host_cat_dims,
            virus_cat_dims=virus_cat_dims,
            dtype=torch.float32
        )

        model.to(device)
        batch_size = 128

        logger.info("Batch size: %s", batch_size)

        train_dataset = VirusHostDataset_V1(
            train_pairs, protein_embed_dict, protein_func_map, func_list,
            host_info_dict, virus_info_dict, host_cats, virus_cats, host_cat2id, virus_cat2id,
            host_emb_cache,
            metadata_list=train_metadata)
        
        val_dataset = VirusHostDataset_V1(
            val_pairs, protein_embed_dict, protein_func_map, func_list,
            host_info_dict, virus_info_dict, host_cats, virus_cats, host_cat2id, virus_cat2id,
            host_emb_cache,
            metadata_list=val_metadata)
        
        test_dataset = VirusHostDataset_V1(
            test_pairs, protein_embed_dict, protein_func_map, func_list,
            host_info_dict, virus_info_dict, host_cats, virus_cats, host_cat2id, virus_cat2id,
            host_emb_cache,
            metadata_list=test_metadata)

        train_model(model, train_dataset, val_dataset, test_dataset, epochs=30, batch_size=batch_size,
                    host_cat_dims=host_cat_dims, virus_cat_dims=virus_cat_dims, save_dir=save_dir,
                    early_stop_patience=8, early_stop_metric='val_acc', early_stop_mode='max', min_delta=0.0001)
        logger.info("Evaluating model on test set...")

        criterion = nn.BCEWithLogitsLoss()
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn_gpu_cache(b, next(model.parameters()).device, host_cat_dims, virus_cat_dims),
            pin_memory=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn_gpu_cache(b, next(model.parameters()).device, host_cat_dims, virus_cat_dims),
            pin_memory=False, num_workers=0)

        # 加载最佳模型再评估test
        best_model_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best model from {best_model_path} for test evaluation.")
        else:
            logger.warning(f"Best model {best_model_path} not found, using current model for test.")

        val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_bal_acc = evaluate_dataset_full_metrics(
            model, val_dataloader, criterion, device, split_name="Validation", plot_dir=save_dir)
        test_loss, test_acc, test_auc, test_f1, test_precision, test_recall, test_bal_acc = evaluate_dataset_full_metrics(
            model, test_dataloader, criterion, device, split_name="Test", plot_dir=save_dir)

        logger.info(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | AUC: {test_auc:.4f} | F1: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | Balanced Acc: {test_bal_acc:.4f}")
        
        model_path = os.path.join(save_dir, f"virus_host_model_{timestamp}.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        func_keys = list(func_list.keys())
        func_weights = model.func_weights.detach().cpu().tolist()
        csv_path = os.path.join(save_dir, f"func_weights_and_list_{timestamp}.csv")
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["func_name", "func_weight", "func_category"])
            for i, func_name in enumerate(func_keys):
                writer.writerow([func_name, func_weights[i], func_list[func_name]])
        print(f"Func weights and list saved to {csv_path}")

        save_val_predictions_v1(
            val_dataset,
            model,
            batch_size,
            host_cat_dims,
            virus_cat_dims,
            os.path.join(save_dir, f"val_pred_with_prob_{timestamp}.csv")
        )

        save_val_predictions_v1(
            test_dataset,
            model,
            batch_size,
            host_cat_dims,
            virus_cat_dims,
            os.path.join(save_dir, f"test_pred_with_prob_{timestamp}.csv")
        )

    else:
        logger.error("Insufficient data for training.") 