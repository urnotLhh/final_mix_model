import yaml
import logging
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import sys
from pathlib import Path
import time
from tqdm import tqdm # 导入tqdm以显示进度条
import random # 新增导入，用于负采样
from utils import clean_banner # 导入清洗函数

# --- 日志配置 ---
def setup_logging(log_path):
    """配置日志，只输出到文件，避免与tqdm冲突"""
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            # logging.StreamHandler(sys.stdout) # 移除控制台输出，防止刷屏
        ]
    )

# --- 数据集定义 ---
class TripletDataset(Dataset):
    """
    为 mix v2.4 创建三元组的PyTorch Dataset。
    - 为训练集实现增强的困难负采样，为每个正样本生成多个困难负样本。
    - 为验证/测试集保留标准的、每次一个的负采样逻辑。
    """
    def __init__(self, data_pairs, relation_name, all_devices, config, mode='train'):
        self.data_pairs = data_pairs
        self.relation_name = relation_name
        self.all_devices = all_devices
        self.config = config
        self.mode = mode
        self.relation2id = {relation_name: 0}
        
        # --- 为分层困难负采样构建查找表 (v2.3) ---
        self._build_lookup_tables(all_devices)

        # --- (v2.4) 基于模式创建训练实例 ---
        if self.mode == 'train':
            self.triplets = self._create_augmented_triplets(data_pairs)
        else:
            # 对于 eval/test 模式，我们不预先生成三元组
            # 负采样将在 __getitem__ 中即时进行
            pass

    def _build_lookup_tables(self, all_devices):
        """构建用于快速查找设备的辅助表"""
        self.device_to_community = {} # device -> (type, vendor)
        self.community_to_devices = {} # (type, vendor) -> [devices]
        self.type_to_devices = {}      # type -> [devices]
        
        for device in all_devices:
            try:
                parts = [p.strip() for p in device.split(',')]
                d_type = parts[0].split(':')[1].strip()
                vendor = parts[1].split(':')[1].strip()
                community_key = (d_type, vendor)
                
                self.device_to_community[device] = community_key
                if community_key not in self.community_to_devices:
                    self.community_to_devices[community_key] = []
                self.community_to_devices[community_key].append(device)

                if d_type not in self.type_to_devices:
                    self.type_to_devices[d_type] = []
                self.type_to_devices[d_type].append(device)
            except IndexError:
                logging.warning(f"Skipping malformed device string: {device}")
                continue
    
    def _create_augmented_triplets(self, data_pairs):
        """(v2.4) 为训练集创建增强的三元组列表"""
        logging.info("--- (v2.4) Creating augmented triplets for training set... ---")
        final_triplets = []
        max_hard_negatives = self.config['data_processing'].get('max_hard_negatives_per_positive', 3)

        for banner_text, positive_device_text in tqdm(data_pairs, desc="Augmenting Triplets"):
            # 1. 寻找所有 Level 1 (Hardest) 负样本
            hard_negatives = self._find_hard_negatives(positive_device_text)
            
            if hard_negatives:
                # 如果找到，为每个困难负样本创建一个三元组（有上限）
                selected_negatives = random.sample(hard_negatives, min(len(hard_negatives), max_hard_negatives))
                for neg_dev in selected_negatives:
                    final_triplets.append((banner_text, positive_device_text, neg_dev))
            else:
                # 2. 如果没找到，则用回退策略生成一个普通负样本
                fallback_negative = self._get_fallback_negative_sample(positive_device_text)
                final_triplets.append((banner_text, positive_device_text, fallback_negative))
        
        logging.info(f"Augmented training set: Generated {len(final_triplets)} triplets from {len(data_pairs)} base pairs.")
        random.shuffle(final_triplets) # 打乱增强后的数据集
        return final_triplets

    def _find_hard_negatives(self, positive_device_text):
        """寻找Level 1 (相同type, 相同vendor) 的负样本"""
        community_key = self.device_to_community.get(positive_device_text)
        if not community_key or len(self.community_to_devices.get(community_key, [])) <= 1:
            return []
        return [dev for dev in self.community_to_devices[community_key] if dev != positive_device_text]

    def _get_fallback_negative_sample(self, positive_device_text):
        """分层查找一个负样本 (用于无困难样本时 或 eval模式)"""
        # Level 2: Medium (相同 type, 不同 vendor)
        positive_community_key = self.device_to_community.get(positive_device_text)
        if positive_community_key:
            positive_type = positive_community_key[0]
            possible_negatives = [
                dev for dev in self.type_to_devices.get(positive_type, []) 
                if self.device_to_community.get(dev) and self.device_to_community[dev][1] != positive_community_key[1]
            ]
            if possible_negatives:
                return random.choice(possible_negatives)

        # Level 3: Easy (完全随机)
        negative_device_text = random.choice(self.all_devices)
        while negative_device_text == positive_device_text:
            negative_device_text = random.choice(self.all_devices)
        return negative_device_text

    def __len__(self):
        return len(self.triplets) if self.mode == 'train' else len(self.data_pairs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            banner_text, positive_device_text, negative_device_text = self.triplets[idx]
        else: # eval / test 模式
            banner_text, positive_device_text = self.data_pairs[idx]
            negative_device_text = self._get_fallback_negative_sample(positive_device_text)

        return {
            "banner_text": banner_text,
            "relation_id": torch.tensor(self.relation2id[self.relation_name], dtype=torch.long),
            "triple_text": positive_device_text,
            "negative_triple_text": negative_device_text
        }

# --- 数据加载与预处理主函数 ---
def load_and_prepare_data(config):
    """
    为 mix_0719 v2 方案加载和准备数据。
    v2.1: 修正数据划分逻辑，采用全局随机划分，而不是按设备划分。
    """
    logging.info("--- (mix_0719 v2.1) 开始加载和预处理数据 (全局随机划分) ---")
    
    with open(config['paths']['input_data'], 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    logging.info(f"成功加载 {len(raw_data)} 条原始记录。")

    # 构建实体、关系词汇表和三元组列表
    all_pairs = []
    unique_devices = set()
    template = config['data_processing']['entity_text_template']
    
    for item in raw_data:
        dtype = item.get("device_type")
        vendor = item.get("true_vendor")
        model = item.get("true_product")

        if not all([dtype, vendor, model]):
            continue

        # 将设备三元组转换为描述性文本
        triple_text = template.format(type=dtype, vendor=vendor, model=model)
        unique_devices.add(triple_text)
        
        # 直接创建 (banner, device_text) 对，并清洗banner
        cleaned_banner = clean_banner(item['banner'])
        all_pairs.append((cleaned_banner, triple_text))
        
    logging.info(f"处理后剩余 {len(all_pairs)} 个有效的 Banner-设备对。")
    logging.info(f"发现 {len(unique_devices)} 个独特的设备描述文本。")

    # --- 新的数据划分逻辑 ---
    logging.info("对所有数据对进行全局随机划分...")
    
    # 1. 全局随机打乱
    random.shuffle(all_pairs)
    
    # 2. 按比例切分
    dev_proc = config['data_processing']
    train_ratio = dev_proc['train_ratio']
    valid_ratio = dev_proc['valid_ratio']
    
    train_end = int(len(all_pairs) * train_ratio)
    valid_end = train_end + int(len(all_pairs) * valid_ratio)
    
    train_data = all_pairs[:train_end]
    valid_data = all_pairs[train_end:valid_end]
    test_data = all_pairs[valid_end:]

    logging.info(f"数据集划分完成: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")

    # 关系词汇表 (在v2模型中，关系是固定的)
    relation_name = config['data_processing']['relation_name']
    relation2id = {relation_name: 0}
    
    # 为数据创建Dataset对象，并区分模式
    train_ds = TripletDataset(train_data, relation_name, list(unique_devices), config, mode='train')
    valid_ds = TripletDataset(valid_data, relation_name, list(unique_devices), config, mode='eval')
    test_ds = TripletDataset(test_data, relation_name, list(unique_devices), config, mode='eval')
    
    return train_ds, valid_ds, test_ds, list(unique_devices), relation2id

# --- 联合模型定义 ---
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.optim as optim

class JointModel(nn.Module):
    """
    一个端到端的联合模型 (v2.2)，将SBERT和TransE结合在一起。
    这个版本增加了更深的全连接层来增强模型容量。
    """
    def __init__(self, config, num_relations, all_device_texts):
        super(JointModel, self).__init__()
        self.config = config
        self.model_config = config['joint_model']
        
        # 1. 加载SBERT模型
        logging.info(f"从 {config['paths']['sbert_model_path']} 加载SBERT模型...")
        self.sbert = SentenceTransformer(config['paths']['sbert_model_path'])
        
        if not self.model_config['finetune_sbert']:
            for param in self.sbert.parameters():
                param.requires_grad = False
        
        # --- 核心改造：增加更深的全连接层 ---
        sbert_dim = self.model_config['sbert_embedding_dim']
        entity_dim = self.model_config['entity_embedding_dim']
        hidden_dim = self.model_config.get('hidden_dim', 512) # 从配置读取隐藏层维度
        
        self.projection_layer = nn.Sequential(
            nn.Linear(sbert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, entity_dim)
        )

        # 2. 保留关系嵌入层
        self.relation_embeddings = nn.Embedding(num_relations, entity_dim)
        self.margin = self.model_config['margin']
        self.all_device_texts = all_device_texts
        self._init_weights()

    def _init_weights(self):
        """初始化关系嵌入层和新的全连接层"""
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        # 对新的全连接层进行初始化
        for layer in self.projection_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0)

    def encode_text(self, texts, device):
        """使用SBERT和投影层来编码文本"""
        sbert_embeddings = self.sbert.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=False)
        projected_embeddings = self.projection_layer(sbert_embeddings)
        return projected_embeddings

    def forward(self, batch):
        device = self.relation_embeddings.weight.device
        
        h = self.encode_text(batch['banner_text'], device)
        t_pos = self.encode_text(batch['triple_text'], device)
        t_neg = self.encode_text(batch['negative_triple_text'], device)
        
        r = self.relation_embeddings(batch['relation_id'].to(device))
        
        pos_score = torch.norm(h + r - t_pos, p=self.model_config.get('norm', 2), dim=1)
        neg_score = torch.norm(h + r - t_neg, p=self.model_config.get('norm', 2), dim=1)
        
        return pos_score, neg_score

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip):
    """执行一个完整的训练轮次 (v2.1)"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 直接将整个batch传递给模型，模型内部处理设备移动
        pos_score, neg_score = model(batch)
        
        y = torch.ones_like(pos_score) * -1
        loss = loss_fn(pos_score, neg_score, y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    """在验证集上评估模型 (v2.1)"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # 直接将整个batch传递给模型，模型内部处理设备移动
            pos_score, neg_score = model(batch)
            
            y = torch.ones_like(pos_score) * -1
            loss = loss_fn(pos_score, neg_score, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    """主函数 - 包含完整的 mix v2 训练流程"""
    config_path = Path("mix_0719/0_configs/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    setup_logging(Path(config['paths']['log_file']))
    
    train_ds, valid_ds, _, all_device_texts, relation2id = load_and_prepare_data(config)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    # --- 模型初始化 (v2) ---
    logging.info("\n--- 初始化联合模型 (v2) ---")
    model = JointModel(
        config=config,
        num_relations=len(relation2id),
        all_device_texts=all_device_texts
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已移动到设备: {device}")

    # --- 配置优化器 (v2.2) ---
    sbert_params = list(model.sbert.parameters())
    projection_params = list(model.projection_layer.parameters())
    graph_params = list(model.relation_embeddings.parameters())
    
    optimizer = optim.AdamW([
        {'params': sbert_params, 'lr': config['training']['sbert_lr']},
        {'params': projection_params, 'lr': config['training']['projection_lr']}, # 为新层增加学习率
        {'params': graph_params, 'lr': config['training']['graph_lr']}
    ])
    
    loss_fn = nn.MarginRankingLoss(margin=model.margin)
    print("成功创建模型、优化器和损失函数。")

    # --- 训练循环 ---
    print("\n" + "="*20 + " 开始训练 (mix v2) " + "="*20)
    
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    best_model_path = output_dir / "best_model_v2.pth"

    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config['training']['grad_clip'])
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        
        epoch_summary = f"Epoch: {epoch+1:02} | Time: {epoch_mins}m | Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f}"
        print(epoch_summary)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"验证损失提升，模型已保存到: {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"验证损失未提升。连续未提升轮次: {epochs_no_improve}")
        
        if epochs_no_improve >= config['training']['patience']:
            print(f"连续 {config['training']['patience']} 轮验证损失未提升，触发早停。")
            break
            
    print("\n" + "="*20 + " 训练结束 " + "="*20)
    print(f"最佳验证损失: {best_valid_loss:.4f}")
    print(f"最佳模型已保存在: {best_model_path}")

if __name__ == '__main__':
    main() 