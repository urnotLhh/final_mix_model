import yaml
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from pathlib import Path
import logging
import random
from tqdm import tqdm
import sys

# 将训练目录添加到sys.path以导入clean_banner
# 这是一种常见的解决方法，当脚本和它依赖的模块在不同的、非标准的目录结构中
sys.path.append(str(Path(__file__).resolve().parents[1] / '2_training'))
from utils import clean_banner


# --- 日志和设备配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型定义 (与 predict.py 保持一致) ---
class JointModel(nn.Module):
    def __init__(self, config, num_relations=1, all_device_texts=None):
        super(JointModel, self).__init__()
        self.config = config
        self.model_config = config['joint_model']
        self.sbert = SentenceTransformer(config['paths']['sbert_model_path'])
        sbert_dim = self.model_config['sbert_embedding_dim']
        entity_dim = self.model_config['entity_embedding_dim']
        hidden_dim = self.model_config.get('hidden_dim', 512)
        self.projection_layer = nn.Sequential(
            nn.Linear(sbert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, entity_dim)
        )
        self.relation_embeddings = nn.Embedding(num_relations, entity_dim)

    def encode_text(self, texts, device):
        sbert_embeddings = self.sbert.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=False)
        projected_embeddings = self.projection_layer(sbert_embeddings)
        return projected_embeddings

# --- 辅助函数 ---
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_datasets(config):
    """加载数据并返回测试集和所有独特的设备描述"""
    with open(config['paths']['input_data'], 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    all_pairs = []
    unique_devices = set()
    template = config['data_processing']['entity_text_template']
    
    for item in raw_data:
        dtype, vendor, model = item.get("device_type"), item.get("true_vendor"), item.get("true_product")
        if not all([dtype, vendor, model]):
            continue
        triple_text = template.format(type=dtype, vendor=vendor, model=model)
        unique_devices.add(triple_text)
        # 在加载时就清洗 banner
        cleaned_banner = clean_banner(item['banner'])
        all_pairs.append((cleaned_banner, triple_text))
        
    random.seed(42) # 使用固定种子以确保数据划分可复现
    random.shuffle(all_pairs)
    
    dev_proc = config['data_processing']
    train_end = int(len(all_pairs) * dev_proc['train_ratio'])
    valid_end = train_end + int(len(all_pairs) * dev_proc['valid_ratio'])
    
    test_data = all_pairs[valid_end:]
    return test_data, list(unique_devices)

def main():
    """主执行函数"""
    config_path = "mix_0719/0_configs/config.yaml"
    config = load_config(config_path)
    
    logging.info("Initializing model and loading weights...")
    model = JointModel(config)
    model_path = Path(config['paths']['output_dir']) / "best_model_v2.pth"
    if not model_path.exists():
        logging.error(f"FATAL: Model weights not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    logging.info("Loading test set and device library...")
    test_set, device_library = load_datasets(config)
    
    logging.info(f"Test set size: {len(test_set)}")
    logging.info(f"Device library size: {len(device_library)}")
    
    logging.info("Encoding device library...")
    with torch.no_grad():
        device_embeddings = model.encode_text(device_library, device=DEVICE)
    
    top1_correct = 0
    top5_correct = 0
    total_samples = len(test_set)
    near_misses = []  # 新增：用于存储“惜败”案例
    
    logging.info("Starting evaluation on the test set...")
    for banner, true_device in tqdm(test_set, desc="Evaluating"):
        with torch.no_grad():
            # 此处的 banner 已经是清洗过的，直接使用
            banner_embedding = model.encode_text([banner], device=DEVICE)
            cos_scores = F.cosine_similarity(banner_embedding, device_embeddings, dim=1)
            # 始终获取Top-5，便于后续判断
            top_k_scores, top_k_indices = torch.topk(cos_scores, k=5)
            
            top_k_predictions = [device_library[i] for i in top_k_indices]

            # 检查Top-1
            is_top1_correct = top_k_predictions[0] == true_device
            if is_top1_correct:
                top1_correct += 1
            
            # 检查Top-5
            is_top5_correct = true_device in top_k_predictions
            if is_top5_correct:
                top5_correct += 1
            
            # 检查并记录“惜败”案例
            if not is_top1_correct and is_top5_correct:
                near_misses.append({
                    "banner": banner,
                    "true_device": true_device,
                    "predictions": list(zip(top_k_predictions, top_k_scores.cpu().numpy()))
                })

    # --- 打印最终结果 ---
    top1_accuracy = (top1_correct / total_samples) * 100
    top5_accuracy = (top5_correct / total_samples) * 100
    
    print("\n" + "="*25 + " Test Set Evaluation Summary " + "="*25)
    print(f"Total Test Samples: {total_samples}")
    print("-" * 70)
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_samples})")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total_samples})")
    print("=" * 70 + "\n")

    # --- 打印 "Top-1 错，但 Top-5 对" 的详细案例 ---
    if near_misses:
        print("\n" + "="*25 + " Analysis of 'Near Misses' " + "="*25)
        print(f"(Top-1 Incorrect, but Top-5 Correct): Found {len(near_misses)} cases.\n")
        for i, case in enumerate(near_misses):
            print(f"--- Case {i+1} ---")
            print(f"Banner:\n---\n{case['banner']}\n---")
            print(f"\nCorrect Answer: {case['true_device']}")
            print("Model's Top 5 Predictions:")
            for pred_device, pred_score in case['predictions']:
                indicator = "  <-- CORRECT" if pred_device == case['true_device'] else ""
                print(f"  - Score: {pred_score:.4f} | Device: {pred_device}{indicator}")
            print("-" * 70 + "\n")

if __name__ == '__main__':
    main() 