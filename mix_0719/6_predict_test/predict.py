import yaml
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from pathlib import Path
import logging
import sys

# 将训练目录添加到sys.path以导入clean_banner
sys.path.append(str(Path(__file__).resolve().parents[1] / '2_training'))
from utils import clean_banner

# --- 日志和设备配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型定义 ---
# 注意：为了让此脚本能独立运行，我们在此处复制了模型类的定义。
# 在大型项目中，建议将模型定义移到共享的 `models.py` 文件中。
class JointModel(nn.Module):
    """
    一个端到端的联合模型 (v2.2)，将SBERT和TransE结合在一起。
    这个版本增加了更深的全连接层来增强模型容量。
    """
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
        """使用SBERT和投影层来编码文本"""
        sbert_embeddings = self.sbert.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=False)
        projected_embeddings = self.projection_layer(sbert_embeddings)
        return projected_embeddings

# --- 辅助函数 ---
def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_unique_devices(data_path, template):
    """从输入数据加载所有独特的设备描述文本"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    unique_devices = set()
    for item in raw_data:
        dtype = item.get("device_type")
        vendor = item.get("true_vendor")
        model = item.get("true_product")
        if not all([dtype, vendor, model]):
            continue
        triple_text = template.format(type=dtype, vendor=vendor, model=model)
        unique_devices.add(triple_text)
    return list(unique_devices)

# --- 主预测逻辑 ---
def predict(model, banner_to_predict, device_library, device_embeddings, top_k=5):
    """
    对给定的banner进行预测，并打印结果。
    """
    model.eval()
    
    # 在预测前清洗banner
    cleaned_banner = clean_banner(banner_to_predict)
    
    with torch.no_grad():
        banner_embedding = model.encode_text([cleaned_banner], device=DEVICE)
        cos_scores = F.cosine_similarity(banner_embedding, device_embeddings, dim=1)
        top_k_scores, top_k_indices = torch.topk(cos_scores, k=min(top_k, len(device_library)))
        
        print("\n" + "="*25 + " Prediction Result " + "="*25)
        print(f"Original Banner:\n---\n{banner_to_predict}\n---")
        print(f"Cleaned Banner:\n---\n{cleaned_banner}\n---")
        print(f"\nTop {top_k} Predictions:")
        for i in range(len(top_k_scores)):
            score = top_k_scores[i].item()
            device_text = device_library[top_k_indices[i].item()]
            print(f"  {i+1}. Score: {score:.4f}  |  Device: {device_text}")
        print("="*70 + "\n")

def main():
    """主执行函数"""
    # 1. 加载配置
    config_path = "mix_0719/0_configs/config.yaml"
    config = load_config(config_path)
    logging.info(f"Configuration loaded from {config_path}")
    
    # 2. 初始化模型并加载训练好的权重
    logging.info("Initializing model structure...")
    model = JointModel(config, num_relations=1) # 推理时部分参数可用默认值
    
    model_path = Path(config['paths']['output_dir']) / "best_model_v2.pth"
    if not model_path.exists():
        logging.error(f"FATAL: Model weights not found at {model_path}")
        return
        
    logging.info(f"Loading trained model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. 加载并编码设备库作为“指纹库”
    logging.info("Loading device library...")
    device_template = config['data_processing']['entity_text_template']
    device_library = load_unique_devices(config['paths']['input_data'], device_template)
    
    logging.info(f"Found {len(device_library)} unique devices. Encoding them to create fingerprint library...")
    with torch.no_grad():
        device_embeddings = model.encode_text(device_library, device=DEVICE)
    logging.info("Device fingerprint library created successfully.")
    
    # 4. 定义要测试的新Banner (您可以替换成任何您想测试的Banner)
    banner_1 = 'HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Digest realm="AXIS_ACCC8E270099", nonce="0000a68bY73713f4c653733b86c8f878b3400ab8", stale=FALSE, qop="auth"\r\nContent-Length: 0\r\nContent-Type: text/html; charset=iso-8859-1\r\n'
    predict(model, banner_1, device_library, device_embeddings, top_k=5)

    banner_2 = 'HTTP/1.0 401 Unauthorized\r\nContent-Type: text/html\r\nWWW-Authenticate: Basic realm="netcam"\r\n\r\n'
    predict(model, banner_2, device_library, device_embeddings, top_k=5)

    banner_3 = 'HTTP/1.1 200 OK\r\nServer: an-httpd/2.1\r\nDate: Sun, 16 May 2021 17:15:23 GMT\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n<HTML><HEAD><TITLE>GoAhead WebServer</TITLE></HEAD></HTML>'
    predict(model, banner_3, device_library, device_embeddings, top_k=5)

if __name__ == '__main__':
    main() 