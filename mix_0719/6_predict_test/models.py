import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class BannerModelV2(nn.Module):
    """
    一个集成了SBERT和全连接层的端到端模型 (V2)。
    这个版本将SBERT作为模型的一个可训练（或冻结）部分。
    """
    def __init__(self, sbert_model_path, embedding_dim=768, output_dim=384, dropout_rate=0.5):
        """
        初始化模型。
        
        参数:
        - sbert_model_path (str): 预训练SBERT模型的路径。
        - embedding_dim (int): SBERT输出的嵌入维度。
        - output_dim (int): 最终输出的嵌入维度。
        - dropout_rate (float): Dropout的比率。
        """
        super(BannerModelV2, self).__init__()
        # 加载SBERT模型
        self.sbert_model = SentenceTransformer(sbert_model_path)
        
        # 定义一个简单的全连接层，用于降维和特征提取
        self.fc_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), # 添加一个线性层
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, texts):
        """
        定义模型的前向传播。
        
        参数:
        - texts (list of str): 输入的文本列表 (e.g., banner或设备描述)。
        
        返回:
        - torch.Tensor: 经过全连接层处理后的嵌入向量。
        """
        # 使用SBERT编码文本，注意要指定设备
        # SBERT的encode方法返回numpy数组，我们需要转为torch.Tensor
        # 并且，在训练时，需要将SBERT模型移动到正确的设备
        device = self.fc_layer[0].weight.device
        embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, device=device)
        
        # 将SBERT的输出通过全连接层
        output_embeddings = self.fc_layer(embeddings)
        
        return output_embeddings 