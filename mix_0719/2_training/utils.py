import re

# 定义一个更全面的停用词列表，包括常见的英文停用词和一些技术上可能无意义的词
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 
    'each', 'few', 'for', 'from', 'further', 
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 
    'just', 'me', 'more', 'most', 'my', 'myself', 
    'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    's', 'same', 'she', 'should', 'so', 'some', 'still', 'such', 
    't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 
    'under', 'until', 'up', 'very', 
    'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 
    'you', 'your', 'yours', 'yourself', 'yourselves',
    # 技术相关或常见的无用词
    'http', 'https', 'www', 'com', 'org', 'net', 'banner', 'device', 'type', 'vendor', 'model',
    'error', 'failed', 'login', 'user', 'password', 'access', 'denied', 'success', 'info',
    'get', 'post', 'index', 'html', 'php', 'asp', 'cgi', 'fcgi', 'protocol', 'version'
}

def clean_banner(text: str) -> str:
    """
    清理横幅文本：
    1. 转换为小写
    2. 移除URL和IP地址 (简化版)
    3. 移除非字母数字字符 (保留空格)
    4. 移除停用词
    5. 移除多余的空格
    """
    if not isinstance(text, str):
        return ""
        
    # 1. 小写
    text = text.lower()
    
    # 2. 移除简单的URL和IP (非完美，但能处理常见情况)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    
    # 3. 移除非字母数字字符
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. 移除停用词
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    text = " ".join(filtered_words)
    
    # 5. 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text 