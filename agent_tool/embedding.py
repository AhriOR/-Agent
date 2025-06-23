from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
import torch
import torch.nn.functional as F

class TransformerEmbedding(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,device_map='auto')
        self.model.eval()

    def embed_query(self, text) -> list[float]:
        # 如果不是字符串，尝试提取字符串
        if isinstance(text, dict):
            # 取字典中的第一个value作为文本
            text = list(text.values())[0]
        if not isinstance(text, str):
            raise ValueError(f"embed_query expects str or dict with string value, got {type(text)}")
        return self._embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, dict):
            texts = list(texts.values())
        elif isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError(f"Unsupported input type: {type(texts)}")

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        mean_pooled = F.normalize(mean_pooled, p=2, dim=1)

        return mean_pooled.cpu().tolist()

