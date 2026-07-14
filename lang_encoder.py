import torch
from transformers import AutoTokenizer, AutoModel

# Small, frozen sentence-embedding model loaded via plain `transformers` (already a pinned
# dependency), so no extra `sentence-transformers` package is needed. 384-dim, Apache-2.0.
DEFAULT_LANG_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LANG_EMBED_DIM = 384


class FrozenTextEncoder:
    """Frozen text encoder used to condition DECOVitac on the task's language prompt.

    Training uses embeddings precomputed offline by utils/cal_text_embeddings.py (see
    lerobot_dataset.py's lang_embed_cache); inference uses this class directly since the
    deployed prompt may not be one seen during training.
    """

    def __init__(self, model_name=DEFAULT_LANG_MODEL, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-9)
        return summed / counts

    @torch.no_grad()
    def embed(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**encoded)
        emb = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu()
