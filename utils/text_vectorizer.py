from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "nlpai-lab/KURE-v1"
_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def text_to_vector(text: str) -> List[float]:
    tokenizer, model = _get_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return normalized.squeeze(0).tolist()
