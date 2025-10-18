from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .model_card import ModelCard


def save_ckpt(path: str, model_state: Dict[str, Any], optimizer_state: Optional[Dict[str, Any]], cfg: Dict[str, Any], card: ModelCard) -> None:
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "cfg": cfg,
        "model_card": card.to_json(),
    }
    torch.save(payload, path)


def load_ckpt(path: str):
    payload = torch.load(path, map_location="cpu")
    card = ModelCard.from_json(payload.get("model_card")) if payload.get("model_card") else None
    return payload, card
