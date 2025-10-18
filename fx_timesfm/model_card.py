from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelCard:
    version: str
    created_at: str
    framework: str
    d_model: int
    n_layers: int
    n_heads: int
    input_patch_len: int
    output_patch_len: int
    context_len_max: int
    normalization: str
    train_freq: str
    train_date_range: Tuple[str, str]
    train_symbols: List[str]
    metrics_snapshot: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "ModelCard":
        data = json.loads(s)
        return ModelCard(**data)

    def validate_compatibility(self, other: "ModelCard") -> List[str]:
        issues: List[str] = []
        if self.d_model != other.d_model:
            issues.append("d_model mismatch")
        if self.n_layers != other.n_layers:
            issues.append("n_layers mismatch")
        if self.n_heads != other.n_heads:
            issues.append("n_heads mismatch")
        if self.input_patch_len != other.input_patch_len:
            issues.append("input_patch_len mismatch")
        if self.output_patch_len != other.output_patch_len:
            issues.append("output_patch_len mismatch")
        if self.context_len_max != other.context_len_max:
            issues.append("context_len_max mismatch")
        return issues

    def warn_on_freq_mismatch(self, freq: str) -> Optional[str]:
        if freq and self.train_freq and freq != self.train_freq:
            return f"Warning: data frequency {freq} differs from ModelCard.train_freq {self.train_freq}. Data will be resampled."
        return None

    @staticmethod
    def create(
        *,
        framework: str,
        d_model: int,
        n_layers: int,
        n_heads: int,
        input_patch_len: int,
        output_patch_len: int,
        context_len_max: int,
        normalization: str,
        train_freq: str,
        train_date_range: Tuple[str, str],
        train_symbols: List[str],
        metrics_snapshot: Dict[str, float],
        version: str = "1.0",
    ) -> "ModelCard":
        return ModelCard(
            version=version,
            created_at=datetime.utcnow().isoformat() + "Z",
            framework=framework,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            input_patch_len=input_patch_len,
            output_patch_len=output_patch_len,
            context_len_max=context_len_max,
            normalization=normalization,
            train_freq=train_freq,
            train_date_range=train_date_range,
            train_symbols=train_symbols,
            metrics_snapshot=metrics_snapshot,
        )
