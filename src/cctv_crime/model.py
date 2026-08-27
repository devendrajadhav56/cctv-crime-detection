"""Zero-shot fight vs normal scoring with X-CLIP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

from cctv_crime.config import InferConfig


@dataclass(frozen=True)
class ClipPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class ZeroShotClipClassifier:
    def __init__(self, config: InferConfig, device: str | None = None) -> None:
        self.config = config
        self.labels = list(config.labels)
        self.texts = [config.prompts[label] for label in self.labels]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = XCLIPProcessor.from_pretrained(config.model_name)
        self.model = XCLIPModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frames: list[Image.Image]) -> ClipPrediction:
        inputs = self.processor(
            text=self.texts,
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits_per_video
            probs = logits.softmax(dim=-1)[0]
        probabilities = {
            label: float(probs[index].item()) for index, label in enumerate(self.labels)
        }
        best = max(self.labels, key=lambda label: probabilities[label])
        return ClipPrediction(
            label=best,
            confidence=probabilities[best],
            probabilities=probabilities,
        )
