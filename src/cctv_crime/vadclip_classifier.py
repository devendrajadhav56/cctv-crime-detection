"""VadCLIP (UCF-Crime, weakly-supervised) scoring backend.

Wraps the CLIPVAD checkpoint trained in the sibling vadclip-ucf-crime
reproduction. Reuses the existing per-window frame sampling (frames.py):
the frames_per_clip frames sampled for one sliding window become that
window's VadCLIP "snippet" sequence, zero-padded up to the model's fixed
256-slot buffer, and the per-snippet anomaly scores are max-pooled into a
single window-level score — matching the top-k pooling used in the
model's own training loss (CLAS2).

Two-stage output, mirroring the model's own dual-branch design: the CLAS2
branch (logits1) decides whether the window is anomalous at all; if so,
the CLASM/alignment branch (logits2) names which of the 13 non-Normal
UCF-Crime classes it looks like. The full 14-class distribution is always
returned in `probabilities`, not just the winning label.
"""

from __future__ import annotations

import torch
from PIL import Image

from cctv_crime.config import InferConfig
from cctv_crime.model import ClipPrediction
from cctv_crime.vadclip.clip.clip import _transform
from cctv_crime.vadclip.clipvad import CLIPVAD

# Fixed by the trained checkpoint (VadCLIP AAAI2024, UCF-Crime) — not configurable.
EMBED_DIM = 512
VISUAL_LENGTH = 256
VISUAL_WIDTH = 512
VISUAL_HEAD = 1
VISUAL_LAYERS = 2
ATTN_WINDOW = 8
PROMPT_PREFIX = 10
PROMPT_POSTFIX = 10
CLASS_LABELS = (
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
)


class VadClipClassifier:
    def __init__(self, config: InferConfig, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = CLIPVAD(
            len(CLASS_LABELS),
            EMBED_DIM,
            VISUAL_LENGTH,
            VISUAL_WIDTH,
            VISUAL_HEAD,
            VISUAL_LAYERS,
            ATTN_WINDOW,
            PROMPT_PREFIX,
            PROMPT_POSTFIX,
            self.device,
        )
        state_dict = torch.load(config.vadclip_checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = _transform(self.model.clipmodel.visual.input_resolution)
        self.prompt_text = list(CLASS_LABELS)

    def predict(self, frames: list[Image.Image]) -> ClipPrediction:
        n = len(frames)
        if n == 0:
            raise ValueError("predict() needs at least one frame")
        if n > VISUAL_LENGTH:
            raise ValueError(f"got {n} frames, VadCLIP snippet buffer holds at most {VISUAL_LENGTH}")

        pixel_values = torch.stack([self.preprocess(frame) for frame in frames]).to(self.device)

        with torch.inference_mode():
            snippet_features = self.model.clipmodel.encode_image(pixel_values).to(torch.float)

            visual = torch.zeros(1, VISUAL_LENGTH, VISUAL_WIDTH, device=self.device)
            visual[0, :n] = snippet_features
            lengths = torch.tensor([n])
            padding_mask = torch.zeros(1, VISUAL_LENGTH, dtype=torch.bool, device=self.device)
            padding_mask[0, n:] = True

            _, logits1, logits2 = self.model(visual, padding_mask, self.prompt_text, lengths)

            # Anomaly presence + confidence: CLAS2 branch (logits1), the
            # primary branch — same threshold/shape as before.
            anomaly_probs = torch.sigmoid(logits1[0, :n, 0])
            score = float(anomaly_probs.max().item())

            # Which class: CLASM/alignment branch (logits2), VadCLIP's
            # fine-grained branch — full 14-way distribution, exposed as-is.
            class_probs = torch.softmax(logits2[0, :n], dim=-1).mean(dim=0)
            probabilities = {
                label.lower(): float(prob) for label, prob in zip(CLASS_LABELS, class_probs.tolist())
            }

        if score >= 0.5:
            # Anomalous: name it via the alignment branch, restricted to the
            # 13 non-Normal classes (index 0 is "Normal").
            non_normal = class_probs[1:]
            label = CLASS_LABELS[1 + int(non_normal.argmax().item())].lower()
        else:
            label = "normal"
        confidence = score if label != "normal" else 1.0 - score

        return ClipPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )
