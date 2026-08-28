"""
model.py — Unified Sign Language Recognition Model (v2.0)
Architecture: Linear Projection → BiLSTM × 2 → Self-Attention → Classifier
Input:  (batch_size, 30_frames, 138_features)
Output: (batch_size, num_classes) — raw logits (apply softmax for probabilities)

Feature layout per frame (138 total):
  [  0 – 62 ] : Left hand  — 21 landmarks × 3 (x,y,z)
  [ 63 –125 ] : Right hand — 21 landmarks × 3 (x,y,z)
  [126 –137 ] : Upper body pose — 4 joints (shoulders + elbows) × 3
"""

import torch
import torch.nn as nn


class SignLanguageModel(nn.Module):
    """
    Bidirectional LSTM with Self-Attention for unified sign recognition.

    Handles BOTH static signs (letters) and dynamic motion signs (gestures)
    with a single model — no manual mode switching required.

    Why BiLSTM + Attention vs. old Random Forest + DTW?
    - BiLSTM captures forward & backward temporal patterns simultaneously
    - Self-Attention automatically learns which frames in the sequence matter most
    - One model = no user menu, no mode switching, continuous recognition
    - Scales to any new sign by retraining, not by template engineering
    """

    def __init__(
        self,
        input_dim: int = 138,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 36,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim  : feature vector size per frame (138 for holistic)
            hidden_dim : LSTM hidden units (output is hidden_dim * 2 due to BiLSTM)
            num_layers : stacked LSTM layers
            num_classes: total number of sign classes (including 'idle')
            dropout    : dropout probability for regularization
        """
        super().__init__()

        # ── 1. Per-frame linear projection ────────────────────────────────
        # Project raw 138 features → 128-dim embedding per frame
        # LayerNorm stabilizes training on variable-scale landmark coordinates
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2. Bidirectional LSTM ─────────────────────────────────────────
        # Processes temporal sequence in BOTH directions
        # forward pass  → captures motion build-up
        # backward pass → captures motion follow-through
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Output dimension per time-step: hidden_dim * 2 (bidirectional concat)

        # ── 3. Self-Attention over time ───────────────────────────────────
        # Learns which frames in the 30-frame window are most discriminative
        # e.g., peak finger extension frame gets highest attention weight
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)

        # ── 4. Unified classifier ─────────────────────────────────────────
        # Maps attended context vector → class logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : Tensor of shape (B, T, F)
                B = batch size
                T = 30 frames (sliding window)
                F = 138 features (holistic landmarks)

        Returns:
            logits : Tensor of shape (B, num_classes)
                     Apply torch.softmax(logits, dim=1) for probabilities.
        """
        B, T, F = x.shape

        # Project each frame independently: (B*T, F) → (B*T, 128) → (B, T, 128)
        x = self.input_proj(x.reshape(B * T, F)).reshape(B, T, -1)

        # BiLSTM across time: (B, T, 128) → (B, T, hidden_dim*2)
        lstm_out, _ = self.bilstm(x)

        # Attention weights across time steps: (B, T, 1) → softmax → (B, T, 1)
        attn_logits = self.attention_fc(lstm_out)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Weighted sum over time → context vector: (B, hidden_dim*2)
        context = (attn_weights * lstm_out).sum(dim=1)

        # Classify: (B, hidden_dim*2) → (B, num_classes)
        return self.classifier(context)


# ── Utility: count model parameters ──────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    import json, os

    labels_path = os.path.join("models", "labels.json")
    num_classes = len(json.load(open(labels_path))) if os.path.exists(labels_path) else 36

    model = SignLanguageModel(input_dim=138, hidden_dim=128,
                               num_layers=2, num_classes=num_classes)
    dummy = torch.randn(4, 30, 138)   # batch=4, 30 frames, 138 features
    out   = model(dummy)

    print(f"✓ Model created successfully")
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {out.shape}   (logits)")
    print(f"  Parameters   : {count_parameters(model):,}")
    probs = torch.softmax(out, dim=1)
    print(f"  Prob sum     : {probs.sum(dim=1).tolist()}  (should be ~1.0)")
