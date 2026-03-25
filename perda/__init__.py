from pathlib import Path

from sentence_transformers.cross_encoder import CrossEncoder

_MODEL_DIR = Path(__file__).parent / "models" / "stsb-cross-encoder"
_HF_MODEL_ID = "cross-encoder/stsb-distilroberta-base"

if not _MODEL_DIR.exists():
    print("Downloading cross-encoder model (one-time setup)...")
    _model = CrossEncoder(_HF_MODEL_ID)
    _MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    _model.save(str(_MODEL_DIR))
    print(f"Model saved to: {_MODEL_DIR}")
