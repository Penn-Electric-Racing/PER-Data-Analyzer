from pathlib import Path

from sentence_transformers.cross_encoder import CrossEncoder


MODEL_ID = "cross-encoder/stsb-distilroberta-base"
TARGET_DIR = Path(__file__).resolve().parents[1] / "models" / "stsb-cross-encoder"


def main() -> None:
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    if TARGET_DIR.exists() and all((TARGET_DIR / name).exists() for name in required_files):
        print(f"Model already present at: {TARGET_DIR}")
        return

    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(MODEL_ID)
    model.save(str(TARGET_DIR))
    print(f"Model downloaded and saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()