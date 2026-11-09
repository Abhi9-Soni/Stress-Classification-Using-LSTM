"""Predict a dataset label with a saved LSTM. Run: python predict.py --text '...'."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from preprocessing import clean_text


def predict(text, artifact_dir):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input is empty after cleaning. Enter meaningful text.")
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(artifact_dir / "stress_text_model.keras", compile=False)
    score = float(model(tf.constant([cleaned]), training=False).numpy().ravel()[0])
    return {"predicted_label": "stressed" if score >= config["threshold"] else "normal",
            "stressed_score": score,
            "truncated": len(cleaned.split()) > config["sequence_length"]}


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    default_dir = base / "artifacts" if (base / "artifacts").exists() else base
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Text to classify")
    parser.add_argument("--artifacts", default=str(default_dir), help="Saved model/config folder")
    args = parser.parse_args()
    print(json.dumps(predict(args.text, args.artifacts), indent=2))
