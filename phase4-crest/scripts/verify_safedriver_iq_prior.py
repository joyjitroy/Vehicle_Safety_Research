"""Verify SafeDriver-IQ prior artifact loadability and input/output schema."""
import json
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(r'C:\Personal\EB1A\VehicleSafetyResearch\phase1-safedriver-iq\results\models\best_safety_model.pkl')
OUT_JSON = Path(__file__).resolve().parents[1] / 'outputs' / 'results' / 'safedriver_iq_prior_info.json'
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def main():
    print(f'Loading SafeDriver-IQ prior from {MODEL_PATH} ...')
    model = joblib.load(MODEL_PATH)

    info = {
        'model_path': str(MODEL_PATH),
        'model_type': type(model).__name__,
        'n_features_in': int(model.n_features_in_),
        'feature_names': list(model.feature_names_in_),
        'classes': [str(c) for c in model.classes_],
    }

    # Dummy prediction with zeros (CRSS-like input shape)
    dummy_X = np.zeros((1, model.n_features_in_), dtype=float)
    proba = model.predict_proba(dummy_X)
    safety_score = proba[0, 0] * 100.0  # class 0 = no-crash probability -> score
    info['dummy_prediction'] = {
        'class_0_probability': float(proba[0, 0]),
        'class_1_probability': float(proba[0, 1]),
        'safety_score_0_100': float(safety_score),
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print(f"Model type: {info['model_type']}")
    print(f"Features: {info['n_features_in']} CRSS categorical features")
    print(f"Dummy safety score: {safety_score:.2f}/100")
    print(f"Saved: {OUT_JSON}")
    print('\nNote: SafeDriver-IQ prior is a CRSS crash classifier. Its feature space does not map directly to NGSIM/MiTra trajectory states. A5 remains an optional secondary experiment.')


if __name__ == '__main__':
    main()
