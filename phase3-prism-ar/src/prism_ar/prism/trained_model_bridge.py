"""Bridge to the trained SafeDriver-IQ/PRISM Random Forest model.

Loads the CRSS-trained model and feature-engineering pipeline from the
phase1-safedriver-iq folder in the current repository, maps DrivingScene
attributes to CRSS features, and produces an environmental safety score /
risk estimate.

Requires:
    <repo-root>/phase1-safedriver-iq/results/models/best_safety_model.pkl
    <repo-root>/phase1-safedriver-iq/results/models/feature_names.txt
    <repo-root>/phase1-safedriver-iq/src/feature_engineering.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd


# Paths to trained artifacts in the current repository
def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repository root containing .git")


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
_SDIQ_ROOT = _REPO_ROOT / "phase1-safedriver-iq"
_MODEL_PATH = _SDIQ_ROOT / "results" / "models" / "best_safety_model.pkl"
_FEATURE_NAMES_PATH = _SDIQ_ROOT / "results" / "models" / "feature_names.txt"
_FE_PATH = _SDIQ_ROOT / "src" / "feature_engineering.py"


def _load_feature_engineer():
    """Load the FeatureEngineer class directly without importing the src package."""
    spec = importlib.util.spec_from_file_location("feature_engineering", str(_FE_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureEngineer


def _map_conditions_to_crss(weather: str, lighting: str, road: str) -> Dict[str, int]:
    """Map human-readable conditions to CRSS encoded values."""
    weather = weather.lower()
    lighting = lighting.lower()
    road = road.lower()

    if any(w in weather for w in ["snow", "sleet"]):
        weather_code = 3
    elif any(w in weather for w in ["rain", "wet"]):
        weather_code = 2
    elif any(w in weather for w in ["fog", "mist", "smoke"]):
        weather_code = 4
    else:
        weather_code = 1

    if "dark" in lighting and "light" in lighting:
        light_code = 3
    elif "dark" in lighting:
        light_code = 2
    elif any(l in lighting for l in ["dawn", "dusk"]):
        light_code = 4
    else:
        light_code = 1

    if any(r in road for r in ["ice", "snow", "frost"]):
        road_code = 3
    elif any(r in road for r in ["wet", "slush", "water"]):
        road_code = 2
    elif any(r in road for r in ["mud", "dirt", "gravel"]):
        road_code = 4
    else:
        road_code = 1

    return {
        "WEATHER": weather_code,
        "WEATHER1": weather_code,
        "LGT_COND": light_code,
        "LGTCON_IM": light_code,
        "WEATHR_IM": weather_code,
        "ROAD_COND": road_code,
        "SURF_COND": road_code,
    }


class TrainedPRISMBridge:
    """Load and run the trained PRISM/SafeDriver-IQ Random Forest model."""

    def __init__(self, model_path: Optional[str] = None, feature_names_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else _MODEL_PATH
        self.feature_names_path = Path(feature_names_path) if feature_names_path else _FEATURE_NAMES_PATH
        self.model = joblib.load(self.model_path)
        with open(self.feature_names_path, "r") as f:
            self.feature_names = [line.strip() for line in f.readlines() if line.strip()]
        self.FeatureEngineer = _load_feature_engineer()

    def _build_scenario(self, weather: str, lighting: str, road_condition: str, time_of_day: str, num_vrus: int) -> Dict:
        """Create a CRSS-like scenario dictionary from scene attributes."""
        hour = int(time_of_day.split(":")[0]) if ":" in time_of_day else 12
        minute = int(time_of_day.split(":")[1]) if ":" in time_of_day else 0
        month = 6
        day_week = 3

        cond = _map_conditions_to_crss(weather, lighting, road_condition)
        adverse_weather = 1 if cond["WEATHER"] > 1 else 0
        poor_lighting = 1 if cond["LGT_COND"] > 1 else 0
        adverse_conditions = 1 if adverse_weather or poor_lighting else 0
        is_night = 1 if hour >= 20 or hour <= 5 else 0
        is_weekend = 1 if day_week in (1, 7) else 0
        is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0

        scenario = {
            "PJ": 1,
            "STRATUM": 1,
            "VE_TOTAL": 2,
            "VE_FORMS": 1,
            "PVH_INVL": 0,
            "PEDS": num_vrus,
            "PERMVIT": 2,
            "PERNOTMVIT": num_vrus,
            "NUM_INJ": 0,
            "MONTH": month,
            "DAY_WEEK": day_week,
            "HOUR": hour,
            "MINUTE": minute,
            "HARM_EV": 1,
            "ALCOHOL": 0,
            "MAX_SEV": 0,
            "MAN_COLL": 1,
            "RELJCT1": 1,
            "RELJCT2": 1,
            "TYP_INT": 1,
            "WRK_ZONE": 0,
            "REL_ROAD": cond["ROAD_COND"],
            "LGT_COND": cond["LGT_COND"],
            "WEATHER1": cond["WEATHER1"],
            "WEATHER2": 0,
            "WEATHER": cond["WEATHER"],
            "SCH_BUS": 0,
            "INT_HWY": 0,
            "CF1": 0,
            "CF2": 0,
            "CF3": 0,
            "WKDY_IM": day_week,
            "HOUR_IM": hour,
            "MINUTE_IM": minute,
            "EVENT1_IM": 1,
            "MANCOL_IM": 1,
            "RELJCT1_IM": 1,
            "RELJCT2_IM": 1,
            "LGTCON_IM": cond["LGTCON_IM"],
            "WEATHR_IM": cond["WEATHR_IM"],
            "MAXSEV_IM": 0,
            "NO_INJ_IM": 0,
            "ALCHL_IM": 0,
            "URBANICITY": 1,
            "REGION": 1,
            "PSUSTRAT": 1,
            "PSU_VAR": 1,
            "WEIGHT": 1.0,
            "YEARNAME": 2023,
            "MINUTE_IMNAME": minute,
            "IS_RUSH_HOUR": is_rush_hour,
            "IS_NIGHT": is_night,
            "IS_WEEKEND": is_weekend,
            "ADVERSE_WEATHER": adverse_weather,
            "POOR_LIGHTING": poor_lighting,
            "ADVERSE_CONDITIONS": adverse_conditions,
            "HAS_TRAFFIC_SIGNAL": 0,
            "pedestrian_count": num_vrus,
            "cyclist_count": 0,
            "total_vru": num_vrus,
            "max_vru_injury": 0,
            "fatal_vru": 0,
            "NIGHT_AND_DARK": 1 if is_night and poor_lighting else 0,
            "WEEKEND_NIGHT": 1 if is_weekend and is_night else 0,
        }
        return scenario

    def _create_feature_vector(self, scenario: Dict) -> pd.DataFrame:
        """Replicate the feature vector creation from realtime_calculator.py."""
        fe = self.FeatureEngineer()
        df = pd.DataFrame([scenario])
        df = fe.create_temporal_features(df)
        df = fe.create_environmental_features(df)
        df = fe.create_location_features(df)

        # Ensure all required features exist
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        feature_vector = df[self.feature_names].fillna(0)
        return feature_vector

    def environmental_score(self, weather: str, lighting: str, road_condition: str, time_of_day: str, num_vrus: int) -> float:
        """Return a 0-100 safety score for environmental conditions."""
        scenario = self._build_scenario(weather, lighting, road_condition, time_of_day, num_vrus)
        X = self._create_feature_vector(scenario)
        proba = self.model.predict_proba(X)[0]
        # Probability of safe driving (class 0)
        safety_score = proba[0] * 100.0
        return float(np.clip(safety_score, 0.0, 100.0))

    def environmental_risk(self, weather: str, lighting: str, road_condition: str, time_of_day: str, num_vrus: int) -> float:
        """Return a 0-1 crash risk for environmental conditions."""
        score = self.environmental_score(weather, lighting, road_condition, time_of_day, num_vrus)
        return 1.0 - score / 100.0
