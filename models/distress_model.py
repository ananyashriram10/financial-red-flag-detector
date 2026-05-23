"""
distress_model.py
=================
Financial distress detector: predicts probability that a company is
heading toward bankruptcy / severe financial distress (is_bankrupt = 1).

Different from the fraud model in two key ways:
  1. Target signal is different  — distress is about insolvency, not manipulation
  2. Patterns are more obvious   — distress shows up clearly in liquidity + leverage
     → deeper trees (max_depth=6) are appropriate here

The distress model complements the fraud model:
  - High fraud score  = accounting manipulation suspected
  - High distress score = structural financial weakness / bankruptcy risk
  - Both can be high at the same time (e.g. Enron)
"""

from __future__ import annotations
import logging
from pathlib import Path

from models.base import BaseRiskModel

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent


class DistressModel(BaseRiskModel):
    """
    Financial distress / bankruptcy probability model.

    Target   : is_bankrupt  (1 = company stopped filing + distress signals)
    Positives: ~3-5% of company-years (more common than outright fraud)
    Tuning   : deeper trees (max_depth=6) since distress has clearer
               financial signatures than subtle manipulation
    """

    target_col = "is_bankrupt"
    model_name = "distress_model"

    def __init__(self):
        super().__init__(xgb_overrides={
            "max_depth":        6,      # deeper — distress patterns are more structural
            "n_estimators":     400,
            "learning_rate":    0.05,
            "min_child_weight": 3,      # distress class is larger, allow smaller leaves
            "gamma":            0.05,
            # Isolation Forest contamination set higher for distress (more common)
        })
        # Override ISO contamination for distress (more common than fraud)
        self.iso_model.set_params(contamination=0.04)

    @classmethod
    def load_trained(cls) -> "DistressModel":
        """Load the production model saved after Kaggle training."""
        path = MODELS_DIR / "distress_model.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Trained distress model not found at {path}.\n"
                "Run notebooks/kaggle_pipeline.ipynb on Kaggle first, "
                "then download distress_model.pkl and place it in models/."
            )
        return cls.load(path)
