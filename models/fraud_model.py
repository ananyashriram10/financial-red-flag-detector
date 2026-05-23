"""
fraud_model.py
==============
Fraud detector: predicts probability that a company-year involves
accounting fraud / earnings manipulation (SEC AAER label = 1).

Inherits all training / prediction logic from BaseRiskModel.
Only overrides: target_col, model_name, XGBoost depth (fraud patterns
are subtle — deeper trees risk overfitting the small fraud class).
"""

from __future__ import annotations
import logging
from pathlib import Path

from models.base import BaseRiskModel

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent


class FraudModel(BaseRiskModel):
    """
    Fraud probability model.

    Target   : is_fraud  (1 = SEC AAER enforcement action, 0 = clean)
    Positives: ~1-2% of all company-years
    Tuning   : shallower trees (max_depth=4) to avoid memorising
               the ~2,000 fraud rows in a 30,000-row dataset
    """

    target_col = "is_fraud"
    model_name = "fraud_model"

    def __init__(self):
        super().__init__(xgb_overrides={
            "max_depth":        4,      # shallower = less overfit on small fraud class
            "n_estimators":     500,    # more trees to compensate for shallow depth
            "learning_rate":    0.04,
            "min_child_weight": 8,      # require more samples per leaf
            "gamma":            0.1,    # minimum loss reduction to split
        })

    @classmethod
    def load_trained(cls) -> "FraudModel":
        """Load the production model saved after Kaggle training."""
        path = MODELS_DIR / "fraud_model.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Trained fraud model not found at {path}.\n"
                "Run notebooks/kaggle_pipeline.ipynb on Kaggle first, "
                "then download fraud_model.pkl and place it in models/."
            )
        return cls.load(path)
