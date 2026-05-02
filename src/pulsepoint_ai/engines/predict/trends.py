"""Trend Intelligence: Linear regression to analyze lab value trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression

class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"

@dataclass
class TrendResult:
    test_name: str
    direction: TrendDirection
    slope: float
    message: str

def analyze_trend(test_name: str, values: List[float], higher_is_better: bool = True) -> TrendResult:
    """Uses linear regression to determine if a health metric is improving or worsening."""
    if len(values) < 2:
        return TrendResult(test_name, TrendDirection.STABLE, 0.0, "Not enough data points.")

    # 1. Prepare data (X = time index, Y = value)
    X = np.array(range(len(values))).reshape(-1, 1)
    y = np.array(values)

    # 2. Fit Regression
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])

    # 3. Determine Direction
    # If slope is near zero, it's stable
    if abs(slope) < 0.05 * np.mean(values):
        direction = TrendDirection.STABLE
    elif (slope > 0 and higher_is_better) or (slope < 0 and not higher_is_better):
        direction = TrendDirection.IMPROVING
    else:
        direction = TrendDirection.WORSENING

    messages = {
        TrendDirection.IMPROVING: f"Great! Your {test_name} is trending in the right direction.",
        TrendDirection.STABLE: f"Your {test_name} has remained stable over time.",
        TrendDirection.WORSENING: f"Caution: Your {test_name} trend suggests potential worsening."
    }

    return TrendResult(test_name, direction, slope, messages[direction])
