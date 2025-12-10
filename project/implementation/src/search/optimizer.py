"""
Refactoring Optimizer using Hill Climbing Search Algorithm.

This module implements a heuristic search algorithm to find the optimal
refactoring path for defective software modules, minimizing bug probability
while considering refactoring costs.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


# Refactoring costs per feature
REFACTORING_COSTS = {
    "complexity": 3.0,  # Hard: requires code restructuring
    "effort": 2.5,  # Medium hard: simplify logic
    "size": 2.0,  # Medium: split modules
    "doc_density": 1.0,  # Easy: add comments
    "maintainability": 2.0,  # Medium: general cleanup
}

# How much each feature can realistically be improved
MAX_IMPROVEMENT = {
    "complexity": 0.5,  # Can reduce complexity by up to 50%
    "effort": 0.4,  # Can reduce effort by up to 40%
    "size": 0.3,  # Can reduce size by up to 30%
    "doc_density": 2.0,  # Can increase docs by up to 200%
    "maintainability": 0.5,  # Can increase maintainability by up to 50%
}


@dataclass
class RefactoringRecommendation:
    """Represents a refactoring recommendation for a module."""

    original_features: Dict[str, float]
    optimized_features: Dict[str, float]
    original_risk: float
    optimized_risk: float
    risk_reduction: float
    total_cost: float
    steps: List[Dict]


def hill_climbing_optimizer(
    features: Dict[str, float],
    predict_fn: Callable,
    feature_names: List[str],
    scaler=None,
    max_iterations: int = 100,
    step_size: float = 0.05,
) -> RefactoringRecommendation:
    """
    Hill Climbing algorithm to find optimal refactoring path.

    Args:
        features: Current module features as dict
        predict_fn: Function that predicts bug probability
        feature_names: List of feature names in order
        scaler: StandardScaler used for the model (optional)
        max_iterations: Maximum optimization steps
        step_size: Size of each improvement step (as fraction)

    Returns:
        RefactoringRecommendation with optimal path
    """
    current = np.array([features[f] for f in feature_names], dtype=float)
    original = current.copy()

    # Get original risk
    if scaler is not None:
        original_risk = float(predict_fn(scaler.transform([original]))[0])
    else:
        original_risk = float(predict_fn([original])[0])

    best_features = current.copy()
    best_risk = original_risk
    steps = []
    total_cost = 0.0

    for iteration in range(max_iterations):
        improved = False

        # Try improving each feature
        for i, feat_name in enumerate(feature_names):
            # Calculate step based on feature type
            if feat_name in ["doc_density", "maintainability"]:
                step = abs(current[i]) * step_size
                new_value = current[i] + step
                max_val = features[feat_name] * (1 + MAX_IMPROVEMENT[feat_name])
                new_value = min(new_value, max_val)
            else:
                step = abs(current[i]) * step_size
                new_value = current[i] - step
                min_val = features[feat_name] * (1 - MAX_IMPROVEMENT[feat_name])
                new_value = max(new_value, min_val, 0)

            # Create candidate solution
            candidate = current.copy()
            candidate[i] = new_value

            # Evaluate
            if scaler is not None:
                candidate_risk = float(predict_fn(scaler.transform([candidate]))[0])
            else:
                candidate_risk = float(predict_fn([candidate])[0])

            # If better, accept the change
            if candidate_risk < best_risk:
                improvement = best_risk - candidate_risk
                cost = REFACTORING_COSTS[feat_name] * step_size

                steps.append(
                    {
                        "feature": feat_name,
                        "from": float(current[i]),
                        "to": float(new_value),
                        "risk_before": float(best_risk),
                        "risk_after": float(candidate_risk),
                        "cost": cost,
                    }
                )

                current = candidate.copy()
                best_features = candidate.copy()
                best_risk = candidate_risk
                total_cost += cost
                improved = True
                break  # Greedy

        # Stop if no improvement found
        if not improved:
            break

        # Stop if risk is low enough
        if best_risk < 0.2:
            break

    # Build result
    optimized_features = {
        f: float(best_features[i]) for i, f in enumerate(feature_names)
    }

    return RefactoringRecommendation(
        original_features=features,
        optimized_features=optimized_features,
        original_risk=original_risk,
        optimized_risk=best_risk,
        risk_reduction=original_risk - best_risk,
        total_cost=total_cost,
        steps=steps,
    )


def optimize_high_risk_modules(
    df,
    predict_fn: Callable,
    feature_names: List[str],
    scaler=None,
    risk_threshold: float = 0.5,
    top_n: int = 5,
) -> List[Tuple[int, RefactoringRecommendation]]:
    """
    Find and optimize the highest-risk modules in the dataset.

    Args:
        df: DataFrame with module features
        predict_fn: Prediction function
        feature_names: List of feature names
        scaler: StandardScaler (optional)
        risk_threshold: Minimum risk to consider for optimization
        top_n: Number of modules to optimize

    Returns:
        List of (index, RefactoringRecommendation) tuples
    """
    # Get predictions for all modules
    X = df[feature_names].values
    if scaler is not None:
        risks = predict_fn(scaler.transform(X)).flatten()
    else:
        risks = predict_fn(X).flatten()

    # Find high risk modules
    high_risk_idx = np.where(risks >= risk_threshold)[0]

    # Sort by risk and take top_n
    sorted_idx = high_risk_idx[np.argsort(risks[high_risk_idx])[::-1]][:top_n]

    results = []
    for idx in sorted_idx:
        features = {f: float(df.iloc[idx][f]) for f in feature_names}
        recommendation = hill_climbing_optimizer(
            features=features,
            predict_fn=predict_fn,
            feature_names=feature_names,
            scaler=scaler,
        )
        results.append((int(idx), recommendation))

    return results


def format_recommendation(idx: int, rec: RefactoringRecommendation) -> str:
    """Format a recommendation as a readable string."""
    lines = [
        f"{'='*60}",
        f"MODULE #{idx} - REFACTORING RECOMMENDATION",
        f"{'='*60}",
        f"",
        f"Risk Assessment:",
        f"   Original Risk:  {rec.original_risk:.1%}",
        f"   Optimized Risk: {rec.optimized_risk:.1%}",
        f"   Risk Reduction: {rec.risk_reduction:.1%}",
        f"   Total Cost:     {rec.total_cost:.1f} effort units",
        f"",
        f"Recommended Changes:",
    ]

    for step in rec.steps[:5]:
        direction = "↓" if step["to"] < step["from"] else "↑"
        lines.append(
            f"   • {step['feature']:15} {step['from']:.2f} → {step['to']:.2f} {direction}"
        )

    if len(rec.steps) > 5:
        lines.append(f"   ... and {len(rec.steps) - 5} more steps")

    return "\n".join(lines)
