from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
import json
from datetime import datetime, timedelta
from .models import DriftMetrics, DriftSeverity, InferenceLog
from .schemas import DriftMetricsCreate, DriftSummary
from sqlalchemy.orm import Session
from sqlalchemy import func, desc


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.

    Args:
        p: Reference distribution
        q: Current distribution

    Returns:
        KL divergence value
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon

    # Normalize to ensure they are proper probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))


def calculate_psi(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    Args:
        p: Reference distribution
        q: Current distribution

    Returns:
        PSI value
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon

    # Normalize to ensure they are proper probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum((q - p) * np.log(q / p))


def generate_drift_explanation(
    model_name: str,
    input_kl: float,
    input_psi: float,
    output_kl: float,
    output_psi: float,
    input_dist_ref: Dict[str, Any],
    input_dist_curr: Dict[str, Any],
    output_dist_ref: Dict[str, Any],
    output_dist_curr: Dict[str, Any],
) -> str:
    """
    Generate a natural language explanation for the detected drift.

    Args:
        model_name: Name of the model
        input_kl: KL divergence for input
        input_psi: PSI for input
        output_kl: KL divergence for output
        output_psi: PSI for output
        input_dist_ref: Reference input distribution
        input_dist_curr: Current input distribution
        output_dist_ref: Reference output distribution
        output_dist_curr: Current output distribution

    Returns:
        Natural language explanation
    """
    explanation_parts = []

    # Determine overall drift severity
    if max(input_psi, output_psi) > 0.2:
        severity = "critical"
    elif max(input_psi, output_psi) > 0.1:
        severity = "significant"
    else:
        severity = "minor"

    explanation_parts.append(f"Model '{model_name}' is experiencing {severity} drift.")

    # Input drift explanation
    if input_psi > 0.2:
        explanation_parts.append(
            f"Input distribution shows critical drift (PSI={input_psi:.4f})."
        )
    elif input_psi > 0.1:
        explanation_parts.append(
            f"Input distribution shows significant drift (PSI={input_psi:.4f})."
        )
    elif input_psi > 0.05:
        explanation_parts.append(
            f"Input distribution shows minor drift (PSI={input_psi:.4f})."
        )

    # Output drift explanation
    if output_psi > 0.2:
        explanation_parts.append(
            f"Output distribution shows critical drift (PSI={output_psi:.4f})."
        )
    elif output_psi > 0.1:
        explanation_parts.append(
            f"Output distribution shows significant drift (PSI={output_psi:.4f})."
        )
    elif output_psi > 0.05:
        explanation_parts.append(
            f"Output distribution shows minor drift (PSI={output_psi:.4f})."
        )

    # Add recommendations
    if max(input_psi, output_psi) > 0.2:
        explanation_parts.append(
            "Recommendation: Investigate immediately and consider retraining the model."
        )
    elif max(input_psi, output_psi) > 0.1:
        explanation_parts.append(
            "Recommendation: Monitor closely and prepare for potential retraining."
        )
    else:
        explanation_parts.append("Recommendation: Continue monitoring for changes.")

    return " ".join(explanation_parts)


def determine_drift_severity(input_psi: float, output_psi: float) -> DriftSeverity:
    """
    Determine the overall drift severity based on PSI values.

    Args:
        input_psi: PSI for input
        output_psi: PSI for output

    Returns:
        Drift severity level
    """
    max_psi = max(input_psi, output_psi)

    if max_psi > 0.2:
        return DriftSeverity.CRITICAL
    elif max_psi > 0.1:
        return DriftSeverity.WARNING
    else:
        return DriftSeverity.OK


def compute_drift_metrics(
    db: Session,
    model_name: str,
    reference_period_days: int = 7,
    current_period_days: int = 1,
) -> DriftMetricsCreate:
    """
    Compute drift metrics by comparing distributions from a reference period to a current period.

    Args:
        db: Database session
        model_name: Name of the model to analyze
        reference_period_days: Number of days to use for reference period
        current_period_days: Number of days to use for current period

    Returns:
        DriftMetricsCreate object with computed metrics
    """
    now = datetime.utcnow()

    # Get reference period data
    reference_start = now - timedelta(days=reference_period_days + current_period_days)
    reference_end = now - timedelta(days=current_period_days)

    reference_logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.model_name == model_name,
            InferenceLog.timestamp >= reference_start,
            InferenceLog.timestamp < reference_end,
        )
        .all()
    )

    # Get current period data
    current_start = now - timedelta(days=current_period_days)

    current_logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.model_name == model_name,
            InferenceLog.timestamp >= current_start,
        )
        .all()
    )

    # If not enough data, return None
    if len(reference_logs) < 10 or len(current_logs) < 10:
        return None

    # Compute input distributions (using input_shape as a proxy)
    ref_input_shapes = [tuple(log.input_shape) for log in reference_logs]
    curr_input_shapes = [tuple(log.input_shape) for log in current_logs]

    # Count occurrences of each shape
    ref_input_counts = {}
    for shape in ref_input_shapes:
        ref_input_counts[str(shape)] = ref_input_counts.get(str(shape), 0) + 1

    curr_input_counts = {}
    for shape in curr_input_shapes:
        curr_input_counts[str(shape)] = curr_input_counts.get(str(shape), 0) + 1

    # Create unified list of all shapes
    all_shapes = list(
        set(list(ref_input_counts.keys()) + list(curr_input_counts.keys()))
    )

    # Create arrays for KL and PSI calculation
    ref_input_dist = np.array([ref_input_counts.get(shape, 0) for shape in all_shapes])
    curr_input_dist = np.array(
        [curr_input_counts.get(shape, 0) for shape in all_shapes]
    )

    # Compute output distributions
    ref_output_classes = [log.output_class for log in reference_logs]
    curr_output_classes = [log.output_class for log in current_logs]

    # Count occurrences of each class
    ref_output_counts = {}
    for cls in ref_output_classes:
        ref_output_counts[cls] = ref_output_counts.get(cls, 0) + 1

    curr_output_counts = {}
    for cls in curr_output_classes:
        curr_output_counts[cls] = curr_output_counts.get(cls, 0) + 1

    # Create unified list of all classes
    all_classes = list(
        set(list(ref_output_counts.keys()) + list(curr_output_counts.keys()))
    )

    # Create arrays for KL and PSI calculation
    ref_output_dist = np.array([ref_output_counts.get(cls, 0) for cls in all_classes])
    curr_output_dist = np.array([curr_output_counts.get(cls, 0) for cls in all_classes])

    # Calculate drift metrics
    input_kl = calculate_kl_divergence(ref_input_dist, curr_input_dist)
    input_psi = calculate_psi(ref_input_dist, curr_input_dist)
    output_kl = calculate_kl_divergence(ref_output_dist, curr_output_dist)
    output_psi = calculate_psi(ref_output_dist, curr_output_dist)

    # Determine drift severity
    drift_severity = determine_drift_severity(input_psi, output_psi)

    # Generate explanation
    explanation = generate_drift_explanation(
        model_name,
        input_kl,
        input_psi,
        output_kl,
        output_psi,
        {"shapes": {shape: count for shape, count in zip(all_shapes, ref_input_dist)}},
        {"shapes": {shape: count for shape, count in zip(all_shapes, curr_input_dist)}},
        {"classes": {cls: count for cls, count in zip(all_classes, ref_output_dist)}},
        {"classes": {cls: count for cls, count in zip(all_classes, curr_output_dist)}},
    )

    # Create drift metrics object
    return DriftMetricsCreate(
        model_name=model_name,
        input_kl_divergence=float(input_kl),
        input_psi=float(input_psi),
        input_distribution_reference={
            "shapes": {
                shape: int(count) for shape, count in zip(all_shapes, ref_input_dist)
            }
        },
        input_distribution_current={
            "shapes": {
                shape: int(count) for shape, count in zip(all_shapes, curr_input_dist)
            }
        },
        output_kl_divergence=float(output_kl),
        output_psi=float(output_psi),
        output_distribution_reference={
            "classes": {
                cls: int(count) for cls, count in zip(all_classes, ref_output_dist)
            }
        },
        output_distribution_current={
            "classes": {
                cls: int(count) for cls, count in zip(all_classes, curr_output_dist)
            }
        },
        drift_severity=drift_severity,
        explanation=explanation,
        timestamp=now,
    )


def get_drift_summary(db: Session, model_name: str, days: int = 30) -> DriftSummary:
    """
    Get a summary of drift metrics for a model over time.

    Args:
        db: Database session
        model_name: Name of the model
        days: Number of days to include in the summary

    Returns:
        DriftSummary object
    """
    # Get drift metrics for the specified period
    start_date = datetime.utcnow() - timedelta(days=days)

    drift_metrics = (
        db.query(DriftMetrics)
        .filter(
            DriftMetrics.model_name == model_name, DriftMetrics.timestamp >= start_date
        )
        .order_by(DriftMetrics.timestamp.asc())
        .all()
    )

    # If no metrics, return empty summary
    if not drift_metrics:
        return DriftSummary(
            model_name=model_name,
            current_severity=DriftSeverity.OK,
            input_drift_trend=[],
            output_drift_trend=[],
            latest_explanation=None,
        )

    # Get latest severity and explanation
    latest_metric = drift_metrics[-1]
    current_severity = latest_metric.drift_severity
    latest_explanation = latest_metric.explanation

    # Prepare drift trends
    input_drift_trend = [
        {
            "timestamp": metric.timestamp.isoformat(),
            "kl_divergence": metric.input_kl_divergence,
            "psi": metric.input_psi,
            "severity": metric.drift_severity,
        }
        for metric in drift_metrics
    ]

    output_drift_trend = [
        {
            "timestamp": metric.timestamp.isoformat(),
            "kl_divergence": metric.output_kl_divergence,
            "psi": metric.output_psi,
            "severity": metric.drift_severity,
        }
        for metric in drift_metrics
    ]

    return DriftSummary(
        model_name=model_name,
        current_severity=current_severity,
        input_drift_trend=input_drift_trend,
        output_drift_trend=output_drift_trend,
        latest_explanation=latest_explanation,
    )
