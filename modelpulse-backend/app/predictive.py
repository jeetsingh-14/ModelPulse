from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from datetime import datetime, timedelta
import json
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from .models import DriftMetrics, InferenceLog, AlertThreshold, DriftSeverity
from .schemas import DriftMetricsResponse
from .drift import determine_drift_severity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_time_series_data(
    db: Session, 
    model_name: str, 
    metric_type: str,
    days_history: int = 30
) -> pd.DataFrame:
    """
    Prepare time series data for forecasting.
    
    Args:
        db: Database session
        model_name: Name of the model to analyze
        metric_type: Type of metric to forecast ('drift', 'latency', 'confidence')
        days_history: Number of days of historical data to use
        
    Returns:
        DataFrame with time series data
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days_history)
    
    if metric_type == 'drift':
        # Get drift metrics history
        drift_metrics = (
            db.query(DriftMetrics)
            .filter(
                DriftMetrics.model_name == model_name,
                DriftMetrics.timestamp >= start_date
            )
            .order_by(DriftMetrics.timestamp)
            .all()
        )
        
        if not drift_metrics:
            logger.warning(f"No drift metrics found for model {model_name}")
            return None
            
        # Create DataFrame
        data = []
        for metric in drift_metrics:
            data.append({
                'timestamp': metric.timestamp,
                'input_psi': metric.input_psi,
                'output_psi': metric.output_psi,
                'input_kl_divergence': metric.input_kl_divergence,
                'output_kl_divergence': metric.output_kl_divergence,
                'drift_severity': metric.drift_severity
            })
        
        df = pd.DataFrame(data)
        
    elif metric_type in ['latency', 'confidence']:
        # Get inference logs
        logs = (
            db.query(
                func.date_trunc('hour', InferenceLog.timestamp).label('hour'),
                func.avg(InferenceLog.latency_ms).label('avg_latency'),
                func.avg(InferenceLog.confidence).label('avg_confidence'),
                func.count(InferenceLog.id).label('count')
            )
            .filter(
                InferenceLog.model_name == model_name,
                InferenceLog.timestamp >= start_date
            )
            .group_by(func.date_trunc('hour', InferenceLog.timestamp))
            .order_by(func.date_trunc('hour', InferenceLog.timestamp))
            .all()
        )
        
        if not logs:
            logger.warning(f"No inference logs found for model {model_name}")
            return None
            
        # Create DataFrame
        data = []
        for log in logs:
            data.append({
                'timestamp': log.hour,
                'avg_latency': log.avg_latency,
                'avg_confidence': log.avg_confidence,
                'count': log.count
            })
        
        df = pd.DataFrame(data)
    
    else:
        logger.error(f"Unknown metric type: {metric_type}")
        return None
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    return df

def train_forecast_model(
    df: pd.DataFrame, 
    target_column: str,
    forecast_horizon: int = 24,
    model_type: str = 'xgboost'
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Train a forecasting model and generate predictions.
    
    Args:
        df: DataFrame with time series data
        target_column: Column to forecast
        forecast_horizon: Number of hours to forecast
        model_type: Type of model to use ('linear', 'random_forest', 'xgboost')
        
    Returns:
        Tuple of (trained model, forecast results)
    """
    if df is None or len(df) < 10:
        logger.warning("Not enough data to train forecast model")
        return None, []
    
    # Prepare features and target
    X = df[['hour', 'day_of_week', 'day_of_month', 'month']]
    y = df[target_column]
    
    # Train model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None, []
    
    model.fit(X, y)
    
    # Generate future timestamps
    last_timestamp = df['timestamp'].iloc[-1]
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(forecast_horizon)]
    
    # Create features for future timestamps
    future_features = []
    for ts in future_timestamps:
        future_features.append({
            'timestamp': ts,
            'hour': ts.hour,
            'day_of_week': ts.dayofweek,
            'day_of_month': ts.day,
            'month': ts.month
        })
    
    future_df = pd.DataFrame(future_features)
    
    # Make predictions
    future_X = future_df[['hour', 'day_of_week', 'day_of_month', 'month']]
    predictions = model.predict(future_X)
    
    # Create forecast results
    forecast_results = []
    for i, ts in enumerate(future_timestamps):
        forecast_results.append({
            'timestamp': ts.isoformat(),
            'predicted_value': float(predictions[i]),
            'confidence_interval_lower': float(predictions[i] * 0.9),  # Simple approximation
            'confidence_interval_upper': float(predictions[i] * 1.1)   # Simple approximation
        })
    
    return model, forecast_results

def predict_drift_metrics(
    db: Session,
    model_name: str,
    forecast_horizon: int = 24,
    threshold_days: int = 30
) -> Dict[str, Any]:
    """
    Predict future drift metrics for a model.
    
    Args:
        db: Database session
        model_name: Name of the model to analyze
        forecast_horizon: Number of hours to forecast
        threshold_days: Number of days of historical data to use
        
    Returns:
        Dictionary with prediction results
    """
    # Get historical drift data
    df = prepare_time_series_data(db, model_name, 'drift', threshold_days)
    
    if df is None or len(df) < 10:
        return {
            "model_name": model_name,
            "error": "Not enough historical data for predictions",
            "input_psi_forecast": [],
            "output_psi_forecast": []
        }
    
    # Train models and get forecasts
    _, input_psi_forecast = train_forecast_model(df, 'input_psi', forecast_horizon)
    _, output_psi_forecast = train_forecast_model(df, 'output_psi', forecast_horizon)
    
    # Determine predicted severity for each forecast point
    for i in range(len(input_psi_forecast)):
        input_psi = input_psi_forecast[i]['predicted_value']
        output_psi = output_psi_forecast[i]['predicted_value']
        
        # Ensure non-negative values
        input_psi = max(0, input_psi)
        output_psi = max(0, output_psi)
        
        severity = determine_drift_severity(input_psi, output_psi)
        
        input_psi_forecast[i]['predicted_severity'] = severity
        output_psi_forecast[i]['predicted_severity'] = severity
    
    # Find the earliest predicted breach
    breach_time = None
    for forecast in input_psi_forecast:
        if forecast['predicted_severity'] in [DriftSeverity.WARNING, DriftSeverity.CRITICAL]:
            breach_time = forecast['timestamp']
            break
    
    if breach_time is None:
        for forecast in output_psi_forecast:
            if forecast['predicted_severity'] in [DriftSeverity.WARNING, DriftSeverity.CRITICAL]:
                breach_time = forecast['timestamp']
                break
    
    return {
        "model_name": model_name,
        "input_psi_forecast": input_psi_forecast,
        "output_psi_forecast": output_psi_forecast,
        "predicted_breach_time": breach_time,
        "forecast_generated_at": datetime.utcnow().isoformat()
    }

def predict_performance_metrics(
    db: Session,
    model_name: str,
    metric_type: str,
    forecast_horizon: int = 24,
    threshold_days: int = 30
) -> Dict[str, Any]:
    """
    Predict future performance metrics for a model.
    
    Args:
        db: Database session
        model_name: Name of the model to analyze
        metric_type: Type of metric to forecast ('latency', 'confidence')
        forecast_horizon: Number of hours to forecast
        threshold_days: Number of days of historical data to use
        
    Returns:
        Dictionary with prediction results
    """
    # Get historical performance data
    df = prepare_time_series_data(db, model_name, metric_type, threshold_days)
    
    if df is None or len(df) < 10:
        return {
            "model_name": model_name,
            "error": "Not enough historical data for predictions",
            "forecast": []
        }
    
    # Train model and get forecast
    target_column = f"avg_{metric_type}"
    _, forecast = train_forecast_model(df, target_column, forecast_horizon)
    
    # Get alert thresholds for this metric
    thresholds = (
        db.query(AlertThreshold)
        .filter(
            AlertThreshold.model_name == model_name,
            AlertThreshold.metric_name == metric_type,
            AlertThreshold.is_active == True
        )
        .all()
    )
    
    # Determine predicted breaches
    breach_time = None
    for i, pred in enumerate(forecast):
        predicted_value = pred['predicted_value']
        
        # Check against thresholds
        for threshold in thresholds:
            is_breach = False
            
            # For latency, breach if predicted value is higher than threshold
            if metric_type == 'latency' and predicted_value > threshold.threshold_value:
                is_breach = True
            
            # For confidence, breach if predicted value is lower than threshold
            elif metric_type == 'confidence' and predicted_value < threshold.threshold_value:
                is_breach = True
            
            if is_breach:
                forecast[i]['threshold_breach'] = True
                forecast[i]['threshold_id'] = threshold.id
                forecast[i]['threshold_value'] = threshold.threshold_value
                
                if breach_time is None:
                    breach_time = pred['timestamp']
                
                break
            else:
                forecast[i]['threshold_breach'] = False
    
    return {
        "model_name": model_name,
        "metric_type": metric_type,
        "forecast": forecast,
        "predicted_breach_time": breach_time,
        "forecast_generated_at": datetime.utcnow().isoformat()
    }

def get_all_predictive_alerts(
    db: Session,
    model_name: Optional[str] = None,
    forecast_horizon: int = 24
) -> List[Dict[str, Any]]:
    """
    Get all predictive alerts for models.
    
    Args:
        db: Database session
        model_name: Optional filter for specific model
        forecast_horizon: Number of hours to forecast
        
    Returns:
        List of predictive alerts
    """
    # Get all models or specific model
    if model_name:
        models = [model_name]
    else:
        # Get unique model names from inference logs
        model_names = db.query(InferenceLog.model_name).distinct().all()
        models = [m[0] for m in model_names]
    
    alerts = []
    
    for model in models:
        # Predict drift metrics
        drift_prediction = predict_drift_metrics(db, model, forecast_horizon)
        
        # Predict latency
        latency_prediction = predict_performance_metrics(db, model, 'latency', forecast_horizon)
        
        # Predict confidence
        confidence_prediction = predict_performance_metrics(db, model, 'confidence', forecast_horizon)
        
        # Combine predictions
        model_alerts = {
            "model_name": model,
            "drift_prediction": drift_prediction,
            "latency_prediction": latency_prediction,
            "confidence_prediction": confidence_prediction,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        alerts.append(model_alerts)
    
    return alerts