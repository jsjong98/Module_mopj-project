"""
Prediction package for LSTM and VARMAX predictions
"""

# Background tasks
from .background_tasks import (
    calculate_estimated_time_remaining,
    format_time_duration,
    run_accumulated_predictions_with_save,
    background_accumulated_prediction,
    background_prediction_simple_compatible,
    generate_visualizations_realtime,
    regenerate_visualizations_from_cache,
    generate_accumulated_report,
    background_varmax_prediction
)

# Metrics calculations
from .metrics import (
    calculate_interval_averages_and_scores,
    decide_purchase_interval,
    compute_performance_metrics_improved,
    calculate_f1_score,
    calculate_direction_accuracy,
    calculate_direction_weighted_score,
    calculate_mape,
    calculate_moving_averages_with_history,
    calculate_prediction_consistency,
    calculate_accumulated_purchase_reliability,
    calculate_accumulated_purchase_reliability_with_debug,
    calculate_actual_business_days,
    get_previous_semimonthly_period
)

# Prediction functions
from .predictor import (
    generate_predictions,
    generate_predictions_compatible,
    generate_predictions_with_save,
    generate_predictions_with_attention_save,
    # check_existing_prediction,
    find_compatible_hyperparameters
)

__all__ = [
    # Background tasks
    'calculate_estimated_time_remaining',
    'format_time_duration',
    'run_accumulated_predictions_with_save',
    'background_accumulated_prediction',
    'background_prediction_simple_compatible',
    'generate_visualizations_realtime',
    'regenerate_visualizations_from_cache',
    'generate_accumulated_report',
    'background_varmax_prediction',
    
    # Metrics
    'calculate_interval_averages_and_scores',
    'decide_purchase_interval',
    'compute_performance_metrics_improved',
    'calculate_f1_score',
    'calculate_direction_accuracy',
    'calculate_direction_weighted_score',
    'calculate_mape',
    'calculate_moving_averages_with_history',
    'calculate_prediction_consistency',
    'calculate_accumulated_purchase_reliability',
    'calculate_accumulated_purchase_reliability_with_debug',
    'calculate_actual_business_days',
    'get_previous_semimonthly_period',
    
    # Predictors
    'generate_predictions',
    'generate_predictions_compatible',
    'generate_predictions_with_save',
    'generate_predictions_with_attention_save',
    'check_existing_prediction',
    'find_compatible_hyperparameters'
]
