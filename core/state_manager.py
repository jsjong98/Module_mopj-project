_dataframe_cache = {}
_cache_expiry_seconds = 3600

prediction_state = {
    'current_data': None,
    'latest_predictions': None,
    'latest_interval_scores': None,
    'latest_attention_data': None,
    'latest_ma_results': None,
    'latest_plots': None,
    'latest_metrics': None,
    'current_date': None,
    'current_file': None,
    'is_predicting': False,
    'prediction_progress': 0,
    'prediction_start_time': None,
    'error': None,
    'selected_features': None,
    'feature_importance': None,
    'semimonthly_period': None,
    'next_semimonthly_period': None,
    'accumulated_predictions': [],
    'accumulated_metrics': {},
    'prediction_dates': [],
    'accumulated_consistency_scores': {},
    'accumulated_purchase_reliability': 0, # 새로 추가된 필드
    'accumulated_purchase_debug': {}, # 새로 추가된 필드
    'cache_statistics': {}, # 새로 추가된 필드
    'varmax_predictions': None,
    'varmax_half_month_averages': None, # 새로 추가된 필드
    'varmax_metrics': None,
    'varmax_ma_results': None,
    'varmax_selected_features': None,
    'varmax_current_date': None,
    'varmax_model_info': None,
    'varmax_plots': None,
    'varmax_is_predicting': False,
    'varmax_prediction_progress': 0,
    'varmax_prediction_start_time': None,
    'varmax_error': None,
}