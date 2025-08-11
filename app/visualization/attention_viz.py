import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import io
import base64
import os
import logging
from datetime import timedelta
import logging

from app.utils.date_utils import format_date
from app.data.cache_manager import get_file_cache_dirs

logger = logging.getLogger(__name__)

def visualize_attention_weights(model, features, prev_value, sequence_end_date, feature_names=None, actual_sequence_dates=None):
    from app.utils.date_utils import is_holiday
    """
    모델의 어텐션 가중치를 시각화하는 함수 - 2x2 레이아웃으로 개선
    sequence_end_date: 시퀀스 데이터의 마지막 날짜 (예측 시작일 전날)
    """
    model.eval()
    
    # 특성 이름이 없으면 인덱스로 생성
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[2])]
    else:
        # 특성 수에 맞게 조정
        feature_names = feature_names[:features.shape[2]]
    
    # 텐서가 아니면 변환
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features).to(next(model.parameters()).device)
    
    # prev_value 처리
    if prev_value is not None:
        if not isinstance(prev_value, torch.Tensor):
            try:
                prev_value = float(prev_value)
                prev_value = torch.FloatTensor([prev_value]).to(next(model.parameters()).device)
            except (TypeError, ValueError):
                logger.warning("Warning: prev_value를 숫자로 변환할 수 없습니다. 0으로 대체합니다.")
                prev_value = torch.FloatTensor([0.0]).to(next(model.parameters()).device)
    
    # 시퀀스 길이
    seq_len = features.shape[1]
    
    # 날짜 라벨 생성 - 실제 시퀀스 날짜 사용
    date_labels = []
    if actual_sequence_dates is not None and len(actual_sequence_dates) == seq_len:
        # 실제 날짜 정보가 전달된 경우 사용
        for date in actual_sequence_dates:
            try:
                if isinstance(date, str):
                    date_labels.append(date)
                else:
                    date_labels.append(format_date(date, '%Y-%m-%d'))
            except:
                date_labels.append(str(date))
    else:
        # 실제 날짜 정보가 없으면 기존 방식 사용 (시퀀스 마지막 날짜부터 역순으로)
        for i in range(seq_len):
            try:
                # 시퀀스 마지막 날짜에서 거꾸로 계산
                date = sequence_end_date - timedelta(days=seq_len-i-1)
                date_labels.append(format_date(date, '%Y-%m-%d'))
            except:
                # 날짜 변환 오류 시 인덱스 사용
                date_labels.append(f"T-{seq_len-i-1}")
    
    # GridSpec을 사용한 레이아웃 생성 - 상단 2개, 하단 1개 큰 그래프
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
    # 예측 시작일 계산 (주말/휴일 고려)
    prediction_date = sequence_end_date + timedelta(days=1)
    while prediction_date.weekday() >= 5 or is_holiday(prediction_date):
        prediction_date += timedelta(days=1)
        
    fig.suptitle(f"Attention Weight Analysis for Prediction {format_date(prediction_date, '%Y-%m-%d')}", 
                fontsize=24, fontweight='bold')
    
    # 전체 폰트 크기 설정
    plt.rcParams.update({'font.size': 16})
    
    # 특성 중요도 계산을 위해 데이터 준비
    feature_importance = np.zeros(len(feature_names))
    
    # 특성 중요도를 간단한 방법으로 계산
    # 마지막 시점에서 각 특성의 절대값 사용
    feature_importance = np.mean(np.abs(features[0].cpu().numpy()), axis=0)
    
    # 정규화
    if np.sum(feature_importance) > 0:
        feature_importance = feature_importance / np.sum(feature_importance)
    
    # 특성 중요도를 내림차순으로 정렬
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # 플롯 1: 시간적 중요도 (Time Step Importance) - 상단 왼쪽
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 각 시점의 평균 절대값으로 시간적 중요도 추정
    temporal_importance = np.mean(np.abs(features[0].cpu().numpy()), axis=1)
    if np.sum(temporal_importance) > 0:
        temporal_importance = temporal_importance / np.sum(temporal_importance)
    
    try:
        # 막대그래프로 시간적 중요도 표시
        bars = ax1.bar(range(len(date_labels)), temporal_importance, color='skyblue', alpha=0.7)
        
        # X축 라벨 간격 조정 - 너무 많으면 일부만 표시
        if len(date_labels) > 20:
            # 20개 이상이면 7개 간격으로 표시
            step = max(1, len(date_labels) // 7)
            tick_indices = list(range(0, len(date_labels), step))
            # 마지막 날짜도 포함
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45, ha='right', fontsize=14)
        elif len(date_labels) > 10:
            # 10-20개면 3개 간격으로 표시
            step = max(1, len(date_labels) // 5)
            tick_indices = list(range(0, len(date_labels), step))
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45, ha='right', fontsize=14)
        else:
            # 10개 이하면 모두 표시
            ax1.set_xticks(range(len(date_labels)))
            ax1.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=14)
            
        ax1.set_title("Time Step Importance", fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel("Sequence Dates", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Relative Importance", fontsize=16, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        
        # 마지막 시점 강조
        ax1.bar(len(date_labels)-1, temporal_importance[-1], color='red', alpha=0.7)
        
        # 그리드 추가
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        logger.error(f"시간적 중요도 시각화 오류: {str(e)}")
        ax1.text(0.5, 0.5, "Visualization error", ha='center', va='center', fontsize=16)
    
    # 플롯 2: 특성별 중요도 (Feature Importance) - 상단 오른쪽
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 상위 10개 특성만 표시
    top_n = min(10, len(sorted_features))
    
    try:
        # 수평 막대 차트로 표시
        y_pos = range(top_n)
        bars = ax2.barh(y_pos, sorted_importance[:top_n], color='lightgreen', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_features[:top_n], fontsize=14)
        ax2.set_title("Feature Importance", fontsize=18, fontweight='bold', pad=20)
        ax2.set_xlabel("Relative Importance", fontsize=16, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
        # 중요도 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.3f}", va='center', fontsize=13, fontweight='bold')
        
        # 그리드 추가
        ax2.grid(True, alpha=0.3, axis='x')
    except Exception as e:
        logger.error(f"특성 중요도 시각화 오류: {str(e)}")
        ax2.text(0.5, 0.5, "Visualization error", ha='center', va='center', fontsize=16)
    
    # 플롯 3: 상위 특성들의 시계열 그래프 (Top Features Time Series) - 하단 전체
    ax3 = fig.add_subplot(gs[1, :])
    
    try:
        # 상위 8개 특성 사용 (더 많은 특성을 보여줄 수 있음)
        top_n_series = min(8, len(sorted_features))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i in range(top_n_series):
            feature_idx = sorted_idx[i]
            feature_name = feature_names[feature_idx]
            
            # 해당 특성의 시계열 데이터
            feature_data = features[0, :, feature_idx].cpu().numpy()
            
            # min-max 정규화로 모든 특성을 같은 스케일로 표시
            feature_min = feature_data.min()
            feature_max = feature_data.max()
            if feature_max > feature_min:  # 0으로 나누기 방지
                norm_data = (feature_data - feature_min) / (feature_max - feature_min)
            else:
                norm_data = np.zeros_like(feature_data)
            
            # 특성 중요도에 비례하는 선 두께
            line_width = 2 + sorted_importance[i] * 6
            
            # 플롯
            ax3.plot(range(len(date_labels)), norm_data, 
                    label=f"{feature_name[:20]}... ({sorted_importance[i]:.3f})" if len(feature_name) > 20 else f"{feature_name} ({sorted_importance[i]:.3f})",
                    linewidth=line_width, color=colors[i % len(colors)], alpha=0.8, marker='o', markersize=4)
        
        ax3.set_title("Top Features Time Series (Normalized)", fontsize=20, fontweight='bold', pad=25)
        ax3.set_xlabel("Time Steps", fontsize=18, fontweight='bold')
        ax3.set_ylabel("Normalized Value", fontsize=18, fontweight='bold')
        ax3.legend(fontsize=14, loc='best', ncol=2)  # 2열로 범례 표시
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.tick_params(axis='both', which='major', labelsize=15)
        
        # x축 라벨을 간소화 (너무 많으면 가독성 떨어짐)
        if len(date_labels) > 20:
            # 20개 이상이면 7개 간격으로 표시
            step = max(1, len(date_labels) // 7)
            tick_indices = list(range(0, len(date_labels), step))
            # 마지막 날짜도 포함
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels([date_labels[i] for i in tick_indices], 
                              rotation=45, ha='right', fontsize=14)
        elif len(date_labels) > 10:
            # 10-20개면 5개 간격으로 표시
            step = max(1, len(date_labels) // 5)
            tick_indices = list(range(0, len(date_labels), step))
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels([date_labels[i] for i in tick_indices], 
                              rotation=45, ha='right', fontsize=14)
        else:
            # 10개 이하면 모두 표시
            ax3.set_xticks(range(len(date_labels)))
            ax3.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=14)
            
    except Exception as e:
        logger.error(f"시계열 시각화 오류: {str(e)}")
        ax3.text(0.5, 0.5, "Visualization error", ha='center', va='center', fontsize=18)
    

    
    plt.tight_layout(pad=3.0)
    
    # 이미지를 메모리에 저장
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    img_buf.seek(0)
    
    # Base64로 인코딩
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    
    # 파일 저장 - 파일별 캐시 디렉토리 사용
    try:
        cache_dirs = get_file_cache_dirs()  # 현재 파일의 캐시 디렉토리 가져오기
        attn_dir = cache_dirs['attention_plots']  # attention_plots 디렉토리 사용
        
        # attention 디렉토리가 없으면 생성
        os.makedirs(attn_dir, exist_ok=True)
        
        filename = os.path.join(attn_dir, f"attention_{format_date(prediction_date, '%Y%m%d')}.png")
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_str))
    except Exception as e:
        logger.error(f"Error saving attention image: {str(e)}")
        filename = None
    
    return filename, img_str, {
        'feature_importance': dict(zip(sorted_features, sorted_importance.tolist())),
        'temporal_importance': dict(zip(date_labels, temporal_importance.tolist()))
    }