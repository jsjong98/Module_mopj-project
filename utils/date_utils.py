import pandas as pd
import calendar
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)

def format_date(date_obj, format_str='%Y-%m-%d'):
    # 기존 코드 그대로 유지
    """날짜 객체를 문자열로 안전하게 변환"""
    try:
        # pandas Timestamp 또는 datetime.datetime
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime(format_str)
        
        # numpy.datetime64
        elif isinstance(date_obj, np.datetime64):
            # 날짜 포맷이 'YYYY-MM-DD'인 경우
            return str(date_obj)[:10]
        
        # 문자열인 경우 이미 날짜 형식이라면 추가 처리
        elif isinstance(date_obj, str):
            # GMT 형식이면 파싱하여 변환
            if 'GMT' in date_obj:
                parsed_date = datetime.strptime(date_obj, '%a, %d %b %Y %H:%M:%S GMT')
                return parsed_date.strftime(format_str)
            return date_obj[:10] if len(date_obj) > 10 else date_obj
        
        # 그 외 경우
        else:
            return str(date_obj)
    
    except Exception as e:
        logger.warning(f"날짜 포맷팅 오류: {str(e)}")
        return str(date_obj)
    
# 1. 반월 기간 계산 함수
def get_semimonthly_period(date):
    """
    날짜를 반월 기간으로 변환하는 함수
    - 1일~15일: "YYYY-MM-SM1"
    - 16일~말일: "YYYY-MM-SM2"
    """
    year = date.year
    month = date.month
    day = date.day
    
    if day <= 15:
        semimonthly = f"{year}-{month:02d}-SM1"
    else:
        semimonthly = f"{year}-{month:02d}-SM2"
    
    return semimonthly

# 2. 특정 날짜 이후의 다음 반월 기간 계산 함수
def get_next_semimonthly_period(date):
    """
    주어진 날짜 이후의 다음 반월 기간을 계산하는 함수
    """
    year = date.year
    month = date.month
    day = date.day
    
    if day <= 15:
        # 현재 상반월이면 같은 달의 하반월
        semimonthly = f"{year}-{month:02d}-SM2"
    else:
        # 현재 하반월이면 다음 달의 상반월
        if month == 12:
            # 12월 하반월이면 다음 해 1월 상반월
            semimonthly = f"{year+1}-01-SM1"
        else:
            semimonthly = f"{year}-{(month+1):02d}-SM1"
    
    return semimonthly

# 3. 반월 기간의 시작일과 종료일 계산 함수
def get_semimonthly_date_range(semimonthly_period):
    """
    반월 기간 문자열을 받아 시작일과 종료일을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    tuple
        (시작일, 종료일) - datetime 객체
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월 (1일~15일)
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = pd.Timestamp(year=year, month=month, day=15)
    else:
        # 하반월 (16일~말일)
        start_date = pd.Timestamp(year=year, month=month, day=16)
        _, last_day = calendar.monthrange(year, month)
        end_date = pd.Timestamp(year=year, month=month, day=last_day)
    
    return start_date, end_date

# 4. 다음 반월의 모든 날짜 목록 생성 함수
def get_next_semimonthly_dates(reference_date, original_df):
    """
    참조 날짜 기준으로 다음 반월 기간에 속하는 모든 영업일 목록을 반환하는 함수
    """
    # 다음 반월 기간 계산
    next_period = get_next_semimonthly_period(reference_date)
    
    logger.info(f"Calculating next semimonthly dates from reference: {format_date(reference_date)} → target period: {next_period}")
    
    # 반월 기간의 시작일과 종료일 계산
    start_date, end_date = get_semimonthly_date_range(next_period)
    
    logger.info(f"Target period date range: {format_date(start_date)} ~ {format_date(end_date)}")
    
    # 이 기간에 속하는 영업일(월~금, 휴일 제외) 선택
    business_days = []
    
    # 원본 데이터에서 찾기
    future_dates = original_df.index[original_df.index > reference_date]
    for date in future_dates:
        if start_date <= date <= end_date and date.weekday() < 5 and not is_holiday(date):
            business_days.append(date)
    
    # 원본 데이터에 없는 경우, 날짜 범위에서 직접 생성
    if len(business_days) == 0:
        logger.info(f"No business days found in original data for period {next_period}. Generating from date range.")
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5 and not is_holiday(current_date):
                business_days.append(current_date)
            current_date += pd.Timedelta(days=1)
    
    # 날짜가 없거나 부족하면 추가 로직
    min_required_days = 5
    if len(business_days) < min_required_days:
        logger.warning(f"Only {len(business_days)} business days found in period {next_period}. Creating synthetic dates.")
        
        if business_days:
            synthetic_date = business_days[-1] + pd.Timedelta(days=1)
        else:
            synthetic_date = max(reference_date, start_date) + pd.Timedelta(days=1)
        
        while len(business_days) < 15 and synthetic_date <= end_date:
            if synthetic_date.weekday() < 5 and not is_holiday(synthetic_date):
                business_days.append(synthetic_date)
            synthetic_date += pd.Timedelta(days=1)
        
        logger.info(f"Created synthetic dates. Total business days: {len(business_days)} for period {next_period}")
    
    business_days.sort()
    logger.info(f"Final business days for purchase interval: {len(business_days)} days in {next_period}")
    
    return business_days, next_period

# 5. 다음 N 영업일 계산 함수
def get_next_n_business_days(current_date, original_df, n_days=23):
    """
    현재 날짜 이후의 n_days 영업일을 반환하는 함수 - 원본 데이터에 없는 미래 날짜도 생성
    휴일(주말 및 공휴일)은 제외
    """
    # 현재 날짜 이후의 데이터프레임에서 영업일 찾기
    future_df = original_df[original_df.index > current_date]
    
    # 필요한 수의 영업일 선택
    business_days = []
    
    # 먼저 데이터프레임에 있는 영업일 추가
    for date in future_df.index:
        if date.weekday() < 5 and not is_holiday(date):  # 월~금이고 휴일이 아닌 경우만 선택
            business_days.append(date)
        
        if len(business_days) >= n_days:
            break
    
    # 데이터프레임에서 충분한 날짜를 찾지 못한 경우 합성 날짜 생성
    if len(business_days) < n_days:
        # 마지막 날짜 또는 현재 날짜에서 시작
        last_date = business_days[-1] if business_days else current_date
        
        # 필요한 만큼 추가 날짜 생성
        current = last_date + pd.Timedelta(days=1)
        while len(business_days) < n_days:
            if current.weekday() < 5 and not is_holiday(current):  # 월~금이고 휴일이 아닌 경우만 포함
                business_days.append(current)
            current += pd.Timedelta(days=1)
    
    logger.info(f"Generated {len(business_days)} business days, excluding holidays")
    return business_days

def get_previous_semimonthly_period(semimonthly_period):
    """
    주어진 반월 기간의 이전 반월 기간을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    str
        이전 반월 기간
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월인 경우 이전 월의 하반월로
        if month == 1:
            return f"{year-1}-12-SM2"
        else:
            return f"{year}-{month-1:02d}-SM2"
    else:
        # 하반월인 경우 같은 월의 상반월로
        return f"{year}-{month:02d}-SM1"

# ... 기존 코드 ...

def is_holiday(date):
    """주어진 날짜가 휴일인지 확인하는 함수"""
    # preprocessor에서 관리하는 holidays 사용
    from app.data.preprocessor import holidays
    date_str = format_date(date, '%Y-%m-%d')
    return date_str in holidays