"""
휴일 관련 유틸리티
"""
import logging

logger = logging.getLogger(__name__)

def is_holiday(date):
    """주어진 날짜가 휴일인지 확인하는 함수"""
    from .date_utils import format_date
    date_str = format_date(date, '%Y-%m-%d')
    return date_str in holidays