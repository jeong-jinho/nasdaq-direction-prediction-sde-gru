"""
공통 전처리 유틸리티 함수

GRU, SDE 모델 양쪽에서 공통으로 사용하는 문자열→숫자 변환 함수들입니다.
나스닥 원본 CSV의 거래량('10M', '1.5B' 등)과 변동률('1.5%') 컬럼을
숫자형으로 변환할 때 사용합니다.
"""
import numpy as np


def convert_volume(value):
    """거래량 문자열(예: '10M', '5K', '1.2B')을 float 숫자로 변환합니다."""
    if isinstance(value, str):
        value = value.replace(',', '').strip().upper()
        if value == '-' or not value:
            return np.nan
        if value.endswith('K'):
            return float(value[:-1]) * 1_000
        if value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        if value.endswith('B'):
            return float(value[:-1]) * 1_000_000_000
        try:
            return float(value)
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan


def convert_change_percent(value):
    """변동률 문자열(예: '1.5%')을 소수(0.015)로 변환합니다."""
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        if value == '-' or not value:
            return np.nan
        try:
            return float(value) / 100.0
        except ValueError:
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value) / 100.0
    return np.nan
