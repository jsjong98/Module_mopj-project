import pandas as pd
import numpy as np
import calendar
import re
import logging
import os
import traceback

from pathlib import Path

# holidays 전역 변수 사용 (아래 holiday 관련 함수는 여기에 배치)
from app.config import HOLIDAY_DIR
from app.utils.date_utils import format_date
from app.utils.file_utils import set_seed

logger = logging.getLogger(__name__)  # logger 정의 추가

holidays = set()

def create_proper_column_names(file_path, sheet_name):
    from app.data.loader import safe_read_excel
    """헤더 3행을 읽어서 적절한 열 이름 생성"""
    # 헤더 3행을 읽어옴
    header_rows = safe_read_excel(file_path, sheet_name=sheet_name, header=None, nrows=3)
    
    # 각 열별로 적절한 이름 생성
    column_names = []
    prev_main_category = None  # 이전 메인 카테고리 저장
    
    for col_idx in range(header_rows.shape[1]):
        values = [str(header_rows.iloc[i, col_idx]).strip() 
                 for i in range(3) 
                 if pd.notna(header_rows.iloc[i, col_idx]) and str(header_rows.iloc[i, col_idx]).strip() != 'nan']
        
        # 첫 번째 행의 값이 있으면 메인 카테고리로 저장
        if pd.notna(header_rows.iloc[0, col_idx]) and str(header_rows.iloc[0, col_idx]).strip() != 'nan':
            prev_main_category = str(header_rows.iloc[0, col_idx]).strip()
        
        # 열 이름 생성 로직
        if 'Date' in values:
            column_names.append('Date')
        else:
            # 값이 하나도 없는 경우
            if not values:
                column_names.append(f'Unnamed_{col_idx}')
                continue
                
            # 메인 카테고리가 있고, 현재 값들에 포함되지 않은 경우 추가
            if prev_main_category and prev_main_category not in values:
                values.insert(0, prev_main_category)
            
            # 특수 케이스 처리 (예: WS, Naphtha 등)
            if 'WS' in values and 'SG-Korea' in values:
                column_names.append('WS_SG-Korea')
            elif 'Naphtha' in values and 'Platts' in values:
                column_names.append('Naphtha_Platts_' + '_'.join([v for v in values if v not in ['Naphtha', 'Platts']]))
            else:
                column_names.append('_'.join(values))
    
    return column_names

def remove_high_missing_columns(data, threshold=70):
    """높은 결측치 비율을 가진 열 제거"""
    missing_ratio = (data.isnull().sum() / len(data)) * 100
    high_missing_cols = missing_ratio[missing_ratio >= threshold].index
    
    print(f"\n=== {threshold}% 이상 결측치가 있어 제거될 열 목록 ===")
    for col in high_missing_cols:
        print(f"- {col}: {missing_ratio[col]:.1f}%")
    
    cleaned_data = data.drop(columns=high_missing_cols)
    print(f"\n원본 데이터 형태: {data.shape}")
    print(f"정제된 데이터 형태: {cleaned_data.shape}")
    
    return cleaned_data

def clean_text_values_advanced(data):
    """고급 텍스트 값 정제 (쉼표 소수점 처리 포함)"""
    cleaned_data = data.copy()
    
    def fix_comma_decimal(value_str):
        """쉼표로 된 소수점을 점으로 변경하는 함수"""
        if not isinstance(value_str, str) or ',' not in value_str:
            return value_str
            
        import re
        
        # 패턴 1: 단순 소수점 쉼표 (예: "123,45")
        if re.match(r'^-?\d+,\d{1,3}$', value_str):
            return value_str.replace(',', '.')
            
        # 패턴 2: 천 단위 구분자 + 소수점 쉼표 (예: "1.234,56")
        if re.match(r'^-?\d{1,3}(\.\d{3})*,\d{1,3}$', value_str):
            # 마지막 쉼표만 소수점으로 변경
            last_comma_pos = value_str.rfind(',')
            return value_str[:last_comma_pos] + '.' + value_str[last_comma_pos+1:]
            
        # 패턴 3: 쉼표만 천 단위 구분자로 사용 (예: "1,234,567")
        if re.match(r'^-?\d{1,3}(,\d{3})+$', value_str):
            return value_str.replace(',', '')
            
        return value_str
    
    def process_value(x):
        if pd.isna(x):  # 이미 NaN인 경우
            return x
        
        # 문자열로 변환하여 처리
        x_str = str(x).strip()
        
        # 1. 먼저 쉼표 소수점 문제 해결
        x_str = fix_comma_decimal(x_str)
        
        # 2. 휴일/미발표 데이터 처리
        if x_str.upper() in ['NOP', 'NO PUBLICATION', 'NO PUB']:
            return np.nan
            
        # 3. TBA (To Be Announced) 값 처리 - 특별 마킹하여 나중에 전날값으로 대체
        if x_str.upper() in ['TBA', 'TO BE ANNOUNCED']:
            return 'TBA_REPLACE'
            
        # 4. '*' 포함된 계산식 처리
        if '*' in x_str:
            try:
                # 계산식 실행
                return float(eval(x_str.replace(' ', '')))
            except:
                return x
        
        # 5. 숫자로 변환 시도
        try:
            return float(x_str)
        except:
            return x

    # 쉼표 처리 통계를 위한 변수
    comma_fixes = 0
    
    # 각 열에 대해 처리
    for column in cleaned_data.columns:
        if column != 'Date':  # Date 열 제외
            # 처리 전 쉼표가 있는 값들 확인
            before_comma_count = cleaned_data[column].astype(str).str.contains(',', na=False).sum()
            
            cleaned_data[column] = cleaned_data[column].apply(process_value)
            
            # 처리 후 쉼표가 있는 값들 확인
            after_comma_count = cleaned_data[column].astype(str).str.contains(',', na=False).sum()
            
            if before_comma_count > after_comma_count:
                fixed_count = before_comma_count - after_comma_count
                comma_fixes += fixed_count
                print(f"열 '{column}': {fixed_count}개의 쉼표 소수점을 수정했습니다.")
    
    if comma_fixes > 0:
        print(f"\n총 {comma_fixes}개의 쉼표 소수점을 점으로 수정했습니다.")
    
    # MOPJ 변수 처리 (결측치가 있는 행 제거)
    mopj_columns = [col for col in cleaned_data.columns if 'MOPJ' in col or 'Naphtha_Platts_MOPJ' in col]
    if mopj_columns:
        mopj_col = mopj_columns[0]  # 첫 번째 MOPJ 관련 열 사용
        print(f"\n=== {mopj_col} 변수 처리 전 데이터 크기 ===")
        print(f"행 수: {len(cleaned_data)}")
        
        # 결측치가 있는 행 제거
        cleaned_data = cleaned_data.dropna(subset=[mopj_col])
        
        # 문자열 값이 있는 행 제거
        try:
            pd.to_numeric(cleaned_data[mopj_col], errors='raise')
        except:
            # 숫자로 변환할 수 없는 행 찾기
            numeric_mask = pd.to_numeric(cleaned_data[mopj_col], errors='coerce').notna()
            cleaned_data = cleaned_data[numeric_mask]
        
        print(f"\n=== {mopj_col} 변수 처리 후 데이터 크기 ===")
        print(f"행 수: {len(cleaned_data)}")
    
    # 🔧 TBA 값을 전날 값으로 대체
    tba_replacements = 0
    if 'Date' in cleaned_data.columns:
        # 날짜순으로 정렬 (중요: 전날 값 참조를 위해)
        cleaned_data = cleaned_data.sort_values('Date').reset_index(drop=True)
        
        for column in cleaned_data.columns:
            if column != 'Date':  # Date 열 제외
                # TBA_REPLACE 마킹된 값들 찾기
                tba_mask = cleaned_data[column] == 'TBA_REPLACE'
                tba_indices = cleaned_data[tba_mask].index.tolist()
                
                if tba_indices:
                    print(f"\n[TBA 처리] 열 '{column}'에서 {len(tba_indices)}개의 TBA 값 발견")
                    
                    for idx in tba_indices:
                        # 🔧 개선: 가장 최근의 유효한 값 찾기 (연속 TBA 처리)
                        replacement_value = None
                        source_description = ""
                        
                        # 이전 행들을 거슬러 올라가면서 유효한 값 찾기
                        for prev_idx in range(idx-1, -1, -1):
                            candidate_value = cleaned_data.loc[prev_idx, column]
                            try:
                                if pd.notna(candidate_value) and candidate_value != 'TBA_REPLACE':
                                    replacement_value = float(candidate_value)
                                    days_back = idx - prev_idx
                                    if days_back == 1:
                                        source_description = "전날 값"
                                    else:
                                        source_description = f"{days_back}일 전 값"
                                    break
                            except (ValueError, TypeError):
                                continue
                        
                        # 값 대체 수행
                        if replacement_value is not None:
                            cleaned_data.loc[idx, column] = replacement_value
                            tba_replacements += 1
                            print(f"  - 행 {idx+1}: TBA → {replacement_value} ({source_description})")
                        else:
                            # 유효한 이전 값을 찾을 수 없는 경우
                            cleaned_data.loc[idx, column] = np.nan
                            print(f"  - 행 {idx+1}: TBA → NaN (유효한 이전 값 없음)")
    
    if tba_replacements > 0:
        print(f"\n✅ 총 {tba_replacements}개의 TBA 값을 전날 값으로 대체했습니다.")
    
    return cleaned_data

def fill_missing_values_advanced(data):
    """고급 결측치 채우기 (forward fill + backward fill)"""
    filled_data = data.copy()
    
    # Date 열 제외한 모든 수치형 열에 대해
    numeric_cols = filled_data.select_dtypes(include=[np.number]).columns
    
    # 이전 값으로 결측치 채우기 (forward fill)
    filled_data[numeric_cols] = filled_data[numeric_cols].ffill()
    
    # 남은 결측치가 있는 경우 다음 값으로 채우기 (backward fill)
    filled_data[numeric_cols] = filled_data[numeric_cols].bfill()
    
    return filled_data

def rename_columns_to_standard(data):
    """열 이름을 표준 형태로 변경"""
    column_mapping = {
        'Date': 'Date',
        'Crude Oil_WTI': 'WTI',
        'Crude Oil_Brent': 'Brent',
        'Crude Oil_Dubai': 'Dubai',
        'WS_AG-SG_55': 'WS_55',
        'WS_75.0': 'WS_75',
        'Naphtha_Platts_MOPJ': 'MOPJ',
        'Naphtha_MOPAG': 'MOPAG',
        'Naphtha_MOPS': 'MOPS',
        'Naphtha_Monthly Spread': 'Monthly Spread',
        'LPG_Argus FEI_C3': 'C3_LPG',
        'LPG_C4': 'C4_LPG',
        'Gasoline_FOB SP_92RON': 'Gasoline_92RON',
        'Gasoline_95RON': 'Gasoline_95RON',
        'Ethylene_Platts_CFR NEA': 'EL_CRF NEA',
        'Ethylene_CFR SEA': 'EL_CRF SEA',
        'Propylene_Platts_FOB Korea': 'PL_FOB Korea',
        'Benzene_Platts_FOB Korea': 'BZ_FOB Korea',
        'Benzene_Platts_FOB SEA': 'BZ_FOB SEA',
        'Benzene_Platts_FOB US M1': 'BZ_FOB US M1',
        'Benzene_Platts_FOB US M2': 'BZ_FOB US M2',
        'Benzene_Platts_H2-TIME SPREAD': 'BZ_H2-TIME SPREAD',
        'Toluene_Platts_FOB Korea': 'TL_FOB Korea',
        'Toluene_Platts_FOB US M1': 'TL_FOB US M1',
        'Toluene_Platts_FOB US M2': 'TL_FOB US M2',
        'MX_Platts FE_FOB K': 'MX_FOB Korea',
        'PX_FOB   Korea': 'PX_FOB Korea',
        'SM_FOB   Korea': 'SM_FOB Korea',
        'RPG Value_Calculated_FOB PG': 'RPG Value_FOB PG',
        'FO_Platts_HSFO 180 CST': 'FO_HSFO 180 CST',
        'MTBE_Platts_FOB S\'pore': 'MTBE_FOB Singapore',
        'MTBE_Dow_Jones': 'Dow_Jones',
        'MTBE_Euro': 'Euro',
        'MTBE_Gold': 'Gold',
        'PP (ICIS)_CIF NWE': 'Europe_CIF NWE',
        'PP (ICIS)_M.G.\n10ppm': 'Europe_M.G_10ppm',
        'PP (ICIS)_RBOB (NYMEX)_M1': 'RBOB (NYMEX)_M1',
        'Brent_WTI': 'Brent_WTI',
        'MOPJ_Mopag_Nap': 'MOPJ_MOPAG',
        'MOPJ_MOPS_Nap': 'MOPJ_MOPS',
        'Naphtha_Spread': 'Naphtha_Spread',
        'MG92_E Nap': 'MG92_E Nap',
        'C3_MOPJ': 'C3_MOPJ',
        'C4_MOPJ': 'C4_MOPJ',
        'Nap_Dubai': 'Nap_Dubai',
        'MG92_Nap_mops': 'MG92_Nap_MOPS',
        '95R_92R_Asia': '95R_92R_Asia',
        'M1_M2_RBOB': 'M1_M2_RBOB',
        'RBOB_Brent_m1': 'RBOB_Brent_m1',
        'RBOB_Brent_m2': 'RBOB_Brent_m2',
        'EL': 'EL_MOPJ',
        'PL': 'PL_MOPJ',
        'BZ_MOPJ': 'BZ_MOPJ',
        'TL': 'TL_MOPJ',
        'PX': 'PX_MOPJ',
        'HD': 'HD_EL',
        'LD_EL': 'LD_EL',
        'LLD': 'LLD_EL',
        'PP_PL': 'PP_PL',
        'SM_EL+BZ_Margin': 'SM_EL+BZ',
        'US_FOBK_BZ': 'US_FOBK_BZ',
        'NAP_HSFO_180': 'NAP_HSFO_180',
        'MTBE_MOPJ': 'MTBE_MOPJ',
        'MTBE_PG': 'Freight_55_PG',
        'MTBE_Maili': 'Freight_55_Maili',
        'Freight (55)_Ruwais_Yosu': 'Freight_55_Yosu',
        'Freight (55)_Daes\'': 'Freight_55_Daes',
        'Freight (55)_Chiba': 'Freight_55_Chiba',
        'Freight (55)_PG': 'Freight_75_PG',
        'Freight (55)_Maili': 'Freight_75_Maili',
        'Freight (75)_Ruwais_Yosu': 'Freight_75_Yosu',
        'Freight (75)_Daes\'': 'Freight_75_Daes',
        'Freight (75)_Chiba': 'Freight_75_Chiba',
        'Freight (75)_PG': 'Flat Rate_PG',
        'Freight (75)_Maili': 'Flat Rate_Maili',
        'Flat Rate_Ruwais_Yosu': 'Flat Rate_Yosu',
        'Flat Rate_Daes\'': 'Flat Rate_Daes',
        'Flat Rate_Chiba': 'Flat Rate_Chiba'
    }
    
    # 실제 존재하는 열만 매핑
    existing_columns = data.columns.tolist()
    final_mapping = {}
    
    for old_name, new_name in column_mapping.items():
        if old_name in existing_columns:
            final_mapping[old_name] = new_name
    
    # 매핑되지 않은 열들 확인
    unmapped_columns = [col for col in existing_columns if col not in column_mapping.keys()]
    if unmapped_columns:
        print(f"\n=== 매핑되지 않은 열들 ===")
        for col in unmapped_columns:
            print(f"- {col}")
    
    # 열 이름 변경
    renamed_data = data.rename(columns=final_mapping)
    
    print(f"\n=== 열 이름 변경 완료 ===")
    print(f"변경된 열 개수: {len(final_mapping)}")
    print(f"최종 데이터 형태: {renamed_data.shape}")
    
    return renamed_data

# process_data_250620.py의 추가 함수들
def remove_missing_and_analyze(data, threshold=10):
    """
    중간 수준의 결측치 비율을 가진 열을 제거하고 분석하는 함수
    (process_data_250620.py에서 가져온 함수)
    """
    # 결측치 비율 계산
    missing_ratio = (data.isnull().sum() / len(data)) * 100
    
    # threshold% 이상 결측치가 있는 열 식별
    high_missing_cols = missing_ratio[missing_ratio >= threshold]
    
    if len(high_missing_cols) > 0:
        logger.info(f"\n=== {threshold}% 이상 결측치가 있어 제거될 열 목록 ===")
        for col, ratio in high_missing_cols.items():
            logger.info(f"- {col}: {ratio:.1f}%")
        
        # 결측치가 threshold% 이상인 열 제거
        cleaned_data = data.drop(columns=high_missing_cols.index)
        logger.info(f"\n원본 데이터 형태: {data.shape}")
        logger.info(f"정제된 데이터 형태: {cleaned_data.shape}")
    else:
        cleaned_data = data
        logger.info(f"\n제거할 {threshold}% 이상 결측치 열 없음: {data.shape}")
    
    return cleaned_data

def find_text_missings(data, text_patterns=['NOP', 'No Publication']):
    """
    문자열 형태의 결측치를 찾는 함수
    (process_data_250620.py에서 가져온 함수)
    """
    logger.info("\n=== 문자열 형태의 결측치 분석 ===")
    
    # 각 패턴별로 검사
    for pattern in text_patterns:
        logger.info(f"\n['{pattern}' 포함된 데이터 확인]")
        
        # 모든 열에 대해 검사
        for column in data.columns:
            # 문자열 데이터만 검사
            if data[column].dtype == 'object':
                # 해당 패턴이 포함된 데이터 찾기
                mask = data[column].astype(str).str.contains(pattern, na=False, case=False)
                matches = data[mask]
                
                if len(matches) > 0:
                    logger.info(f"\n열: {column}")
                    logger.info(f"발견된 횟수: {len(matches)}")

def final_clean_data_improved(data):
    """
    최종 데이터 정제 함수 (process_data_250620.py에서 가져온 함수)
    M1_M2_RBOB 컬럼의 결측치나 'Q' 값을 RBOB_Brent_m1 - RBOB_Brent_m2로 계산해서 채움
    """
    # 데이터 복사본 생성
    cleaned_data = data.copy()
    
    # MTBE_Dow_Jones 열 특별 처리
    for col in ['MTBE_Dow_Jones']:
        if col in cleaned_data.columns:
            # 숫자로 변환 시도
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    # 🔧 M1_M2_RBOB 열 특별 처리: 결측치와 'Q' 값을 계산으로 채우기
    if 'M1_M2_RBOB' in cleaned_data.columns and 'RBOB_Brent_m1' in cleaned_data.columns and 'RBOB_Brent_m2' in cleaned_data.columns:
        logger.info(f"\n=== M1_M2_RBOB 열 처리 시작 ===")
        logger.info(f"처리 전 데이터 타입: {cleaned_data['M1_M2_RBOB'].dtype}")
        logger.info(f"처리 전 결측치 개수: {cleaned_data['M1_M2_RBOB'].isnull().sum()}")
        
        # 'Q' 값들과 기타 문자열 값들을 NaN으로 변환
        original_values = cleaned_data['M1_M2_RBOB'].copy()
        q_count = 0
        other_string_count = 0
        
        # 'Q' 값 개수 확인
        if cleaned_data['M1_M2_RBOB'].dtype == 'object':
            q_mask = cleaned_data['M1_M2_RBOB'].astype(str).str.upper() == 'Q'
            q_count = q_mask.sum()
            
            # 기타 문자열 값들 확인
            numeric_convertible = pd.to_numeric(cleaned_data['M1_M2_RBOB'], errors='coerce')
            string_mask = pd.isna(numeric_convertible) & cleaned_data['M1_M2_RBOB'].notna()
            other_string_count = string_mask.sum() - q_count
            
            if q_count > 0:
                logger.info(f"'Q' 값 {q_count}개 발견")
            if other_string_count > 0:
                logger.info(f"기타 문자열 값 {other_string_count}개 발견")
        
        # 'Q' 값들과 기타 문자열을 NaN으로 변환
        cleaned_data['M1_M2_RBOB'] = cleaned_data['M1_M2_RBOB'].replace('Q', np.nan)
        cleaned_data['M1_M2_RBOB'] = cleaned_data['M1_M2_RBOB'].replace('q', np.nan)
        
        # 문자열로 저장된 숫자들을 실제 숫자로 변환
        cleaned_data['M1_M2_RBOB'] = pd.to_numeric(cleaned_data['M1_M2_RBOB'], errors='coerce')
        
        # 결측치와 'Q' 값들을 계산으로 채우기: M1_M2_RBOB = RBOB_Brent_m1 - RBOB_Brent_m2
        missing_mask = cleaned_data['M1_M2_RBOB'].isnull()
        missing_count_before = missing_mask.sum()
        
        if missing_count_before > 0:
            logger.info(f"결측치 {missing_count_before}개를 계산으로 채웁니다: M1_M2_RBOB = RBOB_Brent_m1 - RBOB_Brent_m2")
            
            # 계산 가능한 행들만 선택 (m1, m2 둘 다 유효한 값이 있는 경우)
            can_calculate = (missing_mask & 
                           cleaned_data['RBOB_Brent_m1'].notna() & 
                           cleaned_data['RBOB_Brent_m2'].notna())
            calculated_count = can_calculate.sum()
            
            if calculated_count > 0:
                # 계산 수행
                calculated_values = (cleaned_data.loc[can_calculate, 'RBOB_Brent_m1'] - 
                                   cleaned_data.loc[can_calculate, 'RBOB_Brent_m2'])
                
                cleaned_data.loc[can_calculate, 'M1_M2_RBOB'] = calculated_values
                logger.info(f"실제로 계산된 값: {calculated_count}개")
                
                # 계산 검증 (처음 5개 값 출력)
                logger.info(f"=== 계산 검증 (처음 5개 계산된 값) ===")
                calculated_rows = cleaned_data[can_calculate].head(5)
                for idx, row in calculated_rows.iterrows():
                    m1_val = row['RBOB_Brent_m1']
                    m2_val = row['RBOB_Brent_m2']
                    calculated_val = row['M1_M2_RBOB']
                    logger.info(f"인덱스 {idx}: {m1_val:.6f} - {m2_val:.6f} = {calculated_val:.6f}")
                    
            else:
                logger.warning("계산 가능한 행이 없습니다 (RBOB_Brent_m1 또는 RBOB_Brent_m2에 결측치가 있음)")
        
        # 처리 후 결과 확인
        missing_count_after = cleaned_data['M1_M2_RBOB'].isnull().sum()
        valid_count = cleaned_data['M1_M2_RBOB'].notna().sum()
        
        logger.info(f"\n=== M1_M2_RBOB 열 처리 후 ===")
        logger.info(f"데이터 타입: {cleaned_data['M1_M2_RBOB'].dtype}")
        logger.info(f"결측치 개수: {missing_count_after}")
        logger.info(f"유효 데이터 개수: {valid_count}")
        logger.info(f"처리된 결측치 개수: {missing_count_before - missing_count_after}")
        
        if valid_count > 0:
            logger.info(f"최소값: {cleaned_data['M1_M2_RBOB'].min():.6f}")
            logger.info(f"최대값: {cleaned_data['M1_M2_RBOB'].max():.6f}")
            logger.info(f"평균값: {cleaned_data['M1_M2_RBOB'].mean():.6f}")
    
    else:
        # 필요한 컬럼이 없는 경우
        missing_cols = []
        for col in ['M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2']:
            if col not in cleaned_data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"M1_M2_RBOB 계산에 필요한 컬럼이 없습니다: {missing_cols}")
    
    return cleaned_data

def clean_and_trim_data(data, start_date='2013-02-06'):
    """
    데이터 정제 및 날짜 범위 조정 함수
    (process_data_250620.py에서 가져온 함수)
    """
    # 시작 날짜 이후의 데이터만 선택
    cleaned_data = data[data['Date'] >= pd.to_datetime(start_date)].copy()
    
    # 기본 정보 출력
    logger.info(f"=== 데이터 처리 결과 ===")
    logger.info(f"원본 데이터 기간: {data['Date'].min()} ~ {data['Date'].max()}")
    logger.info(f"처리된 데이터 기간: {cleaned_data['Date'].min()} ~ {cleaned_data['Date'].max()}")
    logger.info(f"원본 데이터 행 수: {len(data)}")
    logger.info(f"처리된 데이터 행 수: {len(cleaned_data)}")
    
    return cleaned_data

def load_and_process_data_improved(file_path, sheet_name, start_date):
    from app.data.loader import safe_read_excel
    """
    개선된 데이터 로드 및 처리 함수
    (process_data_250620.py에서 가져온 함수)
    """
    # 열 이름 생성
    column_names = create_proper_column_names(file_path, sheet_name)
    
    # 실제 데이터 읽기
    data = safe_read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=3)
    data.columns = column_names
    
    # Date 열 변환
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # 시작 날짜 이후 데이터만 필터링
    data = data[data['Date'] >= start_date]
    
    # 불필요한 열 제거
    data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    
    return data

def process_excel_data_complete(file_path, sheet_name='29 Nov 2010 till todate', start_date='2013-01-04'):
    """
    Excel 데이터를 완전히 처리하는 통합 함수
    (process_data_250620.py의 메인 처리 파이프라인을 함수화)
    """
    try:
        logger.info("=== Excel 데이터 완전 처리 시작 === 📊")
        
        # 1. 데이터 로드 및 기본 처리
        cleaned_data = load_and_process_data_improved(file_path, sheet_name, pd.Timestamp(start_date))
        logger.info(f"초기 데이터 형태: {cleaned_data.shape}")
        
        # 2. 70% 이상 결측치가 있는 열 제거
        final_data = remove_high_missing_columns(cleaned_data, threshold=70)
        
        # 3. 10% 이상 결측치가 있는 열 제거  
        final_cleaned_data = remove_missing_and_analyze(final_data, threshold=10)
        
        # 4. 텍스트 형태의 결측치 처리
        text_patterns = ['NOP', 'No Publication', 'N/A', 'na', 'NA', 'none', 'None', '-']
        find_text_missings(final_cleaned_data, text_patterns)
        
        # 5. 텍스트 값들 정제
        final_cleaned_data_v2 = clean_text_values_advanced(final_cleaned_data)
        
        # 6. 최종 정제
        final_data_clean = final_clean_data_improved(final_cleaned_data_v2)
        
        # 7. 결측치 채우기
        filled_final_data = fill_missing_values_advanced(final_data_clean)
        
        # 8. 날짜 범위 조정
        trimmed_data = clean_and_trim_data(filled_final_data, start_date='2013-02-06')
        
        # 9. 열 이름을 최종 형태로 변경
        final_renamed_data = rename_columns_to_standard(trimmed_data)
        
        logger.info(f"\n=== 최종 결과 ===")
        logger.info(f"최종 데이터 형태: {final_renamed_data.shape}")
        logger.info(f"최종 열 이름들: {len(final_renamed_data.columns)}개")
        
        return final_renamed_data
        
    except Exception as e:
        logger.error(f"Excel 데이터 처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prepare_data(train_data, val_data, sequence_length, predict_window, target_col_idx, augment=False):
    """학습 및 검증 데이터를 시퀀스 형태로 준비"""
    X_train, y_train, prev_train = [], [], []
    for i in range(len(train_data) - sequence_length - predict_window + 1):
        seq = train_data[i:i+sequence_length]
        target = train_data[i+sequence_length:i+sequence_length+predict_window, target_col_idx]
        prev_value = train_data[i+sequence_length-1, target_col_idx]
        X_train.append(seq)
        y_train.append(target)
        prev_train.append(prev_value)
        if augment:
            # 간단한 데이터 증강
            noise = np.random.normal(0, 0.001, seq.shape)
            aug_seq = seq + noise
            X_train.append(aug_seq)
            y_train.append(target)
            prev_train.append(prev_value)
    
    X_val, y_val, prev_val = [], [], []
    for i in range(len(val_data) - sequence_length - predict_window + 1):
        X_val.append(val_data[i:i+sequence_length])
        y_val.append(val_data[i+sequence_length:i+sequence_length+predict_window, target_col_idx])
        prev_val.append(val_data[i+sequence_length-1, target_col_idx])
    
    return map(np.array, [X_train, y_train, prev_train, X_val, y_val, prev_val])

# 데이터에서 평일 빈 날짜를 휴일로 감지하는 함수
def detect_missing_weekdays_as_holidays(df, date_column='Date'):
    """
    데이터프레임에서 평일(월~금)인데 데이터가 없는 날짜들을 휴일로 감지하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        date_column (str): 날짜 컬럼명
    
    Returns:
        set: 감지된 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    if df.empty or date_column not in df.columns:
        return set()
    
    try:
        # 날짜 컬럼을 datetime으로 변환
        df_dates = pd.to_datetime(df[date_column]).dt.date
        date_set = set(df_dates)
        
        # 데이터 범위의 첫 날과 마지막 날
        start_date = min(df_dates)
        end_date = max(df_dates)
        
        # 전체 기간의 모든 평일 생성
        current_date = start_date
        missing_weekdays = set()
        
        while current_date <= end_date:
            # 평일인지 확인 (월요일=0, 일요일=6)
            if current_date.weekday() < 5:  # 월~금
                if current_date not in date_set:
                    missing_weekdays.add(current_date.strftime('%Y-%m-%d'))
            current_date += pd.Timedelta(days=1)
        
        logger.info(f"Detected {len(missing_weekdays)} missing weekdays as potential holidays")
        if missing_weekdays:
            logger.info(f"Missing weekdays sample: {list(missing_weekdays)[:10]}")
        
        return missing_weekdays
        
    except Exception as e:
        logger.error(f"Error detecting missing weekdays: {str(e)}")
        return set()

def load_holidays_from_file(filepath=None):
    from app.data.loader import load_data_safe_holidays, load_csv_safe_with_fallback, safe_read_excel
    """
    CSV 또는 Excel 파일에서 휴일 목록을 로드하는 함수
    
    Args:
        filepath (str): 휴일 목록 파일 경로, None이면 기본 경로 사용
    
    Returns:
        set: 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    # 기본 파일 경로 - holidays 폴더로 변경
    if filepath is None:
        holidays_dir = Path('holidays')
        holidays_dir.mkdir(exist_ok=True)
        filepath = str(holidays_dir / 'holidays.csv')
    
    # 파일 확장자 확인
    _, ext = os.path.splitext(filepath)
    
    # 파일이 존재하지 않으면 기본 휴일 목록 생성
    if not os.path.exists(filepath):
        logger.warning(f"Holiday file {filepath} not found. Creating default holiday file.")
        
        # 기본 2025년 싱가폴 공휴일
        default_holidays = [
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-04-18", 
            "2025-05-01", "2025-05-12", "2025-06-07", "2025-08-09", "2025-10-20", 
            "2025-12-25", "2026-01-01"
        ]
        
        # 기본 파일 생성
        df = pd.DataFrame({'date': default_holidays, 'description': ['Singapore Holiday']*len(default_holidays)})
        
        if ext.lower() == '.xlsx':
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        logger.info(f"Created default holiday file at {filepath}")
        return set(default_holidays)
    
    try:
        from app.data.loader import load_data_safe_holidays, load_csv_safe_with_fallback, safe_read_excel
        # 파일 로드 - 보안 문제를 고려한 안전한 로딩 사용
        if ext.lower() == '.xlsx':
            # Excel 파일의 경우 xlwings 보안 우회 기능 사용
            try:
                df = load_data_safe_holidays(filepath)
            except Exception as e:
                logger.warning(f"⚠️ [HOLIDAYS] xlwings loading failed, using pandas: {str(e)}")
                df = safe_read_excel(filepath)
        else:
            # CSV 파일 로드 - 안전한 fallback 사용
            df = load_csv_safe_with_fallback(filepath)
        
        # 'date' 컬럼이 있는지 확인
        if 'date' not in df.columns:
            logger.error(f"Holiday file {filepath} does not have 'date' column")
            return set()
        
        # 날짜 형식 표준화
        holidays = set()
        for date_str in df['date']:
            try:
                date = pd.to_datetime(date_str)
                holidays.add(date.strftime('%Y-%m-%d'))
            except:
                logger.warning(f"Invalid date format: {date_str}")
        
        logger.info(f"Loaded {len(holidays)} holidays from {filepath}")
        return holidays
        
    except Exception as e:
        logger.error(f"Error loading holiday file: {str(e)}")
        logger.error(traceback.format_exc())
        return set()

# 휴일 정보와 데이터 빈 날짜를 결합하는 함수
def get_combined_holidays(df=None, filepath=None):
    """
    휴일 파일의 휴일과 데이터에서 감지된 휴일을 결합하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임 (빈 날짜 감지용)
        filepath (str): 휴일 파일 경로
    
    Returns:
        set: 결합된 휴일 날짜 집합
    """
    # 휴일 파일에서 휴일 로드
    file_holidays = load_holidays_from_file(filepath)
    
    # 데이터에서 빈 평일 감지
    data_holidays = set()
    if df is not None:
        data_holidays = detect_missing_weekdays_as_holidays(df)
    
    # 두 세트 결합
    combined_holidays = file_holidays.union(data_holidays)
    
    logger.info(f"Combined holidays: {len(file_holidays)} from file + {len(data_holidays)} from data = {len(combined_holidays)} total")
    
    return combined_holidays

# 휴일 정보 업데이트 함수
def update_holidays(filepath=None, df=None):
    """휴일 정보를 재로드하는 함수 (데이터 빈 날짜 포함)"""
    global holidays
    holidays = get_combined_holidays(df, filepath)
    return holidays

def update_holidays_safe(filepath=None, df=None):
    """
    안전한 휴일 정보 업데이트 함수 - xlwings 보안 우회 기능 포함
    """
    global holidays
    
    # XLWINGS_AVAILABLE을 함수 내부에서 import
    from app.data.loader import XLWINGS_AVAILABLE
    
    try:
        # 기본 방식으로 휴일 로드 시도
        holidays = get_combined_holidays(df, filepath)
        logger.info(f"✅ [HOLIDAY_SAFE] Standard holiday loading successful: {len(holidays)} holidays")
        return holidays
        
    except (PermissionError, OSError, pd.errors.ExcelFileError) as e:
        # 파일 접근 오류 시 xlwings로 대체 시도 (Excel 파일만)
        if filepath and filepath.endswith(('.xlsx', '.xls')) and XLWINGS_AVAILABLE:
            logger.warning(f"⚠️ [HOLIDAY_BYPASS] Standard holiday loading failed: {str(e)}")
            logger.info("🔓 [HOLIDAY_BYPASS] Attempting xlwings bypass for holiday file...")
            
            try:
                # xlwings로 휴일 파일 로드
                file_holidays = load_holidays_from_file_safe(filepath)
                
                # 데이터에서 빈 평일 감지 (기존 방식)
                data_holidays = set()
                if df is not None:
                    data_holidays = detect_missing_weekdays_as_holidays(df)
                
                # 두 세트 결합
                holidays = file_holidays.union(data_holidays)
                
                logger.info(f"✅ [HOLIDAY_BYPASS] xlwings holiday loading successful: {len(file_holidays)} from file + {len(data_holidays)} from data = {len(holidays)} total")
                return holidays
                
            except Exception as xlwings_error:
                logger.error(f"❌ [HOLIDAY_BYPASS] xlwings holiday loading also failed: {str(xlwings_error)}")
                # 기본 휴일로 폴백
                logger.info("🔄 [HOLIDAY_FALLBACK] Using default holidays")
                holidays = load_holidays_from_file()  # 기본 파일에서 로드
                return holidays
        else:
            # xlwings를 사용할 수 없으면 기본 휴일로 폴백
            logger.warning(f"⚠️ [HOLIDAY_FALLBACK] Cannot use xlwings, using default holidays: {str(e)}")
            holidays = load_holidays_from_file()  # 기본 파일에서 로드
            return holidays

def load_holidays_from_file_safe(filepath):
    """
    xlwings를 사용한 안전한 휴일 파일 로딩 (CSV 및 Excel 지원)
    """
    import os  # 필요한 경우 추가
    
    try:
        # 함수 내부에서 import (순환 참조 방지)
        from app.data.loader import load_data_safe_holidays, load_csv_safe_with_fallback
        # 파일 확장자 확인하여 적절한 xlwings 함수 사용
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() in ['.xlsx', '.xls']:
            # Excel 파일의 경우 기존 함수 사용
            df = load_data_safe_holidays(filepath)
        else:
            # CSV 파일의 경우 안전한 fallback 사용
            logger.info(f"🔓 [HOLIDAY_CSV_SAFE] Loading CSV holiday file with xlwings: {os.path.basename(filepath)}")
            df = load_csv_safe_with_fallback(filepath)
            
            # CSV의 경우 컬럼명 정규화 (Excel은 load_data_safe_holidays에서 처리됨)
            df.columns = df.columns.str.lower()
            
            # 필수 컬럼 확인
            if 'date' not in df.columns:
                first_col = df.columns[0]
                df = df.rename(columns={first_col: 'date'})
                logger.info(f"🔄 [HOLIDAY_CSV_SAFE] Renamed '{first_col}' to 'date'")
            
            # description 컬럼이 없으면 추가
            if 'description' not in df.columns:
                df['description'] = 'Holiday'
                logger.info(f"➕ [HOLIDAY_CSV_SAFE] Added default 'description' column")
        
        # 날짜 형식 표준화
        holidays_set = set()
        for date_str in df['date']:
            try:
                date = pd.to_datetime(date_str)
                holidays_set.add(date.strftime('%Y-%m-%d'))
            except:
                logger.warning(f"Invalid date format in xlwings holiday data: {date_str}")
        
        logger.info(f"🔓 [HOLIDAY_XLWINGS_SAFE] Loaded {len(holidays_set)} holidays with xlwings ({ext.lower()} file)")
        return holidays_set
        
    except Exception as e:
        logger.error(f"❌ [HOLIDAY_XLWINGS_SAFE] xlwings holiday loading failed: {str(e)}")
        raise e
    
# 변수 그룹 정의 (preprocessor에 두는 것이 적합)
variable_groups = { # app_rev.py에서 그대로 가져옴
    'crude_oil': ['WTI', 'Brent', 'Dubai', 'Brent_Singapore'],
    'gasoline': ['Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'],
    'naphtha': ['MOPAG', 'MOPS', 'Europe_CIF NWE'],
    'lpg': ['C3_LPG', 'C4_LPG'],
    'product': ['EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2',
    'MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 'FO_HSFO 180 CST', 'MTBE_FOB Singapore'],
    'spread': ['biweekly Spread','BZ_H2-TIME SPREAD', 'Brent_WTI', 'MOPJ_MOPAG', 'MOPJ_MOPS', 'Naphtha_Spread', 'MG92_E Nap', 'C3_MOPJ', 'C4_MOPJ', 'Nap_Dubai',
    'MG92_Nap_MOPS', '95R_92R_Asia', 'M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2', 'EL_MOPJ', 'PL_MOPJ', 'BZ_MOPJ', 'TL_MOPJ', 'PX_MOPJ', 'HD_EL', 'LD_EL', 'LLD_EL', 'PP_PL',
    'SM_EL+BZ', 'US_FOBK_BZ', 'NAP_HSFO_180', 'MTBE_MOPJ'],
    'economics': ['Dow_Jones', 'Euro', 'Gold', 'Exchange'],
    'freight': ['Freight_55_PG', 'Freight_55_Maili', 'Freight_55_Yosu', 'Freight_55_Daes', 'Freight_55_Chiba',
    'Freight_75_PG', 'Freight_75_Maili', 'Freight_75_Yosu', 'Freight_75_Daes', 'Freight_75_Chiba', 'Flat Rate_PG', 'Flat Rate_Maili', 'Flat Rate_Yosu', 'Flat Rate_Daes',
    'Flat Rate_Chiba'],
    'ETF': ['DIG', 'DUG', 'IYE', 'VDE', 'XLE']
}

def calculate_group_vif(df, variables):
    """그룹 내 변수들의 VIF 계산"""
    # 변수가 한 개 이하면 VIF 계산 불가
    if len(variables) <= 1:
        return pd.DataFrame({
            "Feature": variables,
            "VIF": [1.0] * len(variables)
        })
    
    # 모든 변수가 데이터프레임에 존재하는지 확인
    available_vars = [var for var in variables if var in df.columns]
    if len(available_vars) <= 1:
        return pd.DataFrame({
            "Feature": available_vars,
            "VIF": [1.0] * len(available_vars)
        })
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = available_vars
        vif_data["VIF"] = [variance_inflation_factor(df[available_vars].values, i) 
                          for i in range(len(available_vars))]
        return vif_data.sort_values('VIF', ascending=False)
    except Exception as e:
        logger.error(f"Error calculating VIF: {str(e)}")
        # 오류 발생 시 기본값 반환
        return pd.DataFrame({
            "Feature": available_vars,
            "VIF": [float('nan')] * len(available_vars)
        })

def analyze_group_correlations(df, variable_groups, target_col='MOPJ'):
    """그룹별 상관관계 분석"""
    logger.info("Analyzing correlations for each group:")
    group_correlations = {}
    
    for group_name, variables in variable_groups.items():
        # 각 그룹의 변수들과 타겟 변수의 상관관계 계산
        # 해당 그룹의 변수들이 데이터프레임에 존재하는지 확인
        available_vars = [var for var in variables if var in df.columns]
        if not available_vars:
            logger.warning(f"Warning: No variables from {group_name} group found in dataframe")
            continue
            
        if target_col not in df.columns:
            logger.warning(f"Warning: Target column {target_col} not found in dataframe")
            continue
            
        correlations = df[available_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        group_correlations[group_name] = correlations
        
        logger.info(f"\n{group_name} group correlations with {target_col}:")
        logger.info(str(correlations))
    
    return group_correlations

def select_features_from_groups(df, variable_groups, target_col='MOPJ', vif_threshold=50.0, corr_threshold=0.8):
    """각 그룹에서 대표 변수 선택"""
    selected_features = []
    selection_process = {}
    
    logger.info(f"\nCorrelation threshold: {corr_threshold}")
    
    for group_name, variables in variable_groups.items():
        logger.info(f"\nProcessing {group_name} group:")
        
        # 해당 그룹의 변수들이 df에 존재하는지 확인
        available_vars = [var for var in variables if var in df.columns]
        if not available_vars:
            logger.warning(f"Warning: No variables from {group_name} group found in dataframe")
            continue
            
        if target_col not in df.columns:
            logger.warning(f"Warning: Target column {target_col} not found in dataframe")
            continue
        
        # 그룹 내 상관관계 계산
        correlations = df[available_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        logger.info(f"\nCorrelations with {target_col}:")
        logger.info(str(correlations))
        
        # 상관관계가 임계값 이상인 변수만 필터링
        high_corr_vars = correlations[correlations >= corr_threshold].index.tolist()
        
        if not high_corr_vars:
            logger.warning(f"Warning: No variables in {group_name} group meet the correlation threshold of {corr_threshold}")
            continue
        
        # 상관관계 임계값을 만족하는 변수들에 대해 VIF 계산
        if len(high_corr_vars) > 1:
            vif_data = calculate_group_vif(df[high_corr_vars], high_corr_vars)
            logger.info(f"\nVIF values for {group_name} group (high correlation vars only):")
            logger.info(str(vif_data))
            
            # VIF 기준 적용하여 다중공선성 낮은 변수 선택
            low_vif_vars = vif_data[vif_data['VIF'] < vif_threshold]['Feature'].tolist()
            
            if low_vif_vars:
                # 낮은 VIF 변수들 중 상관관계가 가장 높은 변수 선택
                for var in correlations.index:
                    if var in low_vif_vars:
                        selected_var = var
                        break
                else:
                    selected_var = high_corr_vars[0]
            else:
                selected_var = high_corr_vars[0]
        else:
            selected_var = high_corr_vars[0]
            vif_data = pd.DataFrame({"Feature": [selected_var], "VIF": [1.0]})
        
        # 선택된 변수가 상관관계 임계값을 만족하는지 확인 (안전장치)
        if correlations[selected_var] >= corr_threshold:
            selected_features.append(selected_var)
            
            selection_process[group_name] = {
                'selected_variable': selected_var,
                'correlation': correlations[selected_var],
                'all_correlations': correlations.to_dict(),
                'vif_data': vif_data.to_dict() if not vif_data.empty else {},
                'high_corr_vars': high_corr_vars
            }
            
            logger.info(f"\nSelected variable from {group_name}: {selected_var} (corr: {correlations[selected_var]:.4f})")
        else:
            logger.info(f"\nNo variable selected from {group_name}: correlation threshold not met")
    
    # 상관관계 기준 재확인 (최종 안전장치)
    final_features = []
    for feature in selected_features:
        corr = abs(df[feature].corr(df[target_col]))
        if corr >= corr_threshold:
            final_features.append(feature)
            logger.info(f"Final selection: {feature} (corr: {corr:.4f})")
        else:
            logger.info(f"Excluded: {feature} (corr: {corr:.4f}) - below threshold")
    
    # 타겟 컬럼이 포함되어 있지 않으면 추가
    if target_col not in final_features:
        final_features.append(target_col)
        logger.info(f"Added target column: {target_col}")
    
    # 최소 특성 수 확인
    if len(final_features) < 3:
        logger.warning(f"Selected features ({len(final_features)}) < 3, lowering threshold to 0.5")
        return select_features_from_groups(df, variable_groups, target_col, vif_threshold, 0.5)
    
    return final_features, selection_process
