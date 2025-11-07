import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import calendar # 월 말일 계산을 위해 추가

# --- 1. 기본 설정 및 상수 정의 ---

st.set_page_config(layout="wide")
st.title("KOL 활동 진척률 대시보드 (Pacing 기반)")

# --- 기준일: 월 선택 -> 월 말일로 자동 계산 ---
st.subheader("기준 월 선택")
st.info("선택한 월의 말일을 기준으로 모든 진척률을 계산합니다.")
YEAR = 2025 # 연도 고정

# 월 이름을 숫자로 매핑 (날짜 계산 및 정렬용)
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
MONTH_LIST_SORTED = list(MONTH_MAP.keys())

# 월 선택 슬라이더 (11월 7일이므로 11월 기본 선택)
selected_month_name = st.select_slider(
    "기준 월 (As-of-Month):",
    options=MONTH_LIST_SORTED,
    value="November"
)

selected_month_num = MONTH_MAP[selected_month_name]
# 선택한 월의 마지막 날짜 계산
last_day = calendar.monthrange(YEAR, selected_month_num)[1]
TODAY = pd.to_datetime(datetime.date(YEAR, selected_month_num, last_day))


# 'activity tracking'의 Activity를 'contract'의 Task로 매핑하는 규칙
ACTIVITY_TO_TASK_MAP = {
    'case report': 'Case Report',
    'Lecture': 'Lecture',
    'Article': 'Article',
    'Clinical Paper': 'Article',
    'Webinar': 'Webinar',
    'Testimonial': 'Testimonial',
    'Contents creation': 'SNS Posting',
    'ContentsCreation': 'SNS Posting',
    'Hands-on course': 'Hands-On'
}

# --- 2. 데이터 로딩 및 처리 (캐시 사용) ---

@st.cache_data
def load_data():
    """'contract.csv'와 'tracking.csv'를 직접 로드합니다."""
    try:
        df_plan = pd.read_csv("contract.csv")
        df_actual = pd.read_csv("tracking.csv") 
        
        df_plan['Contract Start'] = pd.to_datetime(df_plan['Contract Start'])
        df_plan['Contract End'] = pd.to_datetime(df_plan['Contract End'])
        
        return df_plan, df_actual
    except FileNotFoundError as e:
        st.error(f"필수 파일 없음: '{e.filename}'") 
        st.info("'contract.csv'와 'tracking.csv' 파일이 .py 파일과 동일한 폴더에 있는지 확인하세요.")
        return None, None
    except Exception as e:
        st.error(f"파일 로드 중 알 수 없는 오류 발생: {e}")
        return None, None

@st.cache_data
def get_dashboard_data(df_plan, df_actual, report_date):
    """
    Pacing Progress (0~100%+) 개념을 도입하여 'Delayed'/'On Track'을 판단합니다.
    """
    
    # --- 2-1. 계획(Plan) 데이터 집계 (계약 기간 포함) ---
    default_start = pd.to_datetime(f"{YEAR}-01-01")
    default_end = pd.to_datetime(f"{YEAR}-12-31")
    
    df_plan['Contract Start'] = df_plan['Contract Start'].fillna(default_start)
    df_plan['Contract End'] = df_plan['Contract End'].fillna(default_end)
    
    kol_master = df_plan.groupby('KOL_ID').agg(
        Name=('Name', 'first'),
        Area=('Area', 'first'),
        Country=('Country', 'first'),
        Contract_Start=('Contract Start', 'min'),
        Contract_End=('Contract End', 'max')
    ).reset_index().dropna(subset=['KOL_ID'])
    
    df_plan_grouped = df_plan.dropna(subset=['KOL_ID', 'Task', 'Frequency'])
    df_plan_grouped = df_plan_grouped.groupby(
        ['KOL_ID', 'Task'], as_index=False
    )['Frequency'].sum()
    df_plan_grouped = df_plan_grouped.rename(columns={'Frequency': 'Target_Count'})
    
    df_plan_master = pd.merge(
        df_plan_grouped,
        kol_master,
        on='KOL_ID',
        how='left'
    )

    # --- 2-2. 실적(Actual) 데이터 집계 (기준일 필터링) ---
    df_actual_processed = df_actual.copy()
    
    df_actual_processed['Month_Num'] = df_actual_processed['Month'].map(MONTH_MAP)
    df_actual_processed['Day'] = df_actual_processed['Week'].str.replace('w', '').astype(int).apply(lambda w: (w-1)*7 + 1)
    df_actual_processed['Year'] = YEAR
    
    df_actual_processed = df_actual_processed.dropna(subset=['Year', 'Month_Num', 'Day'])
    
    df_actual_processed['Activity_Date'] = pd.to_datetime(
        df_actual_processed[['Year', 'Month_Num', 'Day']].rename(columns={'Month_Num': 'Month'})
    )

    # 기준일(report_date) 이전의 실적만 필터링
    df_actual_to_date = df_actual_processed[
        df_actual_processed['Activity_Date'] <= report_date
    ].copy()
    
    df_actual_to_date['Task'] = df_actual_to_date['Activity'].str.strip().map(ACTIVITY_TO_TASK_MAP)
    
    # *** (KeyError 수정) 'KOL_ID'가 NaN이 아닌 실적만 집계
    df_actual_to_date = df_actual_to_date.dropna(subset=['KOL_ID'])
    
    df_actual_counts = df_actual_to_date.dropna(subset=['Task', 'KOL_ID']).groupby(
        ['KOL_ID', 'Task'], as_index=False
    ).size().rename(columns={'size': 'Actual_Count'})

    # --- 2-3. 계획(Plan)과 실적(Actual) 병합 ---
    df_dashboard = pd.merge(
        df_plan_master,
        df_actual_counts,
        on=['KOL_ID', 'Task'],
        how='left'
    )
    df_dashboard['Actual_Count'] = df_dashboard['Actual_Count'].fillna(0).astype(int)
    
    # (수정) KOL_ID가 없는 계획은 제외 (e.g., Area, Country가 NaN인 경우)
    df_dashboard = df_dashboard.dropna(subset=['Area', 'Country'])

    # --- 2-4. (신규) Pacing 진척률 계산 ---
    df_dashboard['Achievement_%'] = (
        (df_dashboard['Actual_Count'] / df_dashboard['Target_Count'])
        .replace([np.inf, -np.inf], 0).fillna(0) * 100
    )
    
    df_dashboard['Total_Contract_Days'] = (df_dashboard['Contract_End'] - df_dashboard['Contract_Start']).dt.days
    df_dashboard['Elapsed_Days'] = (report_date - df_dashboard['Contract_Start']).dt.days
    df_dashboard['Elapsed_Days'] = df_dashboard['Elapsed_Days'].clip(lower=0, upper=df_dashboard['Total_Contract_Days'])
    
    df_dashboard['Elapsed_%'] = 0.0
    valid_days = df_dashboard['Total_Contract_Days'] > 0
    df_dashboard.loc[valid_days, 'Elapsed_%'] = \
        (df_dashboard.loc[valid_days, 'Elapsed_Days'] / df_dashboard.loc[valid_days, 'Total_Contract_Days']) * 100

    df_dashboard['Expected_Count'] = df_dashboard['Target_Count'] * (df_dashboard['Elapsed_%'] / 100.0)

    df_dashboard['Pacing_Progress_%'] = 0.0
    mask_normal = df_dashboard['Expected_Count'] > 0
    df_dashboard.loc[mask_normal, 'Pacing_Progress_%'] = \
        (df_dashboard['Actual_Count'] / df_dashboard['Expected_Count']) * 100.0
        
    mask_not_started = df_dashboard['Expected_Count'] == 0
    df_dashboard.loc[mask_not_started & (df_dashboard['Actual_Count'] > 0), 'Pacing_Progress_%'] = 100.0
    
    def get_status(row):
        if row['Achievement_%'] >= 100:
            return "Completed"
        if row['Target_Count'] == 0:
            return "N/A"
        if row['Elapsed_%'] == 0 and row['Actual_Count'] == 0:
            return "Not Started"
        if row['Pacing_Progress_%'] >= 100:
            return "On Track"
        else:
            return "Delayed"
            
    df_dashboard['Status'] = df_dashboard.apply(get_status, axis=1)
    
    df_dashboard['Gap'] = (df_dashboard['Target_Count'] - df_dashboard['Actual_Count']).apply(lambda x: max(x, 0)).astype(int)
    
    # *** (수정) ID를 정수로 변환 ***
    df_dashboard['KOL_ID'] = df_dashboard['KOL_ID'].astype(int)
    # 실적 데이터의 ID도 정수로 변환 (병합 오류 방지)
    df_actual_to_date['KOL_ID'] = df_actual_to_date['KOL_ID'].astype(int)
    
    return df_dashboard, df_actual_to_date


# --- 3. Altair 차트 헬퍼 함수 ---

def create_donut_chart(percent, title, color_hex):
    percent_value = max(0, min(percent, 1.0))
    source = pd.DataFrame({"category": ["A", "B"], "value": [percent_value, 1.0 - percent_value]})
    base = alt.Chart(source).encode(theta=alt.Theta("value", stack=True))
    pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
        color=alt.Color("category", scale={"domain": ["A", "B"], "range": [color_hex, "#e0e0e0"]}, legend=None),
        order=alt.Order("category", sort="descending")
    )
    text_val = f"{percent_value:.1%}"
    
    text = alt.Chart(pd.DataFrame({'value': [text_val]})).mark_text(
        align='center', baseline='middle', fontSize=30, fontWeight="bold", color=color_hex
    ).encode(text='value')
    return (pie + text).properties(title=title)

def create_pacing_donut(pacing_percent, title, color_map):
    is_delayed = pacing_percent < 100.0
    color = color_map['Delayed'] if is_delayed else color_map['On Track']
    text_color = color_map['Delayed_Text'] if is_delayed else color_map['On Track_Text']
    
    source = pd.DataFrame({"category": ["A", "B"], "value": [1, 0]})
    
    base = alt.Chart(source).encode(theta=alt.Theta("value", stack=True))
    pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
        color=alt.Color("category", scale={"domain": ["A", "B"], "range": [color, "#e0e0e0"]}, legend=None),
    )
    
    text_val = f"{pacing_percent:.1f}%"

    text = alt.Chart(pd.DataFrame({'value': [text_val]})).mark_text(
        align='center', baseline='middle', fontSize=30, fontWeight="bold", color=text_color
    ).encode(text='value')
    return (pie + text).properties(title=title)


def create_pie_chart(data, category_col, value_col, title):
    base = alt.Chart(data).encode(
        theta=alt.Theta(f"{value_col}:Q", stack=True)
    ).properties(title=title)

    pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
        color=alt.Color(f"{category_col}:N"),
        order=alt.Order(f"{value_col}:Q", sort="descending"),
        tooltip=[category_col, value_col]
    )
    return pie

def create_horizontal_bar(data, y_col, x_col, title, color_col, x_title):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{x_col}:Q", title=x_title),
        y=alt.Y(f"{y_col}:N", sort="-x"),
        color=alt.Color(color_col, legend=alt.Legend(title="지역")),
        tooltip=[y_col, color_col, x_col]
    ).properties(
        title=title
    ).interactive()
    return chart

# --- 4. Streamlit 앱 메인 화면 ---

df_plan_raw, df_actual_raw = load_data()

if df_plan_raw is not None and df_actual_raw is not None:
    
    st.success(f"**{TODAY.strftime('%Y년 %m월 %d일')}** (선택 월 말일) 기준으로 데이터를 집계했습니다.")
    
    # 메인 대시보드 데이터 계산
    df_dashboard, df_actual_to_date = get_dashboard_data(df_plan_raw, df_actual_raw, TODAY)
    
    # --- 4-1. KPI 및 시각화 ---
    st.header("종합 진척률 (KPIs)")
    
    col1, col2 = st.columns([1, 2])
    
    # 1. 종합 진척률 (단순 건수 달성률)
    with col1:
        total_actual = df_dashboard['Actual_Count'].sum()
        total_target = df_dashboard['Target_Count'].sum()
        annual_perc = (total_actual / total_target) if total_target > 0 else 0
        
        chart_annual = create_donut_chart(annual_perc, f"종합 진척률 (총 {total_target:.0f}건)", "#008080")
        st.altair_chart(chart_annual, use_container_width=True)

    # 2. 월별 누적 진척률 라인 차트
    with col2:
        st.subheader("월별 누적 진척률 (종합)")
        
        cumulative_data = []
        # (수정) df_dashboard (NaN이 제거된)의 총합을 사용
        total_target_const = df_dashboard['Target_Count'].sum()

        for month_name, month_num in MONTH_MAP.items():
            month_end_day = calendar.monthrange(YEAR, month_num)[1]
            report_date = pd.to_datetime(datetime.date(YEAR, month_num, month_end_day))
            
            rate = 0.0
            if report_date > TODAY:
                if cumulative_data:
                    rate = cumulative_data[-1]['누적 진척률']
            else:
                # get_dashboard_data는 캐시되므로, 원본 데이터가 변경되지 않으면 매우 빠름
                df_dash_month, _ = get_dashboard_data(df_plan_raw, df_actual_raw, report_date)
                total_actual_month = df_dash_month['Actual_Count'].sum()
                rate = (total_actual_month / total_target_const) * 100.0 if total_target_const > 0 else 0.0
            
            cumulative_data.append({'Month': month_name, 'Month_Num': month_num, '누적 진척률': rate})

        df_cumulative = pd.DataFrame(cumulative_data)

        line_chart = alt.Chart(df_cumulative).mark_line(point=True).encode(
            x=alt.X('Month:N', sort=MONTH_LIST_SORTED, title="월"),
            y=alt.Y('누적 진척률:Q', title="누적 진척률 (%)"),
            tooltip=['Month', alt.Tooltip('누적 진척률:Q', format='.1f')]
        ).interactive()
        
        st.altair_chart(line_chart, use_container_width=True)

    st.markdown("---")
    
    # --- 4-2. 지역별 실적 분포 (원그래프) ---
    st.header("지역별 실적 분포 (원그래프)")
    st.info(f"{TODAY.strftime('%Y-%m-%d')}까지 발생한 **실제 활동 건수**의 분포입니다.")

    col_pie_1, col_pie_2 = st.columns(2)

    with col_pie_1:
        # (수정) df_actual_to_date와 df_dashboard의 Area/Country 정보를 병합
        # (df_actual_to_date 자체에는 Area/Country가 없음)
        kol_id_area_map = df_dashboard[['KOL_ID', 'Area']].drop_duplicates()
        area_actuals = pd.merge(df_actual_to_date, kol_id_area_map, on='KOL_ID')
        area_actuals_grouped = area_actuals.groupby('Area', as_index=False).size().rename(columns={'size': '실적 건수'})
        
        chart_area_dist = create_pie_chart(
            area_actuals_grouped, 
            'Area', 
            '실적 건수', 
            "대륙별(Area) 실적 분포"
        )
        st.altair_chart(chart_area_dist, use_container_width=True)

    with col_pie_2:
        kol_id_country_map = df_dashboard[['KOL_ID', 'Country']].drop_duplicates()
        country_actuals = pd.merge(df_actual_to_date, kol_id_country_map, on='KOL_ID')
        country_actuals_grouped = country_actuals.groupby('Country', as_index=False).size().rename(columns={'size': '실적 건수'})
        
        chart_country_dist = create_pie_chart(
            country_actuals_grouped, 
            'Country', 
            '실적 건수', 
            "국가별 실적 분포"
        )
        st.altair_chart(chart_country_dist, use_container_width=True)
        
    st.markdown("---")
    
    # --- 4-3. 지역별 및 개인별 성과 (테이블 및 바 차트) ---
    main_col, side_col = st.columns([2, 1])

    with main_col:
        st.header("지역별 진척률 (테이블)")
        
        format_dict = {
            '단순 달성률 (%)': '{:.1f}%',
            '평균 Pacing (%)': '{:.1f}%'
        }
        
        # 1. 대륙(Area)별 집계
        st.subheader("대륙별(Area) 진척률")
        area_agg = df_dashboard.groupby('Area').agg(
            Target_Count=('Target_Count', 'sum'),
            Actual_Count=('Actual_Count', 'sum')
        ).reset_index()
        area_agg['단순 달성률 (%)'] = (area_agg['Actual_Count'] / area_agg['Target_Count']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        
        area_pacing = df_dashboard[df_dashboard['Status'].isin(['On Track', 'Delayed'])].groupby('Area', as_index=False)['Pacing_Progress_%'].mean()
        area_pacing = area_pacing.rename(columns={'Pacing_Progress_%': '평균 Pacing (%)'})
        
        area_data = pd.merge(area_agg, area_pacing, on='Area', how='left').fillna(0)
        st.dataframe(area_data.style.format(format_dict), use_container_width=True)
        
        # 2. 국가(Country)별 집계
        st.subheader("국가별(Country) 진척률")
        country_agg = df_dashboard.groupby(['Area', 'Country']).agg(
            Target_Count=('Target_Count', 'sum'),
            Actual_Count=('Actual_Count', 'sum')
        ).reset_index()
        country_agg['단순 달성률 (%)'] = (country_agg['Actual_Count'] / country_agg['Target_Count']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        
        country_pacing = df_dashboard[df_dashboard['Status'].isin(['On Track', 'Delayed'])].groupby(['Area', 'Country'], as_index=False)['Pacing_Progress_%'].mean()
        country_pacing = country_pacing.rename(columns={'Pacing_Progress_%': '평균 Pacing (%)'})
        
        country_data = pd.merge(country_agg, country_pacing, on=['Area', 'Country'], how='left').fillna(0)
        st.dataframe(country_data.style.format(format_dict), use_container_width=True)


    with side_col:
        st.header("개인별 진척률 (Pacing)")
        st.info("개인별 [진행중 태스크]의 평균 Pacing 진척률입니다.")
        
        # 3. 개인별 집계 (Pacing 기준)
        personal_data = df_dashboard[
            df_dashboard['Status'].isin(['On Track', 'Delayed'])
        ].groupby(['Name', 'Area'], as_index=False)['Pacing_Progress_%'].mean()
        
        chart_personal = create_horizontal_bar(
            personal_data, 
            'Name', 
            'Pacing_Progress_%', 
            "KOL 개인별 평균 Pacing (%)",
            "Area",
            "평균 Pacing (%)"
        )
        st.altair_chart(chart_personal, use_container_width=True)

    st.markdown("---")

    # --- 4-4. (수정) 미완료 태스크 목록 ---
    st.header("미완료 태스크 목록 (Delayed, On Track, Not Started)")
    st.info(f"{TODAY.strftime('%Y-%m-%d')} 기준, 'Completed'가 아닌 모든 태스크입니다. ('Delayed'가 가장 심각)")
    
    df_incomplete = df_dashboard[
        df_dashboard['Status'] != 'Completed'
    ].sort_values(by='Pacing_Progress_%').reset_index(drop=True) # Pacing 낮은 순 정렬
    
    cols_to_show = [
        'KOL_ID', # ID 추가
        'Name', 'Task', 'Status', 
        'Pacing_Progress_%', 'Achievement_%', 'Elapsed_%',
        'Target_Count', 'Actual_Count', 'Gap'
    ]
    
    # 포맷팅 (소수점 첫째자리)
    format_dict_main = {
        'Pacing_Progress_%': '{:.1f}%',
        'Achievement_%': '{:.1f}%',
        'Elapsed_%': '{:.1f}%',
        'KOL_ID': '{}' # 정수 포맷
    }
    
    st.dataframe(df_incomplete[cols_to_show].style.format(format_dict_main), use_container_width=True)
    
    st.markdown("---")

    # --- 4-5. (기존) 전체 상세 데이터 (필터링) ---
    st.header("전체 태스크 상세 현황 (필터링)")
    
    col_f1, col_f2 = st.columns(2)
    kol_list = df_dashboard['Name'].unique()
    selected_kols = col_f1.multiselect("KOL 선택:", options=kol_list, default=None)
    status_list = df_dashboard['Status'].unique()
    selected_status = col_f2.multiselect("상태 선택:", options=status_list, default=None)

    if selected_kols:
        df_display = df_dashboard[df_dashboard['Name'].isin(selected_kols)]
    else:
        df_display = df_dashboard
        
    if selected_status:
        df_display = df_display[df_display['Status'].isin(selected_status)]

    st.dataframe(df_display.reset_index(drop=True).style.format(format_dict_main), use_container_width=True)