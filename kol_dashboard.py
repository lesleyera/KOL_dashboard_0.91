import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import calendar # 월 말일 계산을 위해 추가

# --- 1. 기본 설정 및 상수 정의 ---

st.set_page_config(layout="wide")
st.title("KOL 활동 진척률 대시보드 (Pacing 기반)")

# --- (수정) 기준일: 월 선택 -> 월 말일로 자동 계산 ---
YEAR = 2025 # 연도 고정

# 월 이름을 숫자로 매핑 (날짜 계산 및 정렬용)
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
MONTH_LIST_SORTED = list(MONTH_MAP.keys())

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
        
        # (수정) ID가 비어있는 행을 먼저 제거
        df_plan = df_plan.dropna(subset=['KOL_ID'])
        df_actual = df_actual.dropna(subset=['KOL_ID'])
        
        # (수정) ID를 정수로 변환
        df_plan['KOL_ID'] = df_plan['KOL_ID'].astype(int)
        df_actual['KOL_ID'] = df_actual['KOL_ID'].astype(int)
        
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
# (수정) _today 인자를 받도록 수정 (캐시가 날짜별로 동작)
def get_dashboard_data(df_plan, df_actual, _today):
    """
    'report_date' (기준일)을 인자로 받아, 계약 기간 대비 진척률을 계산합니다.
    """
    report_date = _today # 캐시 키로 _today 사용
    
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
    ).reset_index()
    
    df_plan_grouped = df_plan.dropna(subset=['KOL_ID', 'Task', 'Frequency'])
    df_plan_grouped = df_plan_grouped.groupby(
        ['KOL_ID', 'Task'], as_index=False
    )['Frequency'].sum()
    df_plan_grouped = df_plan_grouped.rename(columns={'Frequency': 'Target_Count'})
    
    # (수정) Target_Count를 정수로 변환
    df_plan_grouped['Target_Count'] = df_plan_grouped['Target_Count'].astype(int)
    
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

    df_actual_to_date = df_actual_processed[
        df_actual_processed['Activity_Date'] <= report_date
    ].copy()
    
    df_actual_to_date['Task'] = df_actual_to_date['Activity'].str.strip().map(ACTIVITY_TO_TASK_MAP)
    
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
    
    df_dashboard = df_dashboard.dropna(subset=['KOL_ID', 'Area', 'Country'])
    df_dashboard['KOL_ID'] = df_dashboard['KOL_ID'].astype(int)


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
    
    return df_dashboard, df_actual_to_date, kol_master


# --- 3. Altair 차트 헬퍼 함수 ---

def create_donut_chart(percent, title, color_hex):
    percent_value = max(0, min(percent, 1.0))
    source = pd.DataFrame({"category": ["A", "B"], "value": [percent_value, 1.0 - percent_value]})
    base = alt.Chart(source).encode(theta=alt.Theta("value", stack=True))
    pie = base.mark_arc(outerRadius=50, innerRadius=30).encode(
        color=alt.Color("category", scale={"domain": ["A", "B"], "range": [color_hex, "#e0e0e0"]}, legend=None),
        order=alt.Order("category", sort="descending")
    )
    text_val = f"{percent_value:.1%}"
    
    text = alt.Chart(pd.DataFrame({'value': [text_val]})).mark_text(
        align='center', baseline='middle', fontSize=18, fontWeight="bold", color=color_hex
    ).encode(text='value')
    return (pie + text).properties(title=alt.Title(title, anchor='middle', fontSize=14))


def create_pacing_donut(pacing_percent, title, color_map):
    is_delayed = pacing_percent < 100.0
    color = color_map['Delayed'] if is_delayed else color_map['On Track']
    text_color = color_map['Delayed_Text'] if is_delayed else color_map['On Track_Text']
    
    source = pd.DataFrame({"category": ["A", "B"], "value": [1, 0]})
    
    base = alt.Chart(source).encode(theta=alt.Theta("value", stack=True))
    pie = base.mark_arc(outerRadius=50, innerRadius=30).encode(
        color=alt.Color("category", scale={"domain": ["A", "B"], "range": [color, "#e0e0e0"]}, legend=None),
    )
    
    text_val = f"{pacing_percent:.1f}%"

    text = alt.Chart(pd.DataFrame({'value': [text_val]})).mark_text(
        align='center', baseline='middle', fontSize=18, fontWeight="bold", color=text_color
    ).encode(text='value')
    return (pie + text).properties(title=alt.Title(title, anchor='middle', fontSize=14))


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

def create_horizontal_bar(data, y_col, x_col, title, color_col, x_title, row_col=None):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{x_col}:Q", title=x_title),
        y=alt.Y(f"{y_col}:N", sort="-x"),
        color=alt.Color(color_col, legend=alt.Legend(title="지역")),
        tooltip=[y_col, color_col, x_col]
    ).properties(
        title=title
    ).interactive()
    
    if row_col:
        chart = chart.encode(
            row=alt.Row(f"{row_col}:N", header=alt.Header(titleOrient="top", labelOrient="top"), sort='ascending')
        )
    
    return chart

# --- 4. Streamlit 앱 메인 화면 ---

# (신규) 사이드바 구성
with st.sidebar:
    st.image("https://medit-web-gcs.s3.ap-northeast-2.amazonaws.com/files/2023-01-31/0d273f0d-e461-4c6e-82f5-19e09d17208d/MEDIT_CI_Dark.png", width=150)
    st.title("KOL Dashboard")
    
    page = st.radio(
        "Navigation",
        ["Overview (그래프 중심)", "상세 데이터 (Tables)"],
        label_visibility="hidden"
    )
    
    st.divider()

    # --- (신규) 기준일 선택 (사이드바) ---
    st.subheader("기준 월 선택")
    selected_month_name = st.select_slider(
        "As-of-Month:",
        options=MONTH_LIST_SORTED,
        value="November",
        label_visibility="collapsed"
    )
    selected_month_num = MONTH_MAP[selected_month_name]
    last_day = calendar.monthrange(YEAR, selected_month_num)[1]
    TODAY = pd.to_datetime(datetime.date(YEAR, selected_month_num, last_day))
    
    st.success(f"기준일: **{TODAY.strftime('%Y-%m-%d')}**")

# 데이터 로드
df_plan_raw, df_actual_raw = load_data()

if df_plan_raw is None or df_actual_raw is None:
    st.stop() # 파일 로드 실패 시 앱 중지

# 메인 대시보드 데이터 계산
# (수정) _today=TODAY를 캐시 키로 전달
df_dashboard, df_actual_to_date, kol_master = get_dashboard_data(df_plan_raw, df_actual_raw, TODAY)


# --- 5. (신규) "Overview (그래프 중심)" 페이지 ---
if page == "Overview (그래프 중심)":
    
    # --- 5-1. (신규) 핵심 요약 (KPI) ---
    st.header("핵심 요약 (KPI)")
    
    # KPI 계산
    total_target = df_dashboard['Target_Count'].sum()
    total_actual = df_dashboard['Actual_Count'].sum()
    annual_perc = (total_actual / total_target) if total_target > 0 else 0
    
    # (신규) 월별 Pacing 계산 (누적 평균)
    cumulative_pacing = []
    for month_name, month_num in MONTH_MAP.items():
        month_end_day = calendar.monthrange(YEAR, month_num)[1]
        report_date = pd.to_datetime(datetime.date(YEAR, month_num, month_end_day))
        
        avg_pacing_perc = 0.0
        if report_date <= TODAY:
            df_dash_month, _, _ = get_dashboard_data(df_plan_raw, df_actual_raw, report_date)
            in_progress_tasks = df_dash_month[df_dash_month['Status'].isin(['On Track', 'Delayed'])]
            if not in_progress_tasks.empty:
                avg_pacing_perc = in_progress_tasks['Pacing_Progress_%'].mean()
        cumulative_pacing.append({'Month': month_name, 'Pacing': avg_pacing_perc})
    
    df_pacing_trend = pd.DataFrame(cumulative_pacing)
    # 현재 Pacing
    current_avg_pacing = df_pacing_trend.loc[df_pacing_trend['Month'] == selected_month_name, 'Pacing'].values[0]

    
    delayed_tasks_count = len(df_dashboard[df_dashboard['Status'] == 'Delayed'])
    
    expiry_date_limit = TODAY + pd.Timedelta(days=30)
    expiring_kols_count = len(kol_master[
        (kol_master['Contract_End'] > TODAY) &
        (kol_master['Contract_End'] <= expiry_date_limit)
    ])
    
    # (수정) 4개 컬럼 -> 3개 컬럼 (종합달성률 + 평균 Pacing 합침)
    col1, col2, col3 = st.columns(3)
    
    # (수정) 종합 진척률 -> 월별 평균 Pacing (요청사항 반영)
    with col1:
        st.subheader(f"{selected_month_name} 평균 Pacing")
        st.write("*(진행중 태스크의 진척 속도)*")
        progress_colors = {
            "On Track": "#2E8B57", "On Track_Text": "#2E8B57", # SeaGreen
            "Delayed": "#DC143C", "Delayed_Text": "#DC143C"  # Crimson
        }
        chart_pacing = create_pacing_donut(
            current_avg_pacing, 
            f"{selected_month_name} 평균 Pacing",
            progress_colors
        )
        st.altair_chart(chart_pacing, use_container_width=True)


    with col2:
        st.subheader("종합 달성률 (누적)")
        st.write("*(총 계획 대비 단순 달성 건수)*")
        chart_annual = create_donut_chart(annual_perc, f"총 {total_target:.0f}건 중 {total_actual:.0f}건", "#008080")
        st.altair_chart(chart_annual, use_container_width=True)

    with col3:
        st.subheader("주요 알림")
        st.metric(
            label="지연 태스크 (Delayed)", 
            value=f"{delayed_tasks_count} 건",
            delta_color="inverse"
        )
        st.metric(
            label="계약 만료 임박 (30일 이내)", 
            value=f"{expiring_kols_count} 명",
            delta_color="off"
        )
    
    st.markdown("---")
    
    # --- 5-2. (수정) 월별 누적 달성률 (막대그래프) ---
    st.header("월별 누적 달성률 (막대그래프)")
    
    with st.container(border=True):
        cumulative_data = []
        total_target_const = df_dashboard['Target_Count'].sum()

        for month_name, month_num in MONTH_MAP.items():
            month_end_day = calendar.monthrange(YEAR, month_num)[1]
            report_date = pd.to_datetime(datetime.date(YEAR, month_num, month_end_day))
            
            rate = 0.0
            if report_date > TODAY:
                if cumulative_data: rate = cumulative_data[-1]['달성률']
            else:
                df_dash_month, _, _ = get_dashboard_data(df_plan_raw, df_actual_raw, report_date)
                total_actual_month = df_dash_month['Actual_Count'].sum()
                rate = (total_actual_month / total_target_const) * 100.0 if total_target_const > 0 else 0.0
            
            cumulative_data.append({'Month': month_name, 'Month_Num': month_num, '달성률': rate})

        df_cumulative = pd.DataFrame(cumulative_data)
        
        # (신규) 스택 바 차트용 데이터 가공
        df_stacked_bar = df_cumulative.copy()
        df_stacked_bar['미달성률'] = 100.0 - df_stacked_bar['달성률']
        
        # Melt
        df_melted = df_stacked_bar.melt(
            id_vars=['Month', 'Month_Num'],
            value_vars=['달성률', '미달성률'],
            var_name='유형',
            value_name='비율'
        )
        
        # (신규) 스택 바 차트
        bar_chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('Month:N', sort=MONTH_LIST_SORTED, title="월"),
            y=alt.Y('비율:Q', title="누적 달성률 (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('유형:N', scale={'domain': ['달성률', '미달성률'], 'range': ['#008080', '#DC143C']}),
            order=alt.Order('유형', sort='descending'), # 미달성이 아래에 깔리게
            tooltip=[
                'Month', 
                '유형', 
                alt.Tooltip('비율:Q', format='.1f', title='비율 (%)')
            ]
        ).interactive()
        
        st.altair_chart(bar_chart, use_container_width=True)
        st.info("차트의 빨간색(미달성) 부분에 마우스를 올리면 미달성 비율을 확인할 수 있습니다. '상세 데이터 (Tables)' 탭에서 미달성 태스크의 상세 내역을 확인하세요.")

    
    st.markdown("---")
    
    # --- 5-3. (신규) 월별 캘린더 (히트맵) ---
    st.header(f"{selected_month_name} 월간 활동 캘린더")
    
    with st.container(border=True):
        monthly_schedule = df_actual_raw[
            (df_actual_raw['Month'] == selected_month_name) &
            (df_actual_raw['KOL_ID'].isin(df_dashboard['KOL_ID']))
        ].copy()
        
        if monthly_schedule.empty:
            st.write(f"{selected_month_name}에 예정된 활동이 없습니다.")
        else:
            # (신규) 캘린더 히트맵 차트
            heatmap = alt.Chart(monthly_schedule).mark_rect().encode(
                x=alt.X('Week:N', sort=['1w', '2w', '3w', '4w', '5w'], title="주(Week)"),
                y=alt.Y('Name:N', title="KOL"),
                color=alt.Color('Activity:N', title="활동 유형"),
                tooltip=['Week', 'Name', 'Activity']
            ).properties(
                title=f"{selected_month_name} 활동 히트맵"
            ).interactive()
            st.altair_chart(heatmap, use_container_width=True)

    st.markdown("---")
    
    # --- 5-4. 지역별 및 개인별 성과 ---
    st.header("지역별 성과 분석")
    col_geo_1, col_geo_2 = st.columns(2)
    
    with col_geo_1:
        with st.container(border=True):
            # 1. 대륙(Area)별 집계
            area_agg = df_dashboard.groupby('Area').agg(
                Target_Count=('Target_Count', 'sum'),
                Actual_Count=('Actual_Count', 'sum')
            ).reset_index()
            area_pacing = df_dashboard[df_dashboard['Status'].isin(['On Track', 'Delayed'])].groupby('Area', as_index=False)['Pacing_Progress_%'].mean()
            area_data = pd.merge(area_agg, area_pacing, on='Area', how='left').fillna(0)
            
            # 대륙별 Pacing 바 차트
            chart_area_pacing = create_horizontal_bar(
                area_data,
                'Area',
                'Pacing_Progress_%',
                '대륙별 평균 Pacing (%)',
                'Area',
                '평균 Pacing (%)'
            )
            st.altair_chart(chart_area_pacing, use_container_width=True)
            
    with col_geo_2:
        with st.container(border=True):
            # 2. 국가(Country)별 집계
            country_agg = df_dashboard.groupby(['Area', 'Country']).agg(
                Target_Count=('Target_Count', 'sum'),
                Actual_Count=('Actual_Count', 'sum')
            ).reset_index()
            country_agg['단순 달성률 (%)'] = (country_agg['Actual_Count'] / country_agg['Target_Count']).replace([np.inf, -np.inf], 0).fillna(0) * 100
            
            country_pacing = df_dashboard[df_dashboard['Status'].isin(['On Track', 'Delayed'])].groupby(['Area', 'Country'], as_index=False)['Pacing_Progress_%'].mean()
            country_pacing = country_pacing.rename(columns={'Pacing_Progress_%': '평균 Pacing (%)'})
            
            country_data = pd.merge(country_agg, country_pacing, on=['Area', 'Country'], how='left').fillna(0)
            
            # (신규) 국가별 성과 Scatter Plot
            scatter_plot = alt.Chart(country_data).mark_circle().encode(
                x=alt.X('단순 달성률 (%)', scale=alt.Scale(zero=False)),
                y=alt.Y('평균 Pacing (%)', scale=alt.Scale(zero=False)),
                color='Area',
                size=alt.Size('Target_Count', title='계획 건수'),
                tooltip=['Country', 'Area', 'Target_Count', '단순 달성률 (%)', '평균 Pacing (%)']
            ).properties(
                title="국가별 성과 분석 (Pacing vs 달성률)"
            ).interactive()
            st.altair_chart(scatter_plot, use_container_width=True)

    st.subheader("개인별 Pacing 진척률 (대륙별)")
    with st.container(border=True):
        personal_data = df_dashboard[
            df_dashboard['Status'].isin(['On Track', 'Delayed'])
        ].groupby(['Name', 'Area'], as_index=False)['Pacing_Progress_%'].mean()
        
        chart_personal = create_horizontal_bar(
            personal_data, 
            'Name', 
            'Pacing_Progress_%', 
            "KOL 개인별 평균 Pacing (%)",
            "Area",
            "평균 Pacing (%)",
            row_col='Area' # Area별로 차트 분리
        )
        st.altair_chart(chart_personal, use_container_width=True)


# --- 6. (신규) "상세 데이터 (Tables)" 페이지 ---
elif page == "상세 데이터 (Tables)":
    
    st.header("미완료 태스크 목록 (Delayed, On Track, Not Started)")
    st.info(f"{TODAY.strftime('%Y-%m-%d')} 기준, 'Completed'가 아닌 모든 태스크입니다. ('Delayed'가 가장 심각)")
    
    df_incomplete = df_dashboard[
        df_dashboard['Status'] != 'Completed'
    ].sort_values(by='Pacing_Progress_%').reset_index(drop=True)
    
    cols_to_show = [
        'KOL_ID', 'Name', 'Task', 'Status', 
        'Pacing_Progress_%', 'Achievement_%', 'Elapsed_%',
        'Target_Count', 'Actual_Count', 'Gap'
    ]
    
    format_dict_main = {
        'Pacing_Progress_%': '{:.1f}%',
        'Achievement_%': '{:.1f}%',
        'Elapsed_%': '{:.1f}%',
        'KOL_ID': '{}'
    }
    
    st.dataframe(df_incomplete[cols_to_show].style.format(format_dict_main), use_container_width=True)
    
    st.markdown("---")

    st.header("전체 태스크 상세 현황 (필터링)")
    
    col_f1, col_f2 = st.columns(2)
    kol_list = sorted(df_dashboard['Name'].unique())
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

    
# --- 7. (신규) 하단 이동: 계약 만료 및 KOL 카드 ---
st.markdown("---")
st.header("KOL 상세 정보 및 계약")

col_final_1, col_final_2 = st.columns(2)

with col_final_1:
    st.subheader("KOL 상세 정보 조회")
    
    kol_list_sorted = sorted(df_dashboard['Name'].unique())
    selected_kol = st.selectbox(
        "조회할 의사 선택:", 
        options=kol_list_sorted,
        index=None, 
        placeholder="의사를 선택하세요...",
        label_visibility="collapsed"
    )

    if selected_kol:
        kol_data = df_dashboard[df_dashboard['Name'] == selected_kol].reset_index(drop=True)
        
        if not kol_data.empty:
            kol_info = kol_data.iloc[0]
            
            with st.container(border=True):
                st.subheader(f"닥터 {kol_info['Name']}")
                
                col_info_1, col_info_2 = st.columns(2)
                with col_info_1:
                    st.write(f"**ID:** {kol_info['KOL_ID']}")
                    st.write(f"**지역:** {kol_info['Area']} / {kol_info['Country']}")
                with col_info_2:
                    st.write(f"**계약:** {kol_info['Contract_Start'].strftime('%Y-%m-%d')} ~ {kol_info['Contract_End'].strftime('%Y-%m-%d')}")
                    st.write(f"**경과율:** {kol_info['Elapsed_%']:.1f}%")
                
                st.divider()
                st.write("**계약 활동 및 진척률**")
                
                format_dict_card = {
                    'Pacing_Progress_%': '{:.1f}%',
                    'Target_Count': '{}',
                    'Actual_Count': '{}',
                    'Gap': '{}'
                }
                
                st.dataframe(
                    kol_data[['Task', 'Status', 'Pacing_Progress_%', 'Target_Count', 'Actual_Count', 'Gap']]
                    .style.format(format_dict_card),
                    use_container_width=True
                )

with col_final_2:
    st.subheader("계약 만료 임박 의사 (30일 이내) 🚨")
    
    expiring_kols = kol_master[
        (kol_master['Contract_End'] > TODAY) &
        (kol_master['Contract_End'] <= expiry_date_limit)
    ].sort_values(by='Contract_End')
    
    if expiring_kols.empty:
        st.info("30일 이내 계약 만료 예정 의사가 없습니다.")
    else:
        st.dataframe(expiring_kols[['Name', 'Area', 'Country', 'Contract_End']], use_container_width=True)