# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from scipy import stats

sns.set_theme(style="whitegrid")

# -------------------------
# Analyzer 클래스
# -------------------------
class ManufacturingAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def numeric_columns(self):
        return self.df.select_dtypes(include='number').columns.tolist()

    def summary_stats(self, columns=None):
        """기본 기술통계"""
        if columns is None:
            columns = self.numeric_columns()
        return self.df[columns].describe().T

    def correlation_matrix(self, columns=None):
        if columns is None:
            columns = self.numeric_columns()
        return self.df[columns].corr()

    def trend_analysis(self, time_col, target_col, resample=None):
        """시간별 추세 분석. resample 예: '1H','1D','5T' (None이면 groupby time_col)"""
        df = self.df[[time_col, target_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col, target_col])
        df = df.set_index(time_col).sort_index()
        if resample:
            ts = df[target_col].resample(resample).mean()
        else:
            ts = df.groupby(df.index)[target_col].mean()
        return ts

    def spc_chart(self, target_col):
        """SPC 관리도용 중심선, 관리한계 (기본은 평균 ± 3σ)"""
        x = self.df[target_col].dropna()
        mean = x.mean()
        std = x.std(ddof=0)
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        return {"mean": mean, "std": std, "ucl": ucl, "lcl": lcl, "series": x.reset_index(drop=True)}

    def failure_distribution(self, target_col):
        """불량/정상 분포 (비율)"""
        return self.df[target_col].value_counts(dropna=False).rename_axis(target_col).reset_index(name='count').assign(pct=lambda d: d['count']/d['count'].sum())

    def time_between_failures(self, failure_col, time_col, unit='hours'):
        """failure_col이 1 혹은 True로 표시된 행들의 간격(시간단위) 반환"""
        df = self.df[[time_col, failure_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(time_col).dropna(subset=[time_col])
        fails = df[df[failure_col].astype(bool)]
        times = fails[time_col].reset_index(drop=True)
        if len(times) < 2:
            return np.array([])
        diffs = times.diff().dropna()
        secs = diffs.dt.total_seconds()
        if unit == 'hours':
            return secs / 3600.0
        elif unit == 'minutes':
            return secs / 60.0
        elif unit == 'days':
            return secs / 86400.0
        else:
            return secs

    def fit_weibull(self, failure_col, time_col, unit='hours'):
        """Weibull 분포 피팅. 반환: (shape, loc, scale), durations array (unit)"""
        durations = self.time_between_failures(failure_col, time_col, unit=unit)
        if durations.size < 3:
            return None, durations
        # fit using scipy.stats.weibull_min; force loc=0 for meaningful life distribution
        c, loc, scale = stats.weibull_min.fit(durations, floc=0)
        return (c, loc, scale), durations

# -------------------------
# Visualizer 클래스
# -------------------------
class ManufacturingVisualizer:
    def __init__(self):
        plt.rcParams.update({'figure.max_open_warning': 0})

    def plot_correlation(self, corr_df):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        return fig

    def plot_trend(self, ts, title=None):
        fig, ax = plt.subplots(figsize=(10,4))
        ts.plot(marker='o', ax=ax)
        ax.set_title(title or "Time Series Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        return fig

    def plot_spc(self, series, spc_info, title=None):
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(series.index, series.values, marker='o', label='value')
        ax.axhline(spc_info['mean'], color='green', linestyle='--', label='Mean')
        ax.axhline(spc_info['ucl'], color='red', linestyle='--', label='UCL')
        ax.axhline(spc_info['lcl'], color='red', linestyle='--', label='LCL')
        ax.set_title(title or "SPC Chart")
        ax.legend()
        return fig

    def plot_failure_dist(self, dist_df, value_col):
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(dist_df[value_col].astype(str), dist_df['count'])
        ax.set_title("Failure Distribution")
        ax.set_ylabel("Count")
        return fig

    def plot_weibull(self, durations, params, unit='hours'):
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        # histogram + PDF
        ax[0].hist(durations, bins=15, density=True, alpha=0.6)
        x = np.linspace(durations.min()*0.9, durations.max()*1.1, 200)
        c, loc, scale = params
        pdf = stats.weibull_min.pdf(x, c, loc=loc, scale=scale)
        ax[0].plot(x, pdf, label=f'Weibull PDF (c={c:.2f}, scale={scale:.2f})')
        ax[0].set_title(f"Weibull Fit Histogram ({unit})")
        ax[0].legend()

        # survival function plot
        sf = stats.weibull_min.sf(x, c, loc=loc, scale=scale)
        ax[1].plot(x, sf, label="Survival Function")
        ax[1].set_title("Weibull Survival Function")
        ax[1].set_xlabel(f"Time ({unit})")
        ax[1].set_ylabel("Survival Probability")
        ax[1].grid(True)
        return fig

    def plot_anomaly_time_series(self, df_time, feature, anomalies_mask, title=None):
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_time.index, df_time[feature], label=feature)
        if anomalies_mask.any():
            ax.scatter(df_time.index[anomalies_mask], df_time[feature].iloc[anomalies_mask], color='red', marker='x', s=50, label='anomaly')
        ax.set_title(title or f"Anomaly detection - {feature}")
        ax.legend()
        return fig

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Manufacturing Quick Dashboard", layout="wide")
st.title("🛠️ Manufacturing Quick Dashboard — MVP (5 modules)")

st.markdown("""
업로드한 CSV 데이터로 **기술통계 / 상관분석 / 트렌드 / SPC / Weibull(고장수명) / 이상탐지**를 빠르게 확인합니다.  
- 컬럼 수 유동적 대응  
- 시간 컬럼 자동 감지(변환 시도)  
""")

uploaded_file = st.file_uploader("CSV 파일 업로드 (machine dataset)", type=["csv"])
if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 분석을 자동으로 실행합니다.")
    st.stop()

# -------------------------
# 데이터 로드
# -------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"CSV 로드 실패: {e}")
    st.stop()

st.subheader("데이터 미리보기")
st.dataframe(df.head())

analyzer = ManufacturingAnalyzer(df)
viz = ManufacturingVisualizer()

# 기본 컬럼 추출
num_cols = analyzer.numeric_columns()
st.write(f"감지된 수치형 컬럼: {num_cols}")

# -------- 기술통계 & 상관분석 ----------
with st.expander("기술통계 & 상관분석", expanded=True):
    st.write("기술 통계")
    st.dataframe(analyzer.summary_stats().round(4))

    st.write("상관분석(Heatmap)")
    corr = analyzer.correlation_matrix()
    fig = viz.plot_correlation(corr)
    st.pyplot(fig)

# -------- 트렌드 분석 ----------
with st.expander("트렌드 분석 (시계열)", expanded=False):
    # detect datetime-like columns (convertible)
    datetime_candidates = [c for c in df.columns if pd.to_datetime(df[c], errors='coerce').notna().sum() > 0]
    time_col = st.selectbox("시간(타임스템프) 컬럼 선택", options=[None] + datetime_candidates)
    target_col = st.selectbox("트렌드 대상(수치형) 선택", options=[None] + num_cols)
    resample_opt = st.selectbox("리샘플 (선택)", options=[None, '1min', '5min', '15min', '1H', '6H', '1D'])
    if time_col and target_col:
        ts = analyzer.trend_analysis(time_col, target_col, resample=resample_opt)
        st.line_chart(ts)
        fig = viz.plot_trend(ts, title=f"{target_col} over {time_col} (resample={resample_opt})")
        st.pyplot(fig)
    else:
        st.info("시간 컬럼과 대상 컬럼을 선택하세요.")

# -------- SPC ----------
with st.expander("SPC 관리도 (센서값 기반)", expanded=False):
    spc_target = st.selectbox("SPC 대상 컬럼", options=[None] + num_cols, key="spc_target")
    if spc_target:
        spc_info = analyzer.spc_chart(spc_target)
        series = spc_info['series']
        fig = viz.plot_spc(series, spc_info, title=f"SPC - {spc_target}")
        st.pyplot(fig)
        st.write(spc_info)
    else:
        st.info("수치형 컬럼을 선택하세요.")

# -------- 불량/고장 분포 (Failure distribution) ----------
with st.expander("불량/고장 분포", expanded=False):
    # let user pick a categorical/flag column indicating failure
    cat_col = st.selectbox("불량/고장 여부 컬럼 선택 (예: failure, fault, is_bad)", options=[None] + df.columns.tolist(), key="failure_col")
    if cat_col:
        dist_df = analyzer.failure_distribution(cat_col)
        st.dataframe(dist_df)
        fig = viz.plot_failure_dist(dist_df, cat_col)
        st.pyplot(fig)
    else:
        st.info("불량/고장 여부 컬럼을 선택하세요.")

# -------- Weibull (수명 분석) ----------
with st.expander("Weibull 고장/수명 분석", expanded=False):
    weibull_failure_col = st.selectbox("Weibull - failure 컬럼", options=[None] + df.columns.tolist(), key="weib_fail")
    weibull_time_col = st.selectbox("Weibull - time 컬럼", options=[None] + df.columns.tolist(), key="weib_time")
    unit = st.selectbox("시간 단위", options=['hours', 'minutes', 'days'], index=0)
    if weibull_failure_col and weibull_time_col:
        params, durations = analyzer.fit_weibull(weibull_failure_col, weibull_time_col, unit=unit)
        if durations.size == 0:
            st.warning("감지된 고장 이벤트가 충분치 않습니다(2회 미만). 로그에 고장 이벤트가 있는지 확인하세요.")
        elif params is None:
            st.warning("Weibull 적합을 위해 고장 간격이 적어도 3개 이상 필요합니다.")
            st.write("검출된 간격(단위:", unit, "):")
            st.write(durations)
        else:
            st.write(f"Fitted Weibull parameters (shape(c), loc, scale): {params}")
            fig = viz.plot_weibull(durations, params, unit=unit)
            st.pyplot(fig)
    else:
        st.info("failure 컬럼과 time 컬럼을 선택하세요.")

# -------- 이상탐지 (Anomaly Detection) ----------
with st.expander("이상탐지 (IsolationForest)", expanded=False):
    anom_features = st.multiselect("이상탐지에 사용할 수치형 컬럼 선택 (2개 이상 권장)", options=num_cols, default=num_cols[:3])
    contamination = st.slider("이상치 비율 (contamination)", 0.001, 0.2, 0.05, step=0.001)
    anom_time_col = st.selectbox("시간 컬럼(선택, 시계열 플롯용)", options=[None] + df.columns.tolist(), key="anom_time")
    if st.button("이상탐지 실행"):
        if not anom_features:
            st.error("한 개 이상의 수치형 컬럼을 선택하세요.")
        else:
            feat_df = df[anom_features].dropna()
            if feat_df.shape[0] < 10:
                st.warning("샘플 수가 작습니다. 결과가 불안정할 수 있습니다.")
            iso = IsolationForest(contamination=float(contamination), random_state=42)
            preds = iso.fit_predict(feat_df)
            # -1 anomaly, 1 normal
            anom_series = pd.Series(preds, index=feat_df.index).map({1:0, -1:1}).rename("anomaly")
            # merge back to original df by index
            df_out = df.copy()
            df_out['anomaly'] = 0
            df_out.loc[anom_series.index, 'anomaly'] = anom_series.values

            st.write("이상치 개수:", int(df_out['anomaly'].sum()))
            st.dataframe(df_out.loc[df_out['anomaly'] == 1, anom_features + ([anom_time_col] if anom_time_col else [])].head(50))

            # 시계열 플롯(시간 컬럼이 있으면)
            if anom_time_col:
                df_time = df_out[[anom_time_col] + anom_features + ['anomaly']].copy()
                df_time[anom_time_col] = pd.to_datetime(df_time[anom_time_col], errors='coerce')
                df_time = df_time.dropna(subset=[anom_time_col])
                df_time = df_time.set_index(anom_time_col).sort_index()
                for f in anom_features[:3]:  # 상위 3개만 시각화
                    fig = viz.plot_anomaly_time_series(df_time, f, df_time['anomaly'].astype(bool), title=f"Anomaly - {f}")
                    st.pyplot(fig)
            else:
                if len(anom_features) >= 2:
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.scatter(df_out[anom_features[0]], df_out[anom_features[1]], c=df_out['anomaly'], cmap='coolwarm', s=10)
                    ax.set_xlabel(anom_features[0]); ax.set_ylabel(anom_features[1])
                    ax.set_title("Anomaly scatter")
                    st.pyplot(fig)

            # 다운로드
            csv_bytes = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("CSV로 결과 다운로드 (anomaly 컬럼 포함)", data=csv_bytes, file_name="results_with_anomalies.csv", mime="text/csv")

# 사용 팁 / 주의사항

# 시간 컬럼은 가능한 ISO 형식(YYYY-MM-DD HH:MM:SS)이나 변환 가능한 문자열이면 자동 변환 시도함. 변환 실패하면 time_col 옵션에서 보이지 않을 수 있음.

# Weibull 적합을 위해서는 고장 이벤트가 최소 3회 이상 필요. (데이터가 충분치 않으면 경고 표시)

# SPC는 센서 값 기반의 간단한 관리도로 구성(평균 ± 3σ). 운영 규칙(웨스턴 일렉트릭 룰 등)을 적용하려면 추가로 로직 확장 필요.

# 이상탐지는 기본 IsolationForest 사용. 필요하면 LOF, Autoencoder 등 추가 가능.

# 추가로 Cpk / OEE / Pareto 등은 별도 데이터(규격, 생산량, 불량 원인 태그)가 필요