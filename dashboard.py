import streamlit as st
import pandas as pd
import numpy as np
from collections import deque
from kafka import KafkaConsumer
import json
import mlflow
from streamlit_autorefresh import st_autorefresh
import os
import sys

# Add project root for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.config_loader import load_config

# --------------------------------------------------------------------------
# Initial Page Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Power Demand Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("⚡ 실시간 전력 수요 예측 대시보드")

# --------------------------------------------------------------------------
# Load Resources
# --------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """Loads configuration and the inference model."""
    try:
        config = load_config("config/config.yml")
        local_model_path = config['deployment']['local_model_dir']
        model = mlflow.pyfunc.load_model(local_model_path)
        st.success("모델을 성공적으로 로드했습니다.")
        return config, model
    except Exception as e:
        st.error(f"리소스 로딩 중 오류 발생: {e}")
        return None, None

config, model = load_resources()
if not all([config, model]):
    st.warning("필수 리소스가 로드되지 않았습니다. 대시보드를 중지합니다.")
    st.stop()

# --------------------------------------------------------------------------
# Initialize Kafka Consumer
# --------------------------------------------------------------------------
@st.cache_resource
def init_consumer(config):
    """Initializes the Kafka Consumer."""
    try:
        kafka_config = config['inference']['kafka']
        consumer = KafkaConsumer(
            kafka_config['topic_name'],
            bootstrap_servers=[kafka_config['bootstrap_servers']],
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            consumer_timeout_ms=1000,
        )
        return consumer
    except Exception as e:
        st.error(f"Kafka Consumer 연결 실패: {e}")
        return None

consumer = init_consumer(config)
if not consumer:
    st.stop()

# --------------------------------------------------------------------------
# Session State Setup
# --------------------------------------------------------------------------
if "realtime_data" not in st.session_state:
    seq_len = config['training']['model_params']['seq_len']
    seed_path = config['inference']['buffer_seed_data_path']
    df = pd.read_csv(seed_path)
    init_seq = df.tail(seq_len).copy()
    init_seq["date"] = pd.to_datetime(init_seq["date"])
    init_seq["date"] += pd.to_timedelta(range(len(init_seq)), unit="h")
    init_seq["predicted_demand"] = np.nan
    st.session_state.realtime_data = init_seq[["date", "demand", "predicted_demand"]]

if "future_predictions" not in st.session_state:
    st.session_state.future_predictions = pd.DataFrame(columns=["date", "predicted_demand"])

# --------------------------------------------------------------------------
# Auto-refresh
# --------------------------------------------------------------------------
st_autorefresh(interval=config['dashboard']['refresh_interval_ms'], key="datarefresh")

# --------------------------------------------------------------------------
# Future Forecasting Logic
# --------------------------------------------------------------------------
def generate_future_forecasts(latest_data_df, model, config):
    seq_len = config['training']['model_params']['seq_len']
    forecast_hours = config['dashboard']['forecast_hours']
    
    last_sequence = latest_data_df["demand"].values[-seq_len:]
    buffer = deque(last_sequence, maxlen=seq_len)
    future_preds_list = []

    for _ in range(forecast_hours):
        input_df = pd.DataFrame(list(buffer), columns=["demand"])
        predicted_df = model.predict(input_df)
        predicted_actual = predicted_df.iloc[0][0]
        future_preds_list.append(predicted_actual)
        buffer.append(predicted_actual)

    last_date = latest_data_df["date"].iloc[-1]
    future_dates = [last_date + pd.Timedelta(hours=i) for i in range(1, forecast_hours + 1)]
    
    return pd.DataFrame({"date": future_dates, "predicted_demand": future_preds_list})

# --------------------------------------------------------------------------
# Poll Kafka and Update Data
# --------------------------------------------------------------------------
new_messages = consumer.poll(timeout_ms=1000, max_records=100)
seq_len = config['training']['model_params']['seq_len']

if new_messages:
    for _, messages in new_messages.items():
        for msg in messages:
            new_data = msg.value
            new_date = pd.to_datetime(new_data["date"])

            last_sequence_df = st.session_state.realtime_data.tail(seq_len).copy()
            model_input_df = pd.DataFrame(last_sequence_df["demand"])
            
            predicted_df = model.predict(model_input_df)
            predicted_demand = predicted_df.iloc[0][0]

            new_row = pd.DataFrame({
                "date": [new_date],
                "demand": [new_data["demand"]],
                "predicted_demand": [predicted_demand],
            })
            st.session_state.realtime_data = pd.concat([st.session_state.realtime_data, new_row], ignore_index=True)

    st.session_state.future_predictions = generate_future_forecasts(st.session_state.realtime_data, model, config)

elif st.session_state.future_predictions.empty:
    st.session_state.future_predictions = generate_future_forecasts(st.session_state.realtime_data, model, config)

# --------------------------------------------------------------------------
# Metrics and Charts
# --------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 실시간 수요 및 예측 현황")

display_df = st.session_state.realtime_data.tail(200)

if not display_df.empty:
    col1, col2 = st.columns(2)
    last_row = display_df.iloc[-1]

    with col1:
        st.metric(
            label=f"🕒 최종 실제 수요량 ({last_row['date'].strftime('%Y-%m-%d %H:%M')})",
            value=f"{last_row['demand']:,.2f} MWh",
        )

    with col2:
        last_pred = last_row["predicted_demand"]
        if pd.notna(last_pred) and len(display_df) > 1:
            prev_actual = display_df.iloc[-2]["demand"]
            st.metric(
                label=f"🔮 최종 예측 수요량 ({last_row['date'].strftime('%Y-%m-%d %H:%M')})",
                value=f"{last_pred:,.2f} MWh",
                delta=f"{last_pred - prev_actual:,.2f} (vs 이전 실제값)",
                delta_color="inverse",
            )
        else:
            st.metric(label="🔮 최종 예측 수요량", value="N/A")

    # Anomaly Detection
    chart_data = display_df.dropna(subset=['predicted_demand']).copy()
    chart_data['error'] = chart_data['demand'] - chart_data['predicted_demand']
    chart_data['anomaly'] = chart_data['error'].abs() > (chart_data['demand'] * 0.10) # 10% threshold

    # Display warning for anomalies
    anomalies = chart_data[chart_data['anomaly']]
    if not anomalies.empty:
        st.warning("🚨 이상치 감지! 예측값과 실제값의 차이가 10% 이상입니다.")
        st.dataframe(anomalies[['date', 'demand', 'predicted_demand', 'error']].set_index('date'))

    # Charting
    chart_data_to_plot = chart_data.set_index("date")[["demand", "predicted_demand"]]
    st.line_chart(chart_data_to_plot)

    # Highlight anomalies on the chart
    if not anomalies.empty:
        anomaly_points = anomalies.set_index('date')[['demand']]
        anomaly_points.columns = ['anomaly']
        st.line_chart(pd.concat([chart_data_to_plot, anomaly_points], axis=1))

st.markdown("---")

if not st.session_state.future_predictions.empty:
    st.subheader(f"📈 미래 {config['dashboard']['forecast_hours']}시간 수요 예측")
    future_chart_data = st.session_state.future_predictions.set_index("date")
    st.line_chart(future_chart_data["predicted_demand"])

with st.expander("데이터 보기"):
    st.dataframe(display_df.tail(10).set_index("date"))