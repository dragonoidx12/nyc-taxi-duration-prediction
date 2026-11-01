
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date, time, datetime

st.set_page_config(
    page_title="NYC Taxi Duration Prediction",
    page_icon="🚕",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).parent / "nyc_taxi_model_v2.joblib"
    )

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

def prepare_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])

    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lon1 = np.radians(df["pickup_longitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.clip(
        np.sin(dlat / 2)**2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2)**2,
        0, 1
    )

    distance = 6371 * 2 * np.arcsin(np.sqrt(a))

    lat_km = np.abs(dlat) * 6371
    lon_km = np.abs(dlon) * 6371 * np.abs(
        np.cos((lat1 + lat2) / 2)
    )

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    X = pd.DataFrame({
        "vendor_id": df["vendor_id"].to_numpy(),
        "passenger_count": df["passenger_count"].to_numpy(),
        "pickup_latitude": df["pickup_latitude"].to_numpy(),
        "pickup_longitude": df["pickup_longitude"].to_numpy(),
        "dropoff_latitude": df["dropoff_latitude"].to_numpy(),
        "dropoff_longitude": df["dropoff_longitude"].to_numpy(),
        "store_and_fwd": df["store_and_fwd_flag"].eq("Y").astype(int).to_numpy(),
        "month": dt.dt.month.to_numpy(),
        "day": dt.dt.day.to_numpy(),
        "weekday": dt.dt.dayofweek.to_numpy(),
        "hour": dt.dt.hour.to_numpy(),
        "minute": dt.dt.minute.to_numpy(),
        "weekend": (dt.dt.dayofweek >= 5).astype(int).to_numpy(),
        "haversine_km": distance,
        "manhattan_km": lat_km + lon_km,
        "bearing": np.degrees(np.arctan2(y, x))
    })

    hour = dt.dt.hour.to_numpy()
    weekday = dt.dt.dayofweek.to_numpy()

    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    X["rush_hour"] = (
        (weekday < 5)
        & (((hour >= 7) & (hour <= 9))
        | ((hour >= 16) & (hour <= 19)))
    ).astype(int)

    X["night_trip"] = (
        (hour >= 22) | (hour <= 5)
    ).astype(int)

    X["distance_log"] = np.log1p(distance)
    X["distance_squared"] = distance ** 2
    X["dayofyear"] = dt.dt.dayofyear.to_numpy()
    X["quarter"] = dt.dt.quarter.to_numpy()

    return X

st.title("🚕 NYC Taxi Trip Duration Prediction")
st.write("HistGradientBoosting + Feature Engineering")
st.metric("Validation RMSLE", "0.4110")

st.info(
    "Model 2016 yılına ait NYC taksi verileriyle eğitilmiştir. "
    "Güncel trafik koşullarını bilmez."
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Yolculuk Bilgileri")
    trip_date = st.date_input(
        "Tarih",
        value=date(2016, 5, 10),
        min_value=date(2016, 1, 1),
        max_value=date(2016, 6, 30)
    )
    trip_time = st.time_input("Saat", value=time(9, 0))
    passengers = st.number_input(
        "Yolcu sayısı", min_value=1, max_value=6, value=1
    )
    vendor = st.selectbox("Taksi sağlayıcısı kodu", [1, 2])
    forwarded = st.checkbox("Store and forward flag")

with c2:
    st.subheader("Konum Bilgileri")

    pickup_lat = st.number_input(
        "Başlangıç enlemi", value=40.7580, format="%.6f"
    )
    pickup_lon = st.number_input(
        "Başlangıç boylamı", value=-73.9855, format="%.6f"
    )
    drop_lat = st.number_input(
        "Varış enlemi", value=40.7300, format="%.6f"
    )
    drop_lon = st.number_input(
        "Varış boylamı", value=-73.9950, format="%.6f"
    )

if st.button("Yolculuk Süresini Tahmin Et", type="primary"):

    row = pd.DataFrame([{
        "vendor_id": vendor,
        "passenger_count": passengers,
        "pickup_datetime": datetime.combine(trip_date, trip_time),
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": drop_lat,
        "dropoff_longitude": drop_lon,
        "store_and_fwd_flag": "Y" if forwarded else "N"
    }])

    X = prepare_features(row)[features]

    if not (
        -90 <= pickup_lat <= 90
        and -90 <= drop_lat <= 90
        and -180 <= pickup_lon <= 180
        and -180 <= drop_lon <= 180
    ):
        st.error("Geçerli koordinatlar girin.")
    else:
        prediction = max(
            0, float(np.expm1(model.predict(X)[0]))
        )

        st.success(
            f"Tahmini yolculuk süresi: {prediction / 60:.1f} dakika"
        )
        st.metric("Tahmini süre (saniye)", f"{prediction:.0f}")
        st.caption(
            "Bu bir model tahminidir, gerçek zamanlı rota veya "
            "trafik bilgisi değildir."
        )

st.caption(
    "Kaynak: Kaggle New York City Taxi Trip Duration"
)
