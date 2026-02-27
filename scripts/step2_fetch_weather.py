"""
step0b_fetch_weather.py
========================
Thu thập dữ liệu thời tiết lịch sử Đà Nẵng từ Open-Meteo Archive API.
- Nguồn  : https://open-meteo.com  (miễn phí, không cần API key)
- Toạ độ : 16.0544°N, 108.2022°E (Đà Nẵng)
- Giai đoạn : 2014-01-01 → hôm qua
- Các chỉ số thu thập:
    · Nhiệt độ TB ngày (°C)
    · Nhiệt độ tối thiểu / tối đa (°C)
    · Lượng mưa tổng ngày (mm)
    · Giờ nắng (h)
    · Tốc độ gió TB (km/h)
    · Độ ẩm TB (%)
- Đầu ra:
    · data/processed/weather_danang_daily.csv    – theo ngày
    · data/processed/weather_danang_monthly.csv  – theo tháng (aggregate)
"""

import os
import datetime
import requests
import pandas as pd

# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
LAT        = 16.0544
LON        = 108.2022
START_DATE = "2014-01-01"
END_DATE   = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED    = os.path.join(PROJECT_DIR, 'data', 'processed')
OUT_DAILY    = os.path.join(PROCESSED, 'weather_danang_daily.csv')
OUT_MONTHLY  = os.path.join(PROCESSED, 'weather_danang_monthly.csv')

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# ─── FETCH ────────────────────────────────────────────────────────────────────

def fetch_raw(start: str, end: str) -> dict:
    """Gọi Open-Meteo Archive API và trả về JSON."""
    params = {
        "latitude"  : LAT,
        "longitude" : LON,
        "start_date": start,
        "end_date"  : end,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "sunshine_duration",       # giây → sẽ chuyển sang giờ
            "wind_speed_10m_max",
        ],
        "hourly": ["relative_humidity_2m"],
        "timezone": "Asia/Bangkok",
    }
    print(f"  → Gọi API Open-Meteo ({start} → {end}) ...")
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ─── XỬ LÝ NGÀY ───────────────────────────────────────────────────────────────

def build_daily_df(data: dict) -> pd.DataFrame:
    """Tạo DataFrame theo ngày từ JSON trả về."""
    d = data["daily"]
    df = pd.DataFrame({
        "date"         : pd.to_datetime(d["time"]),
        "temp_mean"    : d["temperature_2m_mean"],
        "temp_min"     : d["temperature_2m_min"],
        "temp_max"     : d["temperature_2m_max"],
        "rainfall_mm"  : d["precipitation_sum"],
        "sunshine_hours": [s / 3600 if s is not None else None
                           for s in d["sunshine_duration"]],
        "wind_speed_kmh": d["wind_speed_10m_max"],
    })

    # ── Độ ẩm: từ hourly → daily average
    h = data["hourly"]
    df_h = pd.DataFrame({
        "dt"     : pd.to_datetime(h["time"]),
        "humidity": h["relative_humidity_2m"],
    })
    df_h["date"] = df_h["dt"].dt.normalize()
    humidity_daily = df_h.groupby("date")["humidity"].mean().reset_index()

    df = df.merge(humidity_daily, on="date", how="left")

    # ── Cờ ngày mưa (rainfall ≥ 1 mm)
    df["rainy_day"] = (df["rainfall_mm"].fillna(0) >= 1).astype(int)

    # ── Điền giá trị khuyết
    df["rainfall_mm"]   = df["rainfall_mm"].fillna(0)
    df["rainy_day"]     = df["rainy_day"].fillna(0).astype(int)
    for col in ["temp_mean", "temp_min", "temp_max",
                "sunshine_hours", "wind_speed_kmh", "humidity"]:
        df[col] = df[col].interpolate(method="linear").ffill().bfill()

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─── AGGREGATE MONTHLY ────────────────────────────────────────────────────────

def build_monthly_df(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp từ daily → monthly."""
    df = df_daily.copy()
    df["month"] = df["date"].dt.to_period("M")

    agg = df.groupby("month").agg(
        temp_mean     = ("temp_mean",     "mean"),
        temp_min      = ("temp_min",      "mean"),
        temp_max      = ("temp_max",      "mean"),
        rainfall_mm   = ("rainfall_mm",   "sum"),
        rainy_days    = ("rainy_day",     "sum"),
        sunshine_hours= ("sunshine_hours","sum"),
        wind_speed_kmh= ("wind_speed_kmh","mean"),
        humidity      = ("humidity",      "mean"),
    ).reset_index()

    agg["month"] = agg["month"].dt.strftime("%Y-%m")
    agg = agg.round(2)
    return agg


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  THU THẬP DỮ LIỆU THỜI TIẾT ĐÀ NẴNG")
    print("  Nguồn: Open-Meteo Archive API (archive-api.open-meteo.com)")
    print("=" * 60)
    print(f"  Giai đoạn : {START_DATE} → {END_DATE}")
    print(f"  Toạ độ    : {LAT}°N, {LON}°E\n")

    # ── Gọi API
    try:
        raw = fetch_raw(START_DATE, END_DATE)
    except requests.exceptions.RequestException as e:
        print(f"[LỖI] Không thể kết nối API: {e}")
        return

    # ── Daily
    print("  Xây dựng bộ dữ liệu theo ngày...")
    df_daily = build_daily_df(raw)

    # ── Monthly
    print("  Tổng hợp theo tháng...")
    df_monthly = build_monthly_df(df_daily)

    # ── Lưu file
    os.makedirs(PROCESSED, exist_ok=True)
    df_daily.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    df_monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")

    # ── Thống kê nhanh
    print("\n--- Kết quả ---")
    print(f"  ✓ Daily   : {len(df_daily):,} ngày  → {OUT_DAILY}")
    print(f"  ✓ Monthly : {len(df_monthly):,} tháng → {OUT_MONTHLY}")
    print(f"\n  Giai đoạn thực tế   : {df_daily['date'].min().date()} → {df_daily['date'].max().date()}")
    print(f"  Nhiệt độ TB         : {df_daily['temp_mean'].mean():.1f} °C")
    print(f"  Lượng mưa TB/tháng  : {df_monthly['rainfall_mm'].mean():.1f} mm")
    print(f"  Số ngày mưa/năm TB  : {df_daily['rainy_day'].sum() / df_daily['date'].dt.year.nunique():.0f} ngày")

    print("\n  Mẫu 5 dòng daily:")
    print(df_daily.tail(5).to_string(index=False))
    print("\n  Mẫu 5 dòng monthly:")
    print(df_monthly.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
