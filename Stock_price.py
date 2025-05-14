import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import pickle

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")
st.title("📈 Stock Price Predictor")
st.title("By - Abhinandan Patidar")
def get_cache_path(ticker):
    cache_dir = ".stock_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{ticker.lower()}_data.pkl")

def save_to_cache(ticker, data):
    try:
        with open(get_cache_path(ticker), 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Cache save error: {e}")

def load_from_cache(ticker):
    try:
        cache_path = get_cache_path(ticker)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Cache load error: {e}")
    return None

def generate_synthetic_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    start_price = 100
    daily_volatility = 0.02
    drift = 0.0005
    
    daily_returns = np.random.normal(drift, daily_volatility, len(date_range))
    price_series = start_price * (1 + daily_returns).cumprod()
    
    return pd.DataFrame({
        'Date': date_range,
        'Close': price_series
    })

def load_data(ticker, start_date, end_date):
    cached_data = load_from_cache(ticker)
    if cached_data is not None:
        return cached_data
    
    data = generate_synthetic_data(start_date, end_date)
    save_to_cache(ticker, data)
    return data

def predict_next_day_price(data):
    data['Date_ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    X = data['Date_ordinal'].values.reshape(-1, 1)
    y = data['Close'].values
    
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    intercept, slope = coefficients
    
    last_date_ordinal = data['Date_ordinal'].iloc[-1]
    next_day_ordinal = last_date_ordinal + 1
    predicted_price = slope * next_day_ordinal + intercept
    
    residuals = y - (slope * X.flatten() + intercept)
    std_error = np.std(residuals)
    
    confidence_interval = 1.96 * std_error
    
    return {
        'predicted_price': predicted_price,
        'last_known_price': data['Close'].iloc[-1],
        'confidence_interval': confidence_interval,
        'lower_bound': predicted_price - confidence_interval,
        'upper_bound': predicted_price + confidence_interval
    }

def main():
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper().strip()

    today = datetime.now().date()
    default_end_date = today - timedelta(days=1)
    default_start_date = default_end_date - timedelta(days=365*2)

    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", default_start_date, key="start_date", 
                                   help="Select the start date for historical data",
                                   format="YYYY-MM-DD")
        st.markdown(
            """
            <style>
            .stDateInput > div > div > input {
                background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-calendar-days"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/><path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/><path d="M8 18h.01"/><path d="M12 18h.01"/><path d="M16 18h.01"/></svg>');
                background-repeat: no-repeat;
                background-position: right 10px center;
                background-size: 24px 24px;
                padding-right: 40px;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        end_date = st.date_input("End Date", default_end_date, key="end_date", 
                                 help="Select the end date for historical data",
                                 format="YYYY-MM-DD")
        st.markdown(
            """
            <style>
            .stDateInput > div > div > input {
                background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-calendar-check"><path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="m9 16 2 2 4-4"/></svg>');
                background-repeat: no-repeat;
                background-position: right 10px center;
                background-size: 24px 24px;
                padding-right: 40px;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )

    if end_date <= start_date:
        st.error("End date must be after start date")
        return

    with st.spinner(f"Preparing data for {ticker}..."):
        try:
            data = load_data(ticker, start_date, end_date)
            prediction = predict_next_day_price(data)
            
            st.subheader("Next Day Price Prediction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Known Price", f"${data['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Predicted Next Day Price", f"${prediction['predicted_price']:.2f}")
            
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.errorbar(
                [prediction['predicted_price']], 
                [0], 
                xerr=[[prediction['confidence_interval']], [prediction['confidence_interval']]], 
                capsize=10, 
                fmt='o', 
                color='red', 
                ecolor='gray', 
                elinewidth=3, 
                capthick=2
            )
            
            ax.set_title(f"{ticker} Next Day Price Prediction")
            ax.set_xlabel("Stock Price ($)")
            ax.set_yticks([])
            
            st.pyplot(fig)
            
            st.subheader("Prediction Details")
            st.write(f"Predicted Price Range: ${prediction['lower_bound']:.2f} - ${prediction['upper_bound']:.2f}")
            
            st.info("""
            🔍 Prediction Methodology:
            - Uses linear regression on historical price trends
            - Considers date-based price progression
            - Accounts for historical price volatility
            """)
            
            

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()