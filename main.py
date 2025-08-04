# Weather Data Analysis and Prediction
# Install required libraries if not already installed:
# pip install pandas matplotlib prophet

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Load dataset (replace 'weather.csv' with your file)
# Sample dataset: date,temperature,humidity,wind_speed,precipitation
df = pd.read_csv("weather.csv")

# 2. Data cleaning
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'temperature']]  # Only date & temp for now
df = df.dropna()

# Prophet requires columns: ds (date) and y (value to predict)
df = df.rename(columns={'date': 'ds', 'temperature': 'y'})

# 3. Plot historical data
plt.figure(figsize=(10, 5))
plt.plot(df['ds'], df['y'], label='Historical Temperature')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Historical Temperature Trends")
plt.legend()
plt.show()

# 4. Build Prophet model
model = Prophet()
model.fit(df)

# 5. Make future predictions (next 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 6. Plot forecast
model.plot(forecast)
plt.title("Temperature Prediction")
plt.show()

# 7. Plot forecast components (trend & seasonality)
model.plot_components(forecast)
plt.show()

# 8. Save predictions to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("temperature_forecast.csv", index=False)
print("✅ Prediction saved to temperature_forecast.csv")
