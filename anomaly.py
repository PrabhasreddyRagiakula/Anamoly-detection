import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import plotly.graph_objects as go

# Function to fit ARIMA model and get residuals
def fit_arima_model(data, order=(5, 1, 0)):
    """Fits ARIMA model to the data and returns the residuals."""
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    residuals = fitted_model.resid
    return residuals

# Function to detect anomalies using Isolation Forest on ARIMA residuals
def detect_anomalies(residuals, contamination=0.1):
    """Detects anomalies in the ARIMA residuals using Isolation Forest."""
    residuals = pd.Series(residuals)
    residuals_reshaped = residuals.values.reshape(-1, 1)
    model = IsolationForest(contamination=contamination)
    model.fit(residuals_reshaped)
    scores = model.decision_function(residuals_reshaped)
    anomalies = pd.Series(scores < 0, index=residuals.index)  # Anomalies flagged as True
    return anomalies

# Function to decompose the time series into trend, seasonal, and residual components
def decompose_time_series(data, column):
    """Decomposes the time series into trend, seasonal, and residual components."""
    decomposition = seasonal_decompose(data[column], model='additive', period=12)  # Adjust 'period' based on your data frequency
    return decomposition

# Function to visualize seasonal decomposition with explanations
def visualize_decomposition_with_explanation(decomposition):
    """Visualizes the decomposed components and explains them."""
    st.subheader("Seasonal Decomposition of Time Series")
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(decomposition.observed, label="Observed")
    axs[0].set_title("Observed (Original Data)")
    
    axs[1].plot(decomposition.trend, label="Trend", color='orange')
    axs[1].set_title("Trend (Long-term Variation)")
    
    axs[2].plot(decomposition.seasonal, label="Seasonal", color='green')
    axs[2].set_title("Seasonal (Periodic, Repeating Patterns)")
    
    axs[3].plot(decomposition.resid, label="Residual", color='red')
    axs[3].set_title("Residual (Short-term Unexplained Variation)")
    
    plt.tight_layout()
    st.pyplot(fig)

    st.write("**Trend**: Long-term direction of the data.")
    st.write("**Seasonal**: Captures periodic, repeating patterns within the data.")
    st.write("**Residual**: Represents short-term, random variation.")

# Function to forecast using Prophet
def forecast_with_prophet(df, periods):
    """Forecasts future values using the Prophet model."""
    prophet_df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Function to visualize anomalies and extra data points interactively
def visualize_anomalies(data, anomalies, column, chart_type, threshold=None):
    """Visualizes the detected anomalies based on the chosen chart type interactively."""
    if threshold is not None:
        filtered_data = data[data[column] < threshold]
        filtered_anomalies = anomalies[filtered_data.index]
    else:
        filtered_data = data
        filtered_anomalies = anomalies

    st.subheader("Filtered Data Below Threshold")
    st.dataframe(filtered_data)

    st.subheader("Anomaly Data")
    anomaly_data = filtered_data[filtered_anomalies]
    if not anomaly_data.empty:
        st.dataframe(anomaly_data)
    else:
        st.write("No anomalies detected in the filtered data.")

    # Interactive Scatter Plot
    if chart_type == "Scatter Plot":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data[column],
            mode='markers',
            name='Normal Data',
            marker=dict(color='blue'),
            hoverinfo='text',
            text=[f'Index: {idx}<br>{column}: {val}' for idx, val in zip(filtered_data.index, filtered_data[column])]
        ))
        fig.add_trace(go.Scatter(
            x=anomaly_data.index,
            y=anomaly_data[column],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red'),
            hoverinfo='text',
            text=[f'Index: {idx}<br>{column}: {val}' for idx, val in zip(anomaly_data.index, anomaly_data[column])]
        ))
        fig.update_layout(title=f"Anomaly Detection on {column} (Scatter Plot)", xaxis_title="Data Point Index", yaxis_title=column)
        st.plotly_chart(fig)

    # Interactive Pie Chart
    elif chart_type == "Pie Chart":
        anomaly_count = len(anomaly_data)
        normal_count = len(filtered_data) - anomaly_count
        labels = ["Normal Data", "Anomalies"]
        sizes = [normal_count, anomaly_count]

        hover_template = ""
        if not anomaly_data.empty:
            for idx, row in anomaly_data.iterrows():
                hover_template += f"<b>Index:</b> {idx}<br><b>{column}:</b> {row[column]}<br>"

        pie_fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=sizes, 
            hoverinfo='label+percent',
            textinfo='value',
            hovertemplate=hover_template if anomaly_count > 0 else "No anomalies detected"
        )])
        pie_fig.update_layout(title=f"Anomaly Detection on {column} (Pie Chart)")
        st.plotly_chart(pie_fig)

# Function to visualize Prophet forecast
def visualize_forecast(forecast):
    """Visualizes the forecast generated by Prophet with interactive Plotly graph."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='lightblue', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='lightblue', dash='dash')))
    
    fig.update_traces(hoverinfo='text', text=forecast.apply(lambda row: f'Date: {row["ds"].date()}<br>Forecast: {row["yhat"]:.2f}', axis=1))

    fig.update_layout(
        title='Prophet Forecast',
        xaxis_title='Date',
        yaxis_title='Forecasted Value',
        hovermode='closest',
        showlegend=True
    )
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("Time Series Forecasting and Anomaly Detection with ARIMA and Isolation Forest")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        # Select column for anomaly detection
        column = st.selectbox("Select column for anomaly detection", df.columns)

        # Ensure column is selected before running anomaly detection and visualization
        if column:
            # Fit ARIMA model and get residuals
            p = st.number_input("ARIMA(p):", min_value=0, value=5)
            d = st.number_input("ARIMA(d):", min_value=0, value=1)
            q = st.number_input("ARIMA(q):", min_value=0, value=0)
            residuals = fit_arima_model(df[column], order=(p, d, q))

            # Get user input for contamination parameter (for anomaly detection)
            contamination = st.number_input("Contamination (proportion of outliers)", min_value=0.0, max_value=0.5, step=0.01, value=0.1)

            # Detect anomalies in ARIMA residuals
            anomalies = detect_anomalies(residuals, contamination)

            # Decompose the time series to visualize seasonal, trend, and residual components
            decomposition = decompose_time_series(df, column)
            visualize_decomposition_with_explanation(decomposition)

            # Forecasting with Prophet
            periods = st.number_input("Number of periods to forecast", min_value=1, value=10)
            forecast = forecast_with_prophet(df, periods)

            # Display the forecast
            st.subheader("Prophet Forecast")
            visualize_forecast(forecast)

            # Get the threshold value for filtering the data
            threshold = st.number_input(f"Threshold for {column} (Filter values below)", value=float(df[column].max()))

            # Choose the type of chart to display
            chart_type = st.selectbox("Select chart type for visualization", ["Scatter Plot", "Pie Chart"])

            # Visualize anomalies with filtered data
            visualize_anomalies(df, anomalies, column, chart_type, threshold)

if __name__ == "__main__":
    main()
