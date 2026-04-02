import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Download gold prices (spot gold)
data = yf.download("GC=F", start="2020-01-01", end="2025-11-01")
prices = data['Close'].dropna()

# Split into train/test (80/20)
split = int(len(prices) * 0.8)
train, test = prices[:split], prices[split:]

# ======================
# 1. Autoregressive (AR)
# ======================
ar_model = AutoReg(train, lags=30)
ar_fit = ar_model.fit()
ar_pred = ar_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# ======================
# 2. Moving Average (MA)
# ======================
ma_model = ARIMA(train, order=(0, 0, 30))  # (p=0, d=0, q=5)
ma_fit = ma_model.fit()
ma_pred = ma_fit.predict(start=len(train), end=len(train) + len(test) - 1)

# ======================
# 3. ARMA (AR + MA)
# ======================
arma_model = ARIMA(train, order=(30, 0, 30))  # (p=5, d=0, q=5)
arma_fit = arma_model.fit()
arma_pred = arma_fit.predict(start=len(train), end=len(train) + len(test) - 1)

# ======================
# Plot results and export to PDF
# ======================
with PdfPages('gold_price_models.pdf') as pdf:
    # AR Model Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train.index, train, label='Train', color='blue')
    ax1.plot(test.index, test, label='Test', color='orange')
    ax1.plot(test.index, ar_pred, label='Predicted', color='red')
    ax1.set_title('AR(30) Model')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close()

    # MA Model Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(train.index, train, label='Train', color='blue')
    ax2.plot(test.index, test, label='Test', color='orange')
    ax2.plot(test.index, ma_pred, label='Predicted', color='red')
    ax2.set_title('MA(30) Model')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close()

    # ARMA Model Plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(train.index, train, label='Train', color='blue')
    ax3.plot(test.index, test, label='Test', color='orange')
    ax3.plot(test.index, arma_pred, label='Predicted', color='red')
    ax3.set_title('ARMA(30,30) Model')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price')
    ax3.legend()
    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close()

print("Plots exported to 'gold_price_models.pdf'")
print(arma_fit.summary())
