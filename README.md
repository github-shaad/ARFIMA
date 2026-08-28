# ARFIMA Python

A lightweight, fast Python library for modeling long-memory time series using AutoRegressive Fractionally Integrated Moving Average (ARFIMA) models.

## The Math

Traditional ARIMA models require integer differencing, which can erase long-term trends. ARFIMA allows for fractional differencing to capture "long-memory" processes. The model solves the general equation:

$$\phi(L)(1-L)^d y_t = \theta(L)w_t$$

Where:
* $L$ is the lag operator.
* $d$ is the fractional integration parameter ($$-0.5 < d < 0.5$$).
* $\phi$ and $\theta$ are the Autoregressive (AR) and Moving Average (MA) polynomials.
* $w_t$ represents the white noise shocks (residuals).

## Key Features

* **Fast Fractional Differencing:** Uses Fast Fourier Transform (FFT) for fractional integration and differencing.
* **Automatic Grid Search:** Automatically finds the optimal lag structure using information criteria (AIC, BIC, HQIC, or AICc).
* **Statistical Inference:** Calculates the inverse Hessian matrix to provide standard errors, t-statistics, and exact p-values.
* **Forecasting:** Built-in methods for both in-sample historical reconstruction and out-of-sample predictions.

## Installation

```bash
pip install arfima