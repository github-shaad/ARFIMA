# ARFIMA Python

A lightweight, fast Python library for modeling long-memory time series using AutoRegressive Fractionally Integrated Moving Average (ARFIMA) and Fractionally Integrated GARCH (FIGARCH) models.

## The Math: ARFIMA 

Traditional ARIMA models require integer differencing, which can erase long-term trends. ARFIMA allows for fractional differencing to capture "long-memory" processes. The model solves the general equation:

$$\phi(L)(1-L)^d y_t = \theta(L)w_t$$

Where:
* $L$ is the lag operator.
* $d$ is the fractional integration parameter ($-0.5 < d < 0.5$).
* $\phi$ and $\theta$ are the Autoregressive (AR) and Moving Average (MA) polynomials.
* $w_t$ represents the white noise.

## The Math: FIGARCH 

Traditional GARCH models assume that volatility shocks decay exponentially. FIGARCH introduces a fractional difference parameter to the variance equation to model hyperbolic decay, capturing long-memory volatility clustering. The conditional variance solves:

$$\sigma_t^2 = \omega^* + \left[ 1 - \frac{\phi(L)}{\beta(L)} (1-L)^{d_v} \right] \epsilon_t^2$$

Where:
* $\sigma_t^2$ is the conditional variance.
* $\omega^*$ is the adjusted baseline variance constant.
* $d_v$ is the fractional volatility parameter ($0 \le d_v \le 1$).
* $\phi(L)$ and $\beta(L)$ are the ARCH and GARCH polynomials.
* $\epsilon_t^2$ represents the squared residuals extracted from the mean equation.

## Key Features

* **Initialization:** Uses the Geweke and Porter-Hudak (GPH) estimator to seed fractional parameters ($d$ and $d_v$) for optimization.
* **Fractional Differencing:** Computes fractional integration and variance convolutions using Fast Fourier Transform (FFT).
* **Model Selection:** Evaluates lag structures using bounded optimization and standard information criteria (AIC, BIC, HQIC, AICc).
* **Inference:** Extracts standard errors, t-statistics, and p-values from the inverse Hessian matrix.
* **Forecasting:** Calculates in-sample fitted values and out-of-sample predictions for both the mean and variance.

## Installation

```bash
pip install arfima
```

[Click here for the documentation](https://github-shaad.github.io/ARFIMA/)

## Changelog
* **v1.1.1**: Added documentation
### [1.1.0] - 2026-09-13 :
**Added**
Added a `FIGARCH` class to model fractionally integrated GARCH processes.
Added `Guweke and Porter Hudak(GPH)` estimator to generate a better seed fractional difference parameter for both `ARFIMA` and `FIGARCH`. 
* **v1.0.5**: Fix import
* **v1.0.4**: Fix import
* **v1.0.3**: Fix import
* **v1.0.2**: Minor bug fixes and docstring improvements.
* **v1.0.1**: Fixed inverted AR/MA polynomials in forecast generation.
* **v1.0.0**: Initial release.
