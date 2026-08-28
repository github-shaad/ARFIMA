import numpy as np
from scipy import signal
from scipy.optimize import minimize
import scipy.stats as stats
from numpy.typing import NDArray
"""
AutoRegressive Fractionally Integrated Moving Average Time Series Library
"""


class _TimeSeries:
    """
    Wrapper for a numpy array as a Time Series. 
    """
    def __init__(self, data:NDArray[np.float64]):
        self.values = np.array(data, dtype=float)
        self.length = len(self.values)


class ARFIMA:
    """
    ARFIMA(p, d, q) Estimator.
    
    Models time series data featuring long-memory processes through fractional 
    integration, where the difference parameter $d$ is non-integer, solving the 
    general equation $\phi(L)(1-L)^d y_t = \theta(L) w_t$.
    """
    def __init__(self):
        self.p : int = 0
        self.q : int = 0
        self.d : float = 0
        self.ar_coeffs = []
        self.ma_coeffs = []

        self.is_fitted = False

    def __repr__(self):
        return f"ARFIMA(p={self.p}, d={self.d}, q={self.q})"

    def fit(self, time_series:_TimeSeries, p_list:list[int], q_list:list[int], method:str="L-BFGS-B",
            criterion:str='aic'):
        """        
        Fits the optimal ARFIMA structure using Maximum Likelihood Estimation (MLE) via a grid search.
                
                Evaluates combinations of AR and MA lags to minimize the selected information criterion. 
                Upon convergence, calculates the inverse Hessian matrix to derive standard errors, 
                t-statistics, and p-values for statistical inference.

                Parameters
                ----------
                time_series : _TimeSeries
                    The 1D time series data object to fit.
                p_list : list[int]
                    A list of autoregressive (AR) lag orders to test (e.g., [0, 1, 2]).
                q_list : list[int]
                    A list of moving average (MA) lag orders to test (e.g., [0, 1]).
                method : str, default="L-BFGS-B"
                    The SciPy optimization algorithm to use. 
                    Options:
                    * "L-BFGS-B" (Default, highly recommended for bounded parameters)
                    * "TNC"
                    * "SLSQP"
                    * "Powell"
                criterion : str, default='aic'
                    The information criterion used to penalize model complexity.
                    Options:
                    * "aic"  : Akaike Information Criterion
                    * "bic"  : Bayesian Information Criterion
                    * "hqic" : Hannan-Quinn Information Criterion
                    * "aicc" : Corrected AIC 
        """
        raw_data = time_series.values
        best_ic_score = np.inf
        best_params = None
        n = len(raw_data)

        ic_formulas = {
            'aic': lambda n, k, neg_ll: 2 * k + 2 * neg_ll,
            'bic': lambda n, k, neg_ll: k * np.log(n) + 2 * neg_ll,
            'hqic': lambda n, k, neg_ll: 2 * k * np.log(np.log(n)) + 2 * neg_ll,
            'aicc': lambda n, k, neg_ll: (2 * k + 2 * neg_ll) + (2 * k**2 + 2 * k) / (n - k - 1) if (n - k - 1) > 0 else np.inf
        }

        criterion = criterion.lower()

        if criterion not in ic_formulas:
            raise ValueError(f"Invalid Information Criteria {criterion}. Choose from {", ".join(ic_formulas.keys())}")

        for p in p_list:
            for q in q_list:

                initial_guess = [0.1] + [0.0]*p + [0.0]*q
                bounds = [(-0.49,0.49)] + [(-0.99,0.99)]*(p+q)

                res = minimize(fun=_score_function,
                               x0=initial_guess,
                               args=(raw_data, p),
                               method=method,
                               bounds=bounds)

                if res.success:
                    k = p+q+1
                    negative_likelihood = res.fun
                    ic_score = ic_formulas[criterion](n, k, negative_likelihood)

                    if ic_score < best_ic_score:
                        best_ic_score = ic_score
                        self.p = p
                        self.q = q
                        best_params = res.x
                else:
                    print(f"Failed to fit ARFIMA({p}, d, {q}): {res.message}") 

        if best_params is not None:
                    final_initial_guesses = [0.1] + [0.0] * self.p + [0.0] * self.q
                    final_bnds = [(-0.49, 0.49)] + [(-0.99, 0.99)] * (self.p + self.q)
                    
                    final_result = minimize(
                        fun=_score_function, 
                        x0=final_initial_guesses, 
                        args=(raw_data, self.p), 
                        method='L-BFGS-B', 
                        bounds=final_bnds
                    )

                    best_params = final_result.x
                    self.d = best_params[0]
                    self.ar_coeffs = best_params[1 : self.p + 1] if self.p > 0 else []
                    self.ma_coeffs = best_params[self.p + 1 :] if self.q > 0 else []
                    self.is_fitted = True
                    

                    inv_hessian = final_result.hess_inv.todense()
                    variances = np.diag(inv_hessian)
                    self.std_errors = np.sqrt(np.abs(variances))
                    self.t_stats = best_params / self.std_errors 
                    self.p_values = 2 * stats.norm.sf(np.abs(self.t_stats))
                    print(f"\nFinal Fit Complete: ARFIMA({self.p}, {self.d:.4f}, {self.q})")
                    
    def summary(self):
        """
        Prints a formatted statistical regression table for the fitted model.
        
        Outputs the estimated parameters (fractional differencing $d$, AR, and MA coefficients), 
        along with their corresponding standard errors, t-statistics, and two-tailed p-values 
        derived from the inverse Hessian matrix.

        Returns
        -------
        None
            This method prints directly to the console and does not return any objects.
            
        Prints
        ------
        str
            A formatted ASCII table containing the model structure and statistical inference metrics.
        """
        if not self.is_fitted:
            print("Model is not fitted yet.")
            return
            
        print("\n" + "="*50)
        print(f"{'ARFIMA MODEL SUMMARY':^50}")
        print("="*50)
        print(f"Model Structure: ARFIMA({self.p}, d, {self.q})")
        print("-" * 50)
        
        print(f"{'Parameter':<10} | {'Coef':>8} | {'Std Err':>8} | {'t-stat':>8} | {'P>|t|':>8}")
        print("-" * 50)              
        
        def print_row(name, index):
            coef = getattr(self, 'd') if name == 'd' else (self.ar_coeffs[index-1] if 'AR' in name else self.ma_coeffs[index-1 - self.p])
            actual_index = 0 if name == 'd' else index
            
            print(f"{name:<10} | {coef:>8.4f} | {self.std_errors[actual_index]:>8.4f} | {self.t_stats[actual_index]:>8.4f} | {self.p_values[actual_index]:>8.4f}")

        print_row('d', 0)
        
        for i in range(1, self.p + 1):
            print_row(f'AR({i})', i)
            
        for j in range(1, self.q + 1):
            print_row(f'MA({j})', self.p + j)
            
        print("="*50 + "\n")

    def predict(self, time_series:_TimeSeries, steps=0):
        """
        Forecasts future values of the time series out-of-sample.
        
        Mathematically projects the series forward by assuming all future 
        random shocks ($w_t$) are exactly zero. The projected shocks are then 
        fractionally integrated using $-d$ to restore the original scale of the data.

        Parameters
        ----------
        time_series : _TimeSeries
            The historical 1D time series data object used as the baseline for the forecast.
        steps : int, default=0
            The number of future time periods to forecast.

        Returns
        -------
        np.ndarray
            A 1D array of length `steps` containing the out-of-sample forecasted values.
            
        Raises
        ------
        ValueError
            If the method is called before the model has been successfully fitted.
        """
        if not self.is_fitted:
            raise ValueError("ARFIMA not fitted yet")

        y_history = _frac_diff(time_series, self.d)
        w_history = _fast_residuals(y_history, self.ar_coeffs, self.ma_coeffs)

        w_future = np.zeros(steps)
        w_full = np.concatenate((w_history, w_future))

        y_full = _fast_residuals(w_full, self.ma_coeffs, self.ar_coeffs)
        integrated_series = _frac_diff(_TimeSeries(y_full), -self.d)

        return integrated_series[-steps:]
    
    def predict_in_sample(self, time_series):
        """
        Generates in-sample fitted values for the historical data.
        
        Reconstructs the model's historical fit by calculating the exact historical 
        residuals (shocks) via linear filtering, and subtracting those residuals 
        from the raw input data.

        Parameters
        ----------
        time_series : _TimeSeries
            The 1D time series data object that the model was trained on.

        Returns
        -------
        np.ndarray
            A 1D array of fitted values, equal in length to the input time series.
            
        Raises
        ------
        ValueError
            If the method is called before the model has been successfully fitted.
        """
        if not self.is_fitted:
            raise ValueError("ARFIMA not fitted yet")
            
        raw_data = time_series.values
        
        ar_poly = [1.0] + [-phi for phi in self.ar_coeffs]
        ma_poly = [1.0] + list(self.ma_coeffs)
        
        y_history = _frac_diff(time_series, self.d)
        
        w_history = np.asarray(signal.lfilter(ar_poly, ma_poly, y_history))
        
        fitted_values = raw_data - w_history
        
        return fitted_values

def _score_function(params, raw_data, p):
    n = len(raw_data)
    
    d_guess = float(params[0])
    ar_coeffs = params[1 : p + 1]
    ma_coeffs = params[p + 1 :]
    
    ts_obj = _TimeSeries(raw_data)
    y_differenced = _frac_diff(ts_obj, d_guess)
    
    residuals = _fast_residuals(y_differenced, ar_coeffs, ma_coeffs)
    
    variance = np.sum(residuals ** 2) / n
    

    if variance <= 1e-10 or np.isnan(variance):
        return 1e10  
        
    neg_ll = (n / 2) * np.log(2 * np.pi) + (n / 2) * np.log(variance) + (n / 2)
    
    if np.isnan(neg_ll) or np.isinf(neg_ll):
        return 1e10
        
    return neg_ll

def _fast_residuals(diff_data, ar_coeffs, ma_coeffs):
    """
    Computes residuals (nzzz)
    """
    b = [1.0] + [-i for i in ar_coeffs]
    a = [1.0] + list(ma_coeffs)

    residuals = signal.lfilter(b, a, diff_data)

    return np.asarray(residuals)


def _frac_diff(series:_TimeSeries, d:int):
    """
    Fast fractional differencing. Uses fftconvolve to convolute weights and 
    the time series to return the fractionally differenced time series.
    """
    w = [1.0]
    n = series.length
    for i in range(1,n):
        w_i = ((i-1-d) / i)*w[-1]
        w.append(w_i)

    conv_result = signal.fftconvolve(series.values, w, mode="full")

    return conv_result[:n]






    
    

    

