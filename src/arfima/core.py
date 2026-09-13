import numpy as np
from scipy import signal
from scipy.optimize import minimize
import scipy.stats as stats
from numpy.typing import NDArray

"""
AutoRegressive Fractionally Integrated Moving Average Time Series Library.
"""

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
        self.in_sample : NDArray[np.float64] = np.array([], dtype=np.float64)
        self.is_fitted = False

    def __repr__(self):
        return f"ARFIMA(p={self.p}, d={self.d}, q={self.q})"

    def fit(self, time_series:NDArray[np.float64], p_list:list[int], q_list:list[int], method:str="L-BFGS-B",
            criterion:str='aic', returns=True):
        """        
        Fits the optimal ARFIMA structure using Maximum Likelihood Estimation (MLE) via a grid search.
                
                Evaluates combinations of AR and MA lags to minimize the selected information criterion. 
                Upon convergence, calculates the inverse Hessian matrix to derive standard errors, 
                t-statistics, and p-values for statistical inference.

                Parameters
                ----------
                time_series : NDArray
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
        self.in_sample = np.asarray(time_series,dtype=float)
        best_ic_score = np.inf
        best_params = None
        n = len(time_series)

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

                d_seed = _gph_estimator(self.in_sample)
                initial_guess = [d_seed] + [0.0]*p + [0.0]*q
                bounds = [(-0.49,0.49)] + [(-0.99,0.99)]*(p+q)

                res = minimize(fun=_score_function,
                               x0=initial_guess,
                               args=(time_series, p),
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

                    final_bounds = [(-0.49, 0.49)] + [(-0.99, 0.99)] * (self.p + self.q)
                    
                    final_result = minimize(
                        fun=_score_function, 
                        x0=best_params, 
                        args=(time_series, self.p), 
                        method='L-BFGS-B', 
                        bounds=final_bounds
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
        if returns:
            return self
             
    def summary(self, returns:bool=True):
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

        if returns:
            return self
        
    def predict(self, steps:int=0):
        """
        Forecasts future values of the time series out-of-sample.
        
        Mathematically projects the series forward by assuming all future 
        random shocks ($w_t$) are exactly zero. The projected shocks are then 
        fractionally integrated using $-d$ to restore the original scale of the data.

        Parameters
        ----------
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

        y_history = _frac_diff(self.in_sample, self.d)
        w_history = _fast_residuals(y_history, self.ar_coeffs, self.ma_coeffs)

        w_future = np.zeros(steps)
        w_full = np.concatenate((w_history, w_future))

        y_full = _arma_generate(w_full, self.ar_coeffs, self.ma_coeffs)
        integrated_series = _frac_diff(y_full, -self.d)

        return integrated_series[-steps:]
    
    def predict_in_sample(self):
        """
        Generates in-sample fitted values for the historical data.
        
        Reconstructs the model's historical fit by calculating the exact historical 
        residuals via linear filtering, and subtracting those residuals 
        from the raw input data.

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
            
        
        ar_poly = [1.0] + [-phi for phi in self.ar_coeffs]
        ma_poly = [1.0] + list(self.ma_coeffs)
        
        y_history = _frac_diff(self.in_sample, self.d)
        
        w_history = np.asarray(signal.lfilter(ar_poly, ma_poly, y_history))
        
        fitted_values = self.in_sample - w_history
        
        return fitted_values

    @property
    def residuals(self)-> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        y_history = _frac_diff(self.in_sample, self.d)
        return _fast_residuals(y_history, self.ar_coeffs, self.ma_coeffs)

class FIGARCH:
    def __init__(self):
        self.p : int = 0
        self.q : int = 0
        self.d : float = 0.0
        self.omega : float = 0.0
        self.arch_coeffs = []
        self.garch_coeffs = []
        self.is_fitted = False

    def fit(self, squared_residuals: NDArray[np.float64], p_list: list[int], q_list: list[int], method: str="L-BFGS-B", 
            criterion:str='aic', returns=True):
        """
        Fits the optimal FIGARCH structure using Maximum Likelihood Estimation.
        
        Parameters
        ----------
        squared_residuals : NDArray[np.float64]
            The squared white noise shocks (\epsilon_t^2) extracted from the mean equation.
        p_list : list[int]
            List of ARCH lag orders to test.
        q_list : list[int]
            List of GARCH lag orders to test.
        method : str, default="L-BFGS-B"
            The bounded optimization algorithm to use.
        """
        self.in_sample_variance = np.asarray(squared_residuals, dtype=float)
        best_params = None
        
        omega_guess = np.var(self.in_sample_variance)

        best_ic_score = np.inf
        n = len(self.in_sample_variance)

        ic_formulas = {
            'aic': lambda n, k, neg_ll: 2 * k + 2 * neg_ll,
            'bic': lambda n, k, neg_ll: k * np.log(n) + 2 * neg_ll,
            'hqic': lambda n, k, neg_ll: 2 * k * np.log(np.log(n)) + 2 * neg_ll,
            'aicc': lambda n, k, neg_ll: (2 * k + 2 * neg_ll) + (2 * k**2 + 2 * k) / (n - k - 1) if (n - k - 1) > 0 else np.inf
        }

        criterion = criterion.lower()
        d_v_seed = _gph_estimator(self.in_sample_variance)

        for p in p_list:
            for q in q_list:
                
    
                initial_guess = [omega_guess, d_v_seed] + [0.05] * p + [0.8 / q if q > 0 else 0.0] * q
                

                bounds = [(1e-6, None), (0.0, 1.0)] + [(0.0, 0.99)] * p + [(0.0, 0.99)] * q
                
                res = minimize(
                    fun=_figarch_score_function,
                    x0=initial_guess,
                    args=(self.in_sample_variance, p, q),
                    method=method,
                    bounds=bounds
                )
                
                if res.success:
                    k = p + q + 2
                    negative_likelihood = res.fun
                    ic_score = ic_formulas[criterion](n, k, negative_likelihood)
                    
                    if ic_score < best_ic_score:
                        best_ic_score = ic_score
                        self.p = p
                        self.q = q
                        best_params = res.x

                elif not res.success:
                    print(f"Failed to fit FIGARCH({p}, d_v, {q}): {res.message}")

        if best_params is not None:
            final_bounds = [(1e-6, None), (0.0, 1.0)] + [(0.0, 0.99)] * self.p + [(0.0, 0.99)] * self.q
            
            final_result = minimize(
                fun=_figarch_score_function,
                x0=best_params,
                args=(self.in_sample_variance, self.p, self.q),
                method='L-BFGS-B',
                bounds=final_bounds
            )
            
            best_params = final_result.x
            self.omega = best_params[0]
            self.d_v = best_params[1]
            self.arch_coeffs = best_params[2 : self.p + 2] if self.p > 0 else []
            self.garch_coeffs = best_params[self.p + 2 :] if self.q > 0 else []
            self.is_fitted = True
            
            
            inv_hessian = final_result.hess_inv.todense()
            variances = np.diag(inv_hessian)
            self.std_errors = np.sqrt(np.abs(variances))
            self.t_stats = best_params / self.std_errors 
            self.p_values = 2 * stats.norm.sf(np.abs(self.t_stats))
            
            print(f"\nFinal Fit Complete: FIGARCH({self.p}, {self.d_v:.4f}, {self.q})")
            
        return self
    def summary(self, returns:bool=True):
        """
        Prints a formatted statistical regression table for the fitted FIGARCH model.
        
        Outputs the estimated parameters (baseline variance $\omega$, fractional 
        variance differencing $d_v$, ARCH, and GARCH coefficients), along with 
        their corresponding standard errors, t-statistics, and two-tailed p-values 
        derived from the inverse Hessian matrix.
        """
        if not self.is_fitted:
            print("Model is not fitted yet.")
            return
            
        print("\n" + "="*50)
        print(f"{'FIGARCH MODEL SUMMARY':^50}")
        print("="*50)
        print(f"Model Structure: FIGARCH({self.p}, d_v, {self.q})")
        print("-" * 50)
        
        print(f"{'Parameter':<10} | {'Coef':>8} | {'Std Err':>8} | {'t-stat':>8} | {'P>|t|':>8}")
        print("-" * 50)              
        
        def print_row(name, actual_index, coef):
            std_err = self.std_errors[actual_index]
            t_stat = self.t_stats[actual_index]
            p_val = self.p_values[actual_index]
            print(f"{name:<10} | {coef:>8.4f} | {std_err:>8.4f} | {t_stat:>8.4f} | {p_val:>8.4f}")

    
        print_row('omega', 0, self.omega)
        print_row('d_v', 1, self.d_v)
        
  
        for i in range(1, self.p + 1):
            print_row(f'ARCH({i})', i + 1, self.arch_coeffs[i-1])
            

        for j in range(1, self.q + 1):
            print_row(f'GARCH({j})', self.p + 1 + j, self.garch_coeffs[j-1])
            
        print("="*50 + "\n")

        if returns:
            return self
        
    def predict_in_sample(self):
        """
        Generates in-sample conditional variances for the historical data.
        
        Reconstructs the model's historical volatility fit by calculating the 
        ARCH(\infty) decay weights and convoluting them with the actual 
        historical squared shocks.

        Returns
        -------
        np.ndarray
            A 1D array of conditional variances ($\sigma_t^2$), equal in length 
            to the input time series.
            
        Raises
        ------
        ValueError
            If the method is called before the model has been successfully fitted.
        """
        if not self.is_fitted:
            raise ValueError("FIGARCH not fitted yet")
            
        n = len(self.in_sample_variance)
        
        
        weights = _figarch_weights(self.d_v, self.arch_coeffs, self.garch_coeffs, n)
        
       
        beta_sum = np.sum(self.garch_coeffs) if self.q > 0 else 0.0
        omega_star = self.omega / (1.0 - beta_sum)
        
        
        conditional_variance = _dynamic_variance(self.in_sample_variance, omega_star, weights)
        
        return conditional_variance

    def predict(self, steps: int = 0):
        """
        Forecasts future conditional variance out-of-sample.
        
        Mathematically projects volatility forward iteratively. Because future 
        squared shocks ($\epsilon_{t+k}^2$) are unobservable, the model substitutes 
        them with their expected value, which is the forecasted conditional 
        variance ($\sigma_{t+k}^2$).

        Parameters
        ----------
        steps : int, default=0
            The number of future time periods to forecast.

        Returns
        -------
        np.ndarray
            A 1D array of length `steps` containing the forecasted conditional variances.
            
        Raises
        ------
        ValueError
            If the method is called before the model has been successfully fitted.
        """
        if not self.is_fitted:
            raise ValueError("FIGARCH not fitted yet")
            
        if steps <= 0:
            return np.array([])

        n = len(self.in_sample_variance)
        
        beta_sum = np.sum(self.garch_coeffs) if self.q > 0 else 0.0
        omega_star = self.omega / (1.0 - beta_sum)
        
        
        extended_shocks = np.copy(self.in_sample_variance)
        forecasted_variance = np.zeros(steps)
        
        
        for k in range(steps):
            current_len = n + k
            
            
            weights = _figarch_weights(self.d_v, self.arch_coeffs, self.garch_coeffs, current_len)
            current_variance_path = _dynamic_variance(extended_shocks, omega_star, weights)
            next_variance = current_variance_path[-1]
            forecasted_variance[k] = next_variance
            
            extended_shocks = np.append(extended_shocks, next_variance)
            
        return forecasted_variance
    
def _score_function(params:list[float], time_series:NDArray[np.float64], p:int):
    """
    Calculates negative log-likelihood for a given set of ARFIMA parameters.

    Parameters
    ----------
    params : list[float]
        The optimization parameters containing $d$, followed by AR and MA coefficients.
    time_series : NDArray[np.float64]
        The 1D input time series data.
    p : int
        The autoregressive (AR) lag order.
    Returns
    -------
    float
        The negative log-likelihood value, or scalar (1e10) for invalid variances.
    """
    n = len(time_series)
    
    d_guess = float(params[0])
    ar_coeffs = params[1 : p + 1]
    ma_coeffs = params[p + 1 :]
    
    y_differenced = _frac_diff(time_series, d_guess)
    
    residuals = _fast_residuals(y_differenced, ar_coeffs, ma_coeffs)
    
    variance = np.sum(residuals ** 2) / n
    

    if variance <= 1e-10 or np.isnan(variance):
        return 1e10  
        
    neg_ll = (n / 2) * np.log(2 * np.pi) + (n / 2) * np.log(variance) + (n / 2)
    
    if np.isnan(neg_ll) or np.isinf(neg_ll):
        return 1e10
        
    return neg_ll

def _fast_residuals(diff_data:NDArray[np.float64], ar_coeffs:list[float], ma_coeffs:list[float]):
    """
    Returns the residuals w_t from differenced data using linear filtering.
    
    Solves the ARMA equation (1 - phi)y_t = (1 + theta)w_t for w_t.

    Parameters
    ----------
    diff_data : NDArray[np.float64]
        The fractionally differenced time series ($y_t$).
    ar_coeffs : list[float]
        The autoregressive coefficients ($\phi$).
    ma_coeffs : list[float]
        The moving average coefficients ($\theta$).

    Returns
    -------
    np.ndarray
        A 1D array of residuals.
    """
    
    b = [1.0] + [-i for i in ar_coeffs]
    a = [1.0] + list(ma_coeffs)

    residuals = signal.lfilter(b, a, diff_data)

    return np.asarray(residuals)

def _arma_generate(shocks: NDArray[np.float64], ar_coeffs: list[float], ma_coeffs: list[float]):
    """
    Generates time series data y_t from given residuals w_t.
    
    Acts as the inverse of `_fast_residuals` by solving $(1 - phi)y_t = (1 + theta L)w_t for y_t.

    Parameters
    ----------
    shocks : NDArray[np.float64]
        The array of residuals.
    ar_coeffs : list[float]
        The autoregressive coefficients ($\phi$).
    ma_coeffs : list[float]
        The moving average coefficients ($\theta$).

    Returns
    -------
    np.ndarray
        A 1D array of generated ARMA values.
    """
    a = [1.0] + [-i for i in ar_coeffs]  
    b = [1.0] + list(ma_coeffs)          
    return np.asarray(signal.lfilter(b, a, shocks))

def _frac_diff(time_series:NDArray[np.float64], d:float):
    """
    Applies the fractional differencing operator $(1-L)^d$ using Fast Fourier Transform (FFT).

    Parameters
    ----------
    time_series : NDArray[np.float64]
        The 1D time series array to difference.
    d : float
        The fractional integration parameter. Pass $-d$ for inverse differencing (integration).

    Returns
    -------
    np.ndarray
        A 1D array of the fractionally differenced time series.
    """
    w = [1.0]
    n = len(time_series)
    for i in range(1,n):
        w_i = ((i-1-d) / i)*w[-1]
        w.append(w_i)

    conv_result = signal.fftconvolve(time_series, w, mode="full")

    return conv_result[:n]

def _gph_estimator(time_series:NDArray[np.float64] | np.ndarray, bandwidth_power:float = 0.5):
    """
    Approximates d(fractional differencing parameter) using the Geweke and Porter Hudak(GPH) semiparametric
    estimator.

    Parameters
    ----------
    time_series: NDArray[np.float64] | np.ndarray
        1D time series array
    bandwidth_power: float (default = 0.5)
        determines the number of periodogram frequencies included in the regression. Ideally between 0.5 and 0.7
    
    Returns
    -------
    float
        returns the estimated fractional difference parameter d.
        d is clipped between -0.49 and 0.49.
    """
    n = len(time_series)

    freqs, pgram = signal.periodogram(time_series)

    m = int(np.floor(n ** bandwidth_power))
    if m < 2: 
        return 0.1

    lam = freqs[1 : m+1]
    i_lam = pgram[1 : m+1]

    x = 2 * np.log(2 * np.sin(lam / 2))
    y = np.log(i_lam)

    slope, _ = np.polyfit(x, y, 1)
    d_estimate = -slope

    return float(np.clip(d_estimate, -0.49, 0.49))


def _figarch_weights(d_v: float, arch_coeffs: list[float], garch_coeffs: list[float], n: int):
    """
    Computes the ARCH(infinity) representation weights for the FIGARCH process.
    
    Expands the fractional differencing polynomial (1-L)^d_v up to lag n 
    and filters it through the ARCH and GARCH polynomials to derive the 
    decay weights lambda_i for the conditional variance equation:
    lambda(L) = 1 - [phi(L) / beta(L)] * (1-L)^d_v.

    Parameters
    ----------
    d_v : float
        The fractional integration parameter for volatility.
    arch_coeffs : list[float]
        The autoregressive conditional heteroskedasticity (ARCH) coefficients.
    garch_coeffs : list[float]
        The generalized autoregressive conditional heteroskedasticity (GARCH) coefficients.
    n : int
        The length of the time series (the truncation point for the infinite expansion).

    Returns
    -------
    np.ndarray
        A 1D array of length n containing the ARCH(infinity) decay weights.
    """
    frac_weights = np.zeros(n)
    frac_weights[0] = 1.0
    for i in range(1, n):
        frac_weights[i] = ((i - 1 - d_v) / i) * frac_weights[i - 1]

    phi = [1.0] + [-phi_i for phi_i in arch_coeffs]
    beta = [1.0] + [-beta_i for beta_i in garch_coeffs]

    core_expansion = signal.lfilter(phi, beta, frac_weights)
    
    lambda_weights = -core_expansion
    lambda_weights[0] = 0.0  
    
    return lambda_weights


def _dynamic_variance(squared_shocks: NDArray[np.float64], omega: float, weights: NDArray[np.float64]):
    """
    Calculates the conditional variance array (sigma_t^2) using FFT convolution.
    
    Applies the pre-calculated ARCH(infinity) decay weights to the historical 
    squared shocks via Fast Fourier Transform convolution to efficiently construct 
    the dynamic variance path. Applies a strict mathematical floor to prevent 
    variance from dropping below zero.

    Parameters
    ----------
    squared_shocks : NDArray[np.float64]
        The array of squared residuals (epsilon_t^2) extracted from the mean equation.
    omega : float
        The adjusted baseline variance constant (omega^*).
    weights : NDArray[np.float64]
        The array of ARCH(infinity) decay weights.

    Returns
    -------
    np.ndarray
        A 1D array of conditional variances (sigma_t^2) equal in length to the input shocks.
    """
    n = len(squared_shocks)
    
    arch_effect = signal.fftconvolve(squared_shocks, weights, mode="full")[:n]
    cond_variance = omega + arch_effect
    
    return np.maximum(cond_variance, 1e-6)


def _figarch_score_function(params: list[float], squared_shocks: NDArray[np.float64], p: int, q: int):
    """
    Evaluates the Gaussian negative log-likelihood for dynamic volatility.
    
    Extracts the FIGARCH parameters, constructs the conditional variance path 
    (sigma_t^2), and calculates the likelihood score used by the optimizer. 
    Implements strict penalty boundaries to reject non-stationary GARCH 
    parameter combinations or invalid variance arrays.
    
    Solves: -ln L = 0.5 * sum[ ln(sigma_t^2) + (epsilon_t^2 / sigma_t^2) ]

    Parameters
    ----------
    params : list[float]
        The optimization parameters containing omega, d_v, followed by ARCH and GARCH coefficients.
    squared_shocks : NDArray[np.float64]
        The 1D array of squared residuals (epsilon_t^2).
    p : int
        The ARCH lag order.
    q : int
        The GARCH lag order.

    Returns
    -------
    float
        The negative log-likelihood scalar, or a penalty value (1e10) if bounds are violated.
    """
    omega = float(params[0])
    d_v = float(params[1])
    arch_coeffs = params[2 : p + 2]
    garch_coeffs = params[p + 2 :]
    
    n = len(squared_shocks)
    
    weights = _figarch_weights(d_v, arch_coeffs, garch_coeffs, n)
    
    beta_sum = np.sum(garch_coeffs)
    if beta_sum >= 1.0:
        return 1e10 
        
    omega_star = omega / (1.0 - beta_sum)
    
    cond_variance = _dynamic_variance(squared_shocks, omega_star, weights)
    
    log_likelihood = 0.5 * np.sum(np.log(cond_variance) + (squared_shocks / cond_variance))
    
    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
        return 1e10
        
    return log_likelihood

