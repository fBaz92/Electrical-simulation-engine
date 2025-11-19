from __future__ import annotations

import numpy as np
from scipy.constants import k, elementary_charge as q
from scipy.optimize import least_squares, differential_evolution


class PVModelSingleDiode:
    """
    Single-diode model for photovoltaic (PV) module characterization.
    
    This class implements the single-diode equivalent circuit model for PV modules,
    which is widely used to predict the electrical behavior of solar panels under
    various operating conditions. The model represents a PV cell/module using five
    parameters: photocurrent (Iph), diode saturation current (Is), ideality factor
    (n), series resistance (Rs), and shunt resistance (Rsh).
    
    The class provides functionality to:
    - Store datasheet parameters (Isc, Voc, Imp, Vmp, etc.)
    - Solve for the five unknown model parameters using optimization
    - Generate I-V and P-V curves for any irradiance and temperature conditions
    - Account for temperature dependencies using temperature coefficients
    - Support NOCT (Nominal Operating Cell Temperature) modeling
    
    The single-diode equation is:
        I = Iph - Is * (exp((V + I*Rs)/(n*Vt)) - 1) - (V + I*Rs)/Rsh
    
    where Vt is the thermal voltage (k*T/q).
    
    Attributes:
        Isc: Short-circuit current at STC (A).
        Voc: Open-circuit voltage at STC (V).
        Imp: Current at maximum power point at STC (A).
        Vmp: Voltage at maximum power point at STC (V).
        Ns: Number of cells in series.
        Tcell: Cell temperature at STC (°C).
        Pmp: Maximum power at STC (W).
        Iph: Photocurrent (A) - solved parameter.
        Is: Diode saturation current (A) - solved parameter.
        n: Diode ideality factor - solved parameter.
        Rs: Series resistance (Ω) - solved parameter.
        Rsh: Shunt resistance (Ω) - solved parameter.
    
    Example:
        >>> # Create model from datasheet parameters
        >>> model = PVModelSingleDiode(
        ...     Isc=14.18, Voc=38.8, Imp=13.50, Vmp=32.6, Ns=108, Tcell=25.0
        ... )
        >>> # Solve for model parameters
        >>> params = model.solve()
        >>> # Generate I-V curve at STC
        >>> V = np.linspace(0, 38.8, 100)
        >>> I = model.I_of_V(V)
    """

    def __init__(
        self,
        Isc: float,
        Voc: float,
        Imp: float,
        Vmp: float,
        Ns: int,
        Tcell: float = 25.0,
        Pmp: float | None = None,
        temp_coeff_isc: float | None = None,
        temp_coeff_voc: float | None = None,
        temp_coeff_pmax: float | None = None,
        noct_temp: float | None = None,
        noct_irradiance: float = 800.0,
        noct_ambient: float = 20.0,
        noct_Isc: float | None = None,
        noct_Voc: float | None = None,
        noct_Vmp: float | None = None,
        noct_Imp: float | None = None,
        noct_Pmp: float | None = None,
        use_noct_model: bool = False,
    ):
        """
        Initialize PV module model with datasheet parameters.
        
        The initialization process stores the datasheet parameters and optionally
        infers temperature coefficients from NOCT (Nominal Operating Cell Temperature)
        data if provided. The model parameters (Iph, Is, n, Rs, Rsh) are not solved
        until the `solve()` method is called.
        
        Args:
            Isc: Short-circuit current at STC (A).
            Voc: Open-circuit voltage at STC (V).
            Imp: Current at maximum power point at STC (A).
            Vmp: Voltage at maximum power point at STC (V).
            Ns: Number of cells in series in the module.
            Tcell: Cell temperature at STC (°C). Defaults to 25.0.
            Pmp: Maximum power at STC (W). If None, calculated as Vmp * Imp.
            temp_coeff_isc: Temperature coefficient for Isc (%/°C). If absolute value
                > 1.0, assumed to be in %/°C and normalized. If None, may be inferred
                from NOCT data.
            temp_coeff_voc: Temperature coefficient for Voc (%/°C). Normalized if > 1.0.
                If None, may be inferred from NOCT data.
            temp_coeff_pmax: Temperature coefficient for maximum power (%/°C). Normalized
                if > 1.0. If None, may be inferred from NOCT data.
            noct_temp: Cell temperature at NOCT conditions (°C). Used to infer temperature
                coefficients if provided along with NOCT electrical data.
            noct_irradiance: Irradiance at NOCT reference conditions (W/m²). Defaults to 800.0.
            noct_ambient: Ambient temperature at NOCT conditions (°C). Defaults to 20.0.
            noct_Isc: Short-circuit current at NOCT (A). Used to infer temp_coeff_isc.
            noct_Voc: Open-circuit voltage at NOCT (V). Used to infer temp_coeff_voc.
            noct_Vmp: Voltage at MPP at NOCT (V). Used to infer temp_coeff_pmax.
            noct_Imp: Current at MPP at NOCT (A). Used to infer temp_coeff_pmax.
            noct_Pmp: Maximum power at NOCT (W). Used to infer temp_coeff_pmax.
            use_noct_model: If True, temperatures passed to I_of_V() are interpreted as
                ambient temperatures and converted to cell temperatures using NOCT model.
                Defaults to False.
        
        Note:
            STC (Standard Test Conditions): 1000 W/m² irradiance, 25°C cell temperature.
            NOCT (Nominal Operating Cell Temperature): Typically 800 W/m² irradiance,
            20°C ambient temperature, wind speed 1 m/s.
        """
        self.Isc = Isc
        self.Voc = Voc
        self.Imp = Imp
        self.Vmp = Vmp
        self.Ns = Ns
        self.Tcell = Tcell
        self.Pmp = Pmp if Pmp is not None else Vmp * Imp
        self.G_ref = 1000.0  # W/m^2, riferimento tipico datasheet
        self.alpha_I_default = 0.0005  # fallback coefficiente temperatura fotocorrente
        self.Eg_eV = 1.12  # gap energetico silicio (eV)
        self.temp_coeff_isc = self._normalize_temp_coeff(temp_coeff_isc)
        self.temp_coeff_voc = self._normalize_temp_coeff(temp_coeff_voc)
        self.temp_coeff_pmax = self._normalize_temp_coeff(temp_coeff_pmax)
        self.noct_temp = noct_temp
        self.noct_irradiance = noct_irradiance
        self.noct_ambient = noct_ambient
        self.noct_Isc = noct_Isc
        self.noct_Voc = noct_Voc
        self.noct_Vmp = noct_Vmp
        self.noct_Imp = noct_Imp
        self.noct_Pmp = noct_Pmp
        self.use_noct_model = use_noct_model

        # Costanti termiche
        self.Vt = (k * (Tcell + 273.15)) / q  # tensione termica per singola cella
        self.Vt_mod = self.Vt * Ns           # tensione termica moltiplicata per Ns

        # Parametri da determinare
        self.Iph = None
        self.Is = None
        self.n = None
        self.Rs = None
        self.Rsh = None
        self.Rs_prior = 0.2
        self.Rs_prior_weight = 0.2
        self.Rsh_prior = 2000.0
        self.Rsh_prior_weight = 2000.0

        self._infer_coefficients_from_noct()
        self._base_v_norm = None
        self._base_i_norm = None
        self._base_power_norm = None

    @staticmethod
    def _normalize_temp_coeff(coeff: float | None) -> float | None:
        """
        Normalize temperature coefficient to fractional form.
        
        Temperature coefficients can be provided either as fractional values (e.g., 0.0005)
        or as percentage values (e.g., 0.05). This method detects percentage values
        (absolute value > 1.0) and converts them to fractional form.
        
        Args:
            coeff: Temperature coefficient, either fractional or percentage (%/°C).
                If None, returns None.
        
        Returns:
            Normalized temperature coefficient as fractional value, or None if input is None.
        
        Example:
            >>> PVModelSingleDiode._normalize_temp_coeff(0.05)  # 5%/°C -> 0.0005
            0.0005
            >>> PVModelSingleDiode._normalize_temp_coeff(0.0005)  # Already fractional
            0.0005
        """
        if coeff is None:
            return None
        if abs(coeff) > 1.0:
            return coeff / 100.0
        return coeff

    def _infer_coefficients_from_noct(self) -> None:
        """
        Infer temperature coefficients from NOCT data if available.
        
        If NOCT (Nominal Operating Cell Temperature) electrical parameters are provided
        and the corresponding temperature coefficients are missing, this method calculates
        the coefficients by comparing STC and NOCT values.
        
        The temperature coefficients are calculated as:
            temp_coeff = (value_noct - value_stc) / (value_stc * delta_T)
        
        where delta_T = noct_temp - Tcell.
        
        This method is called automatically during initialization if NOCT data is provided.
        It only infers coefficients that are not already specified.
        
        Note:
            Requires noct_temp to be set. If delta_T is too small (< 1e-6), inference
            is skipped to avoid numerical issues.
        """
        if self.noct_temp is None:
            return

        delta_T = self.noct_temp - self.Tcell
        if abs(delta_T) < 1e-6:
            return

        if self.temp_coeff_isc is None and self.noct_Isc is not None and self.Isc > 0:
            self.temp_coeff_isc = (self.noct_Isc - self.Isc) / (self.Isc * delta_T)

        if self.temp_coeff_voc is None and self.noct_Voc is not None and self.Voc > 0:
            self.temp_coeff_voc = (self.noct_Voc - self.Voc) / (self.Voc * delta_T)

        Pmp_stc = self.Pmp
        Pmp_noct = self.noct_Pmp
        if Pmp_noct is None and self.noct_Vmp is not None and self.noct_Imp is not None:
            Pmp_noct = self.noct_Vmp * self.noct_Imp

        if (
            self.temp_coeff_pmax is None
            and Pmp_noct is not None
            and Pmp_stc not in (None, 0)
        ):
            self.temp_coeff_pmax = (Pmp_noct - Pmp_stc) / (Pmp_stc * delta_T)

    def _cell_temperature(self, irradiance, temperature):
        if temperature is None:
            temperature = self.Tcell

        if not self.use_noct_model or self.noct_temp is None:
            return temperature

        irr = max(irradiance, 0.0)
        if self.noct_irradiance <= 0:
            return temperature

        delta_noct = self.noct_temp - self.noct_ambient
        correction = (irr / self.noct_irradiance) * delta_noct
        return temperature + correction

    def single_diode_equation(
        self,
        V: float,
        I: float,
        Iph: float,
        Is: float,
        n: float,
        Rs: float,
        Rsh: float,
        vt_mod: float | None = None,
    ) -> float:
        """
        Evaluate the single-diode equation residual.
        
        The single-diode equation is an implicit equation relating current and voltage:
            I = Iph - Is * (exp((V + I*Rs)/(n*Vt)) - 1) - (V + I*Rs)/Rsh
        
        This method returns the residual (error) of the equation, which should be zero
        when the equation is satisfied. It is used both for parameter fitting and for
        solving I(V) at given operating conditions.
        
        Args:
            V: Terminal voltage (V).
            I: Terminal current (A).
            Iph: Photocurrent (A).
            Is: Diode saturation current (A).
            n: Diode ideality factor.
            Rs: Series resistance (Ω).
            Rsh: Shunt resistance (Ω).
            vt_mod: Thermal voltage for the module (V). If None, uses self.Vt_mod.
        
        Returns:
            Residual value (A). Should be zero when the equation is satisfied.
        
        Note:
            The equation is implicit in I, meaning I appears on both sides. To solve
            for I given V, a numerical root-finding method is required.
        """
        vt = vt_mod if vt_mod is not None else self.Vt_mod
        return Iph - Is * (np.exp((V + I*Rs) / (n * vt)) - 1) - (V + I*Rs)/Rsh - I

    def residuals(self, params: np.ndarray) -> list[float]:
        """
        Compute residuals for parameter optimization.
        
        This method defines the objective function for fitting the five model parameters
        (Iph, Is, n, Rs, Rsh) to match the datasheet characteristics. The residuals
        enforce:
        1. Short-circuit condition: I = Isc when V = 0
        2. Open-circuit condition: I = 0 when V = Voc
        3. Maximum power point: I = Imp when V = Vmp
        4. Maximum power condition: dP/dV = 0 at MPP (power derivative constraint)
        5-6. Regularization terms for Rs and Rsh to avoid degenerate solutions
        
        Args:
            params: Array of model parameters [Iph, Is, n, Rs, Rsh].
        
        Returns:
            List of 6 residual values. All should be close to zero at the solution.
        
        Note:
            The regularization terms (r5, r6) help stabilize the optimization by
            penalizing deviations from prior estimates of Rs and Rsh. This prevents
            the solver from finding unrealistic parameter values.
        """
        Iph, Is, n, Rs, Rsh = params

        # Equazione a corto circuito: V=0, I=Isc
        r1 = self.single_diode_equation(0.0, self.Isc, Iph, Is, n, Rs, Rsh)

        # Equazione a vuoto: V=Voc, I=0
        r2 = self.single_diode_equation(self.Voc, 0.0, Iph, Is, n, Rs, Rsh)

        # Equazione al punto MPP: V=Vmp, I=Imp
        r3 = self.single_diode_equation(self.Vmp, self.Imp, Iph, Is, n, Rs, Rsh)

        # Condizione di massimo di potenza: dP/dV = 0 al MPP
        dIdV_mpp = self.current_derivative(self.Vmp, self.Imp, Iph, Is, n, Rs, Rsh)
        r4 = self.Imp + self.Vmp * dIdV_mpp

        # Regolarizzazione su Rs e Rsh per evitare soluzioni degeneri
        r5 = (Rs - self.Rs_prior) / self.Rs_prior_weight
        r6 = (Rsh - self.Rsh_prior) / self.Rsh_prior_weight

        return [r1, r2, r3, r4, r5, r6]

    def current_derivative(
        self,
        V: float,
        I: float,
        Iph: float,
        Is: float,
        n: float,
        Rs: float,
        Rsh: float,
        vt_mod: float | None = None,
    ) -> float:
        """
        Compute derivative dI/dV from the single-diode equation.
        
        This method calculates the derivative of current with respect to voltage
        using implicit differentiation of the single-diode equation. It is used
        to enforce the maximum power point condition (dP/dV = 0) during parameter
        fitting.
        
        The derivative is computed as:
            dI/dV = -df/dV / df/dI
        
        where f(V, I) = 0 is the single-diode equation.
        
        Args:
            V: Terminal voltage (V).
            I: Terminal current (A).
            Iph: Photocurrent (A).
            Is: Diode saturation current (A).
            n: Diode ideality factor.
            Rs: Series resistance (Ω).
            Rsh: Shunt resistance (Ω).
            vt_mod: Thermal voltage for the module (V). If None, uses self.Vt_mod.
        
        Returns:
            Derivative dI/dV (A/V).
        
        Note:
            The exponential argument is clipped to [-100, 100] to prevent numerical
            overflow. A minimum threshold is applied to df_dI to avoid division by zero.
        """
        vt = vt_mod if vt_mod is not None else self.Vt_mod
        arg = (V + I * Rs) / (n * vt)
        arg = np.clip(arg, -100, 100)  # evita overflow numerico
        exp_term = np.exp(arg)

        df_dV = -Is * exp_term * (1.0 / (n * vt)) - 1.0 / Rsh
        df_dI = -Is * exp_term * (Rs / (n * vt)) - Rs / Rsh - 1.0

        if np.abs(df_dI) < 1e-12:
            df_dI = np.copysign(1e-12, df_dI if df_dI != 0 else 1.0)

        return -df_dV / df_dI

    def solve(self) -> np.ndarray:
        """
        Solve for the five model parameters using optimization.
        
        This method determines the five unknown parameters (Iph, Is, n, Rs, Rsh) by
        minimizing the residuals that enforce the datasheet characteristics. The
        optimization uses a two-stage approach:
        1. Local optimization with Levenberg-Marquardt algorithm (least_squares)
        2. If local optimization fails, global optimization with differential evolution
        
        The method uses reasonable initial guesses and explicit bounds to ensure
        numerical stability and physical feasibility of the solution.
        
        Returns:
            Array of solved parameters [Iph, Is, n, Rs, Rsh].
        
        Raises:
            RuntimeError: If optimization fails to converge after both local and
                global optimization attempts.
        
        Note:
            The solved parameters are stored in instance attributes (self.Iph, self.Is,
            etc.) and can be accessed directly after calling this method.
        
        Example:
            >>> model = PVModelSingleDiode(Isc=14.18, Voc=38.8, Imp=13.50, Vmp=32.6, Ns=108)
            >>> params = model.solve()
            >>> print(f"Iph = {model.Iph:.6f} A")
        """
        # Valori iniziali molto ragionevoli
        Iph0 = self.Isc * 1.02         # fotocorrente leggermente > Isc
        Is0 = 1e-10                    # tipico valore iniziale
        n0 = 1.3                       # idealità per celle Si
        Rs0 = 0.2                      # ordine di grandezza comune
        Rsh0 = 1000                    # resistenza di shunt inizialmente alta

        # Bounds espliciti per garantire fattibilità numerica
        lower_bounds = np.array([0.0, 0.0, 1.0, 0.01, 10.0], dtype=float)
        upper_Iph = max(20.0, 2.0 * self.Isc)
        upper_bounds = np.array([upper_Iph, 1e-3, 2.0, 10.0, 1e5], dtype=float)
        bounds = (lower_bounds, upper_bounds)

        x0 = np.array([Iph0, Is0, n0, Rs0, Rsh0], dtype=float)
        eps = 1e-12
        x0 = np.maximum(x0, lower_bounds + eps)
        finite_mask = np.isfinite(upper_bounds)
        x0[finite_mask] = np.minimum(x0[finite_mask], upper_bounds[finite_mask] - eps)

        def run_least_squares(initial_guess):
            return least_squares(
                self.residuals,
                initial_guess,
                bounds=bounds,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12
            )

        def global_initial_guess():
            """
            Use global optimization to find a plausible initial guess.
            
            This function is invoked only if the local solver fails to converge.
            It uses differential evolution to find a reasonable starting point
            for the local optimizer.
            """
            def objective(params):
                res = self.residuals(params)
                return np.sum(np.square(res))

            de_bounds = list(zip(lower_bounds, upper_bounds))
            result = differential_evolution(
                objective,
                de_bounds,
                maxiter=200,
                tol=1e-6,
                polish=False,
                seed=42
            )
            return result.x

        try:
            result = run_least_squares(x0)
        except ValueError:
            result = None

        if result is None or not result.success:
            x0_global = global_initial_guess()
            x0_global = np.maximum(x0_global, lower_bounds + eps)
            x0_global[finite_mask] = np.minimum(x0_global[finite_mask], upper_bounds[finite_mask] - eps)
            result = run_least_squares(x0_global)

        if not result.success:
            raise RuntimeError("Ottimizzazione non convergente")

        self.Iph, self.Is, self.n, self.Rs, self.Rsh = result.x
        return result.x

    def _operating_point_params(
        self, irradiance: float, temperature: float | None
    ) -> tuple[float, float, float]:
        """
        Calculate model parameters adjusted for operating conditions.
        
        This method adjusts the model parameters (Iph, Is, Vt) based on the
        operating irradiance and temperature. The adjustments account for:
        - Linear scaling of photocurrent with irradiance
        - Temperature dependence of photocurrent (via temp_coeff_isc)
        - Temperature dependence of saturation current (exponential with bandgap)
        - Temperature dependence of thermal voltage
        - Optional NOCT model for ambient-to-cell temperature conversion
        
        Args:
            irradiance: Solar irradiance (W/m²). Clamped to non-negative values.
            temperature: Cell or ambient temperature (°C). If None, uses self.Tcell.
                If use_noct_model is True, interpreted as ambient temperature.
        
        Returns:
            Tuple (Iph_adj, Is_adj, vt_mod) containing:
            - Iph_adj: Adjusted photocurrent (A) for given conditions
            - Is_adj: Adjusted saturation current (A) for given temperature
            - vt_mod: Adjusted thermal voltage (V) for module at given temperature
        
        Note:
            The photocurrent scales linearly with irradiance ratio (G/G_ref) and
            includes temperature corrections. The saturation current follows the
            temperature dependence of the bandgap energy. The thermal voltage
            includes temperature-dependent scaling factors.
        """
        cell_temperature = self._cell_temperature(irradiance, temperature)
        irradiance = max(irradiance, 0.0)

        T_ref_k = self.Tcell + 273.15
        T_k = cell_temperature + 273.15

        delta_T = cell_temperature - self.Tcell
        G_ratio = irradiance / self.G_ref if self.G_ref > 0 else 0.0

        alpha_I = self.temp_coeff_isc if self.temp_coeff_isc is not None else self.alpha_I_default
        Iph_temp = self.Iph * (1 + alpha_I * delta_T)

        # coefficiente potenza massima: ridimensioniamo leggermente la fotocorrente
        power_scale = 1.0
        if self.temp_coeff_pmax is not None:
            power_scale = max(0.1, 1.0 + self.temp_coeff_pmax * delta_T)

        Iph_adj = max(0.0, Iph_temp * G_ratio * power_scale)

        Eg = self.Eg_eV * q
        Is_adj = self.Is * (T_k / T_ref_k)**3 * np.exp((Eg / (k)) * (1 / T_ref_k - 1 / T_k))

        voc_scale = 1.0
        if self.temp_coeff_voc is not None:
            voc_scale = max(0.1, 1.0 + self.temp_coeff_voc * delta_T)

        vt_mod = (k * T_k) / q * self.Ns * voc_scale

        return Iph_adj, Is_adj, vt_mod

    def I_of_V(
        self,
        V: float | np.ndarray,
        irradiance: float | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        """
        Calculate current I(V) for given voltage(s) and operating conditions.
        
        This method solves the implicit single-diode equation to find the current
        corresponding to given voltage(s). Since the equation is implicit in I,
        a numerical root-finding method (least_squares) is used for each voltage point.
        
        The method adjusts model parameters for the specified irradiance and temperature
        before solving, allowing prediction of module behavior under any operating
        conditions.
        
        Args:
            V: Voltage(s) at which to calculate current (V). Can be a single value
                or an array of voltages.
            irradiance: Solar irradiance (W/m²). If None, uses reference irradiance
                (1000 W/m²).
            temperature: Cell or ambient temperature (°C). If None, uses self.Tcell.
                If use_noct_model is True, interpreted as ambient temperature.
        
        Returns:
            Array of current values (A) corresponding to input voltages. Shape matches
            input V. Values are clamped to non-negative (no reverse current).
        
        Example:
            >>> # Generate I-V curve at STC
            >>> V = np.linspace(0, 38.8, 100)
            >>> I = model.I_of_V(V)
            >>> 
            >>> # Calculate current at specific voltage and conditions
            >>> I_single = model.I_of_V(30.0, irradiance=800.0, temperature=45.0)
        """
        if irradiance is None:
            irradiance = self.G_ref
        if temperature is None:
            temperature = self.Tcell

        Iph_adj, Is_adj, vt_mod = self._operating_point_params(irradiance, temperature)

        Ivals = []
        for v in np.atleast_1d(V):
            # guess iniziale: corrente ≈ Isc scalata
            f = lambda i: self.single_diode_equation(v, i, Iph_adj, Is_adj, self.n, self.Rs, self.Rsh, vt_mod=vt_mod)
            sol = least_squares(f, 0.9 * self.Isc)
            Ivals.append(sol.x[0])
        return np.maximum(np.array(Ivals), 0.0)

    def _ensure_base_curve(self, num_points=400):
        if self._base_v_norm is not None and self._base_i_norm is not None:
            return

        V = np.linspace(0, self.Voc, num_points)
        I = self.I_of_V(V, irradiance=self.G_ref, temperature=self.Tcell)
        P = V * I
        self._base_v_norm = np.divide(V, self.Voc, out=np.zeros_like(V), where=self.Voc != 0)
        self._base_i_norm = np.divide(I, self.Isc, out=np.zeros_like(I), where=self.Isc != 0)
        p_max = np.max(P) if np.size(P) else 1.0
        self._base_power_norm = P / p_max if p_max > 0 else P

    def datasheet_curve(self, irradiance, temperature, num_points=400):
        """
        Restituisce una curva I-V costruita usando i coefficienti di temperatura/
        irraggiamento del datasheet. La curva STC viene normalizzata una sola volta
        e poi scalata per rispettare Isc, Voc e Pmax nelle nuove condizioni.
        """
        self._ensure_base_curve(num_points)

        cell_temperature = self._cell_temperature(irradiance, temperature)
        delta_T = cell_temperature - self.Tcell
        G_ratio = max(irradiance, 0.0) / self.G_ref if self.G_ref > 0 else 0.0

        alpha_I = self.temp_coeff_isc if self.temp_coeff_isc is not None else self.alpha_I_default
        alpha_V = self.temp_coeff_voc if self.temp_coeff_voc is not None else 0.0
        alpha_P = self.temp_coeff_pmax if self.temp_coeff_pmax is not None else 0.0

        Isc_adj = self.Isc * max(0.0, 1.0 + alpha_I * delta_T) * G_ratio
        Voc_adj = self.Voc * max(0.1, 1.0 + alpha_V * delta_T)
        Pmp_base = self.Pmp if self.Pmp is not None else self.Vmp * self.Imp
        Pmp_adj = Pmp_base * max(0.0, G_ratio) * max(0.1, 1.0 + alpha_P * delta_T)

        V_scaled = self._base_v_norm * Voc_adj
        I_scaled = self._base_i_norm * Isc_adj
        P_scaled = V_scaled * I_scaled
        p_scaled_max = np.max(P_scaled) if np.size(P_scaled) else 0.0
        if p_scaled_max > 0 and Pmp_adj > 0:
            scale_factor = Pmp_adj / p_scaled_max
            I_scaled *= scale_factor

        return V_scaled, np.maximum(I_scaled, 0.0)


# … (includi la classe PVModelSingleDiode definita prima) …

if __name__ == "__main__":
    # Parametri dal datasheet Zephir 440W
    Isc  = 14.18
    Voc  = 38.8
    Imp  = 13.50
    Vmp  = 32.6
    Ns   = 108
    Tcell = 25.0

    Tcell_noct = 43.0
    model = PVModelSingleDiode(
        Isc=Isc,
        Voc=Voc,
        Imp=Imp,
        Vmp=Vmp,
        Ns=Ns,
        Tcell=Tcell,
        Pmp=Vmp * Imp,
        temp_coeff_isc=0.046,
        temp_coeff_voc=-0.25,
        temp_coeff_pmax=-0.3,
        noct_temp=Tcell_noct,
        noct_irradiance=800.0,
        noct_ambient=20.0,
        noct_Isc=11.49,
        noct_Voc=36.9,
        noct_Vmp=30.7,
        noct_Imp=10.81,
        noct_Pmp=331.0,
        use_noct_model=False,
    )
    params = model.solve()
    print("Parametri trovati:")
    print(f"Iph = {model.Iph}")
    print(f"Is  = {model.Is}")
    print(f"n   = {model.n}")
    print(f"Rs  = {model.Rs}")
    print(f"Rsh = {model.Rsh}")

    # Generazione curva I-V e P-V
    import matplotlib.pyplot as plt

    V = np.linspace(0, Voc, 300)
    I = model.I_of_V(V)
    P = V * I

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(V, I)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Curva I-V")

    plt.subplot(1,2,2)
    plt.plot(V, P)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Power (W)")
    plt.title("Curva P-V")

    plt.tight_layout()
    plt.show()

    # Analisi al variare dell'irraggiamento
    irradiances = np.arange(200, 1001, 200)
    colors_irr = plt.cm.viridis(np.linspace(0, 1, len(irradiances)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for irr, color in zip(irradiances, colors_irr):
        V_irr, I_irr = model.datasheet_curve(irradiance=irr, temperature=Tcell)
        axes[0].plot(V_irr, I_irr, color=color, label=f"{irr} W/m²")
        axes[1].plot(V_irr, V_irr * I_irr, color=color, label=f"{irr} W/m²")

    axes[0].set_title("I-V vs Irraggiamento")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend()

    axes[1].set_title("P-V vs Irraggiamento")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Power (W)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Analisi al variare della temperatura
    temperatures = np.arange(25, 101, 25)
    colors_temp = plt.cm.plasma(np.linspace(0, 1, len(temperatures)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for temp, color in zip(temperatures, colors_temp):
        V_temp, I_temp = model.datasheet_curve(irradiance=1000.0, temperature=temp)
        axes[0].plot(V_temp, I_temp, color=color, label=f"{temp} °C")
        axes[1].plot(V_temp, V_temp * I_temp, color=color, label=f"{temp} °C")

    axes[0].set_title("I-V vs Temperatura")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend()

    axes[1].set_title("P-V vs Temperatura")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Power (W)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
