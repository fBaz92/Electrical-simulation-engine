from __future__ import annotations
import numpy as np
from typing import Tuple

_SQRT_3 = np.sqrt(3.0)


def abc_to_alpha_beta(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Clarke transform: Convert three-phase abc quantities to alpha-beta (stationary) frame.
    
    This transformation converts three-phase balanced quantities (currents, voltages,
    or fluxes) from the abc frame to the two-phase alpha-beta stationary reference
    frame. The transformation assumes a balanced system (a + b + c = 0) and eliminates
    the zero-sequence component.
    
    The transformation matrix is:
        [α]   [2/3  -1/3  -1/3 ] [a]
        [β] = [0    √3/3  -√3/3] [b]
        [0]   [1/3   1/3   1/3 ] [c]
    
    For balanced systems, the zero-sequence component is zero, so only α and β
    are computed.
    
    Mathematical formulas:
        α = (2/3) * (a - 0.5*(b + c))
        β = (2/3) * (√3/2) * (b - c)
    
    Args:
        a: Phase A quantity (A, V, or Wb).
        b: Phase B quantity (A, V, or Wb).
        c: Phase C quantity (A, V, or Wb).
    
    Returns:
        Tuple (alpha, beta) containing:
        - alpha: Alpha-axis component (same units as input).
        - beta: Beta-axis component (same units as input).
    
    Note:
        This transform assumes balanced three-phase quantities (a + b + c = 0).
        For unbalanced systems, use a full Clarke transform that includes zero-sequence.
    
    Example:
        >>> # Balanced three-phase currents: 10A peak, 120° phase separation
        >>> i_a, i_b, i_c = 10.0, -5.0, -5.0
        >>> alpha, beta = abc_to_alpha_beta(i_a, i_b, i_c)
    """
    alpha = (2.0 / 3.0) * (a - 0.5 * (b + c))
    beta = (2.0 / 3.0) * ((_SQRT_3 / 2.0) * (b - c))
    return alpha, beta


def alpha_beta_to_abc(alpha: float, beta: float) -> Tuple[float, float, float]:
    """
    Inverse Clarke transform: Convert alpha-beta (stationary) frame to three-phase abc.
    
    This transformation converts two-phase alpha-beta quantities back to three-phase
    abc quantities. The transformation assumes zero-sequence component is zero
    (balanced system), which is typical for three-phase systems without neutral
    connection.
    
    The inverse transformation matrix is:
        [a]   [1     0   ] [α]
        [b] = [-1/2  √3/2] [β]
        [c]   [-1/2 -√3/2] [0]
    
    Mathematical formulas:
        a = α
        b = -0.5*α + (√3/2)*β
        c = -0.5*α - (√3/2)*β
    
    Args:
        alpha: Alpha-axis component (A, V, or Wb).
        beta: Beta-axis component (A, V, or Wb).
    
    Returns:
        Tuple (a, b, c) containing:
        - a: Phase A quantity (same units as input).
        - b: Phase B quantity (same units as input).
        - c: Phase C quantity (same units as input).
    
    Note:
        This transform assumes zero-sequence = 0 (balanced system). The resulting
        abc quantities will always satisfy a + b + c = 0.
    
    Example:
        >>> # Convert alpha-beta back to three-phase
        >>> alpha, beta = 8.66, 5.0
        >>> i_a, i_b, i_c = alpha_beta_to_abc(alpha, beta)
    """
    a = alpha
    b = -0.5 * alpha + (_SQRT_3 / 2.0) * beta
    c = -0.5 * alpha - (_SQRT_3 / 2.0) * beta
    return a, b, c


def park_transform(alpha: float, beta: float, theta_e: float) -> Tuple[float, float]:
    """
    Park transform: Convert alpha-beta (stationary) frame to dq (rotating) frame.
    
    This transformation rotates the stationary alpha-beta reference frame to a
    rotating dq reference frame synchronized with the rotor. The d-axis typically
    aligns with the rotor flux (or permanent magnet flux), while the q-axis leads
    the d-axis by 90 electrical degrees.
    
    The transformation matrix is:
        [d]   [cos(θ_e)   sin(θ_e)] [α]
        [q] = [-sin(θ_e)  cos(θ_e)] [β]
    
    Mathematical formulas:
        d = α*cos(θ_e) + β*sin(θ_e)
        q = -α*sin(θ_e) + β*cos(θ_e)
    
    In the dq frame, AC quantities at synchronous frequency appear as DC quantities,
    simplifying control and analysis.
    
    Args:
        alpha: Alpha-axis component in stationary frame (A, V, or Wb).
        beta: Beta-axis component in stationary frame (A, V, or Wb).
        theta_e: Electrical angle of the dq reference frame (rad). This is typically
            pole_pairs * mechanical_angle or the angle of the rotor flux.
    
    Returns:
        Tuple (d, q) containing:
        - d: Direct-axis component in rotating frame (same units as input).
        - q: Quadrature-axis component in rotating frame (same units as input).
    
    Note:
        The dq frame rotates at electrical speed omega_e = d(theta_e)/dt. For
        synchronous machines, theta_e = pole_pairs * theta_mech, where theta_mech
        is the mechanical rotor angle.
    
    Example:
        >>> # Transform stationary frame currents to rotating dq frame
        >>> alpha, beta = 10.0, 5.0
        >>> theta_e = np.pi / 4  # 45 electrical degrees
        >>> i_d, i_q = park_transform(alpha, beta, theta_e)
    """
    cos_t = np.cos(theta_e)
    sin_t = np.sin(theta_e)
    d = alpha * cos_t + beta * sin_t
    q = -alpha * sin_t + beta * cos_t
    return d, q


def inverse_park_transform(d: float, q: float, theta_e: float) -> Tuple[float, float]:
    """
    Inverse Park transform: Convert dq (rotating) frame to alpha-beta (stationary) frame.
    
    This transformation rotates the dq reference frame back to the stationary
    alpha-beta frame. It is the inverse of the Park transform and is used to
    convert control commands from the dq frame back to the stationary frame for
    implementation.
    
    The inverse transformation matrix is:
        [α]   [cos(θ_e)  -sin(θ_e)] [d]
        [β] = [sin(θ_e)   cos(θ_e)] [q]
    
    Mathematical formulas:
        α = d*cos(θ_e) - q*sin(θ_e)
        β = d*sin(θ_e) + q*cos(θ_e)
    
    Args:
        d: Direct-axis component in rotating frame (A, V, or Wb).
        q: Quadrature-axis component in rotating frame (A, V, or Wb).
        theta_e: Electrical angle of the dq reference frame (rad). Must match the
            angle used in the forward Park transform.
    
    Returns:
        Tuple (alpha, beta) containing:
        - alpha: Alpha-axis component in stationary frame (same units as input).
        - beta: Beta-axis component in stationary frame (same units as input).
    
    Note:
        The electrical angle theta_e must be the same as used in the forward
        Park transform to ensure correct transformation.
    
    Example:
        >>> # Convert dq frame currents back to stationary frame
        >>> i_d, i_q = 5.0, 10.0
        >>> theta_e = np.pi / 4  # 45 electrical degrees
        >>> alpha, beta = inverse_park_transform(i_d, i_q, theta_e)
    """
    cos_t = np.cos(theta_e)
    sin_t = np.sin(theta_e)
    alpha = d * cos_t - q * sin_t
    beta = d * sin_t + q * cos_t
    return alpha, beta


def abc_to_dq(a: float, b: float, c: float, theta_e: float) -> Tuple[float, float]:
    """
    Combined Clarke-Park transform: Convert three-phase abc to dq (rotating) frame.
    
    This function combines the Clarke transform (abc -> alpha-beta) and Park
    transform (alpha-beta -> dq) into a single operation. It directly converts
    three-phase quantities to the rotating dq reference frame.
    
    The transformation sequence is:
        1. abc -> alpha-beta (Clarke transform)
        2. alpha-beta -> dq (Park transform)
    
    Mathematical formulas:
        α = (2/3) * (a - 0.5*(b + c))
        β = (2/3) * (√3/2) * (b - c)
        d = α*cos(θ_e) + β*sin(θ_e)
        q = -α*sin(θ_e) + β*cos(θ_e)
    
    Args:
        a: Phase A quantity (A, V, or Wb).
        b: Phase B quantity (A, V, or Wb).
        c: Phase C quantity (A, V, or Wb).
        theta_e: Electrical angle of the dq reference frame (rad).
    
    Returns:
        Tuple (d, q) containing:
        - d: Direct-axis component in rotating frame (same units as input).
        - q: Quadrature-axis component in rotating frame (same units as input).
    
    Note:
        Assumes balanced three-phase system (a + b + c = 0). The dq frame rotates
        at electrical speed omega_e = d(theta_e)/dt.
    
    Example:
        >>> # Convert three-phase currents to dq frame
        >>> i_a, i_b, i_c = 10.0, -5.0, -5.0
        >>> theta_e = np.pi / 6  # 30 electrical degrees
        >>> i_d, i_q = abc_to_dq(i_a, i_b, i_c, theta_e)
    """
    alpha, beta = abc_to_alpha_beta(a, b, c)
    return park_transform(alpha, beta, theta_e)


def dq_to_abc(d: float, q: float, theta_e: float) -> Tuple[float, float, float]:
    """
    Combined inverse Park-Clarke transform: Convert dq (rotating) frame to three-phase abc.
    
    This function combines the inverse Park transform (dq -> alpha-beta) and
    inverse Clarke transform (alpha-beta -> abc) into a single operation. It
    directly converts dq frame quantities to three-phase abc quantities.
    
    The transformation sequence is:
        1. dq -> alpha-beta (inverse Park transform)
        2. alpha-beta -> abc (inverse Clarke transform)
    
    Mathematical formulas:
        α = d*cos(θ_e) - q*sin(θ_e)
        β = d*sin(θ_e) + q*cos(θ_e)
        a = α
        b = -0.5*α + (√3/2)*β
        c = -0.5*α - (√3/2)*β
    
    Args:
        d: Direct-axis component in rotating frame (A, V, or Wb).
        q: Quadrature-axis component in rotating frame (A, V, or Wb).
        theta_e: Electrical angle of the dq reference frame (rad).
    
    Returns:
        Tuple (a, b, c) containing:
        - a: Phase A quantity (same units as input).
        - b: Phase B quantity (same units as input).
        - c: Phase C quantity (same units as input).
    
    Note:
        The resulting abc quantities will satisfy a + b + c = 0 (balanced system).
        This function is commonly used to convert dq control commands to three-phase
        voltages or currents for PWM generation.
    
    Example:
        >>> # Convert dq frame currents to three-phase
        >>> i_d, i_q = 5.0, 10.0
        >>> theta_e = np.pi / 6  # 30 electrical degrees
        >>> i_a, i_b, i_c = dq_to_abc(i_d, i_q, theta_e)
    """
    alpha, beta = inverse_park_transform(d, q, theta_e)
    return alpha_beta_to_abc(alpha, beta)


def dq_to_abc_series(d: np.ndarray, q: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized dq-to-abc transformation for time series data.
    
    This function performs the combined inverse Park-Clarke transform on arrays
    of dq quantities, efficiently converting time series data from the rotating
    dq frame to three-phase abc quantities. It is optimized for processing
    simulation results or measurement data.
    
    The transformation is applied element-wise to the input arrays:
        α[i] = d[i]*cos(θ[i]) - q[i]*sin(θ[i])
        β[i] = d[i]*sin(θ[i]) + q[i]*cos(θ[i])
        a[i] = α[i]
        b[i] = -0.5*α[i] + (√3/2)*β[i]
        c[i] = -0.5*α[i] - (√3/2)*β[i]
    
    Args:
        d: Array of direct-axis quantities in rotating frame (A, V, or Wb).
            Shape: (N,) where N is the number of time points.
        q: Array of quadrature-axis quantities in rotating frame (A, V, or Wb).
            Shape: (N,) where N is the number of time points.
        theta: Array of electrical angles (rad). Shape: (N,) where N is the number
            of time points. Each element theta[i] corresponds to the electrical
            angle at time point i.
    
    Returns:
        Tuple (a, b, c) containing:
        - a: Array of Phase A quantities (same units as input). Shape: (N,)
        - b: Array of Phase B quantities (same units as input). Shape: (N,)
        - c: Array of Phase C quantities (same units as input). Shape: (N,)
    
    Note:
        All input arrays must have the same length. The function uses NumPy
        vectorized operations for efficient computation. The resulting abc
        quantities will satisfy a[i] + b[i] + c[i] = 0 for all i (balanced system).
    
    Example:
        >>> # Convert dq current traces from simulation to three-phase
        >>> import numpy as np
        >>> t = np.linspace(0, 1, 1000)
        >>> i_d = 5.0 * np.ones_like(t)  # Constant d-axis current
        >>> i_q = 10.0 * np.sin(2 * np.pi * 50 * t)  # Sinusoidal q-axis current
        >>> theta_e = 2 * np.pi * 50 * t  # Electrical angle
        >>> i_a, i_b, i_c = dq_to_abc_series(i_d, i_q, theta_e)
    """
    alpha = d * np.cos(theta) - q * np.sin(theta)
    beta = d * np.sin(theta) + q * np.cos(theta)
    a = alpha
    b = -0.5 * alpha + (_SQRT_3 / 2.0) * beta
    c = -0.5 * alpha - (_SQRT_3 / 2.0) * beta
    return a, b, c
