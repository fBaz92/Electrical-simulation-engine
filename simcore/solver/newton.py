from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

@dataclass
class NewtonConfig:
    """
    Configuration parameters for Newton-Raphson solver.
    
    Attributes:
        tol: Convergence tolerance for residual norm (default: 1e-8).
        max_iter: Maximum number of Newton iterations (default: 50).
        damping: Enable line search/backtracking damping (default: True).
        c1: Line search reduction factor for step size (default: 0.5).
        rho: Sufficient decrease factor for Armijo condition (default: 0.9).
    """
    tol: float = 1e-8
    max_iter: int = 50
    damping: bool = True
    c1: float = 0.5       # fattore riduzione line-search
    rho: float = 0.9      # sufficiente diminuzione

def newton_solve(FJ_fun, x0: Array, cfg: NewtonConfig) -> Array:
    """
    Solve a nonlinear system of equations using Newton-Raphson method with optional damping.
    
    The method iteratively solves F(x) = 0 by linearizing around the current point:
    J * delta = -F, where J is the Jacobian and F is the residual.
    
    Features:
    - Automatic regularization for singular/ill-conditioned Jacobians
    - Optional backtracking line search for global convergence
    - Armijo condition for sufficient decrease
    
    Args:
        FJ_fun: Function that takes x and returns (F, J) where:
            - F: Residual vector (should be zero at solution)
            - J: Jacobian matrix (dF/dx)
        x0: Initial guess for the solution.
        cfg: NewtonConfig with solver parameters (tolerance, max iterations, etc.).
    
    Returns:
        Solution vector x such that F(x) ≈ 0. Returns the last iterate even if
        convergence is not achieved (within tolerance or max iterations).
    """
    x = x0.copy()
    for _ in range(cfg.max_iter):
        F, J = FJ_fun(x)
        nF = np.linalg.norm(F)
        if nF < cfg.tol:
            return x
        try:
            delta = np.linalg.solve(J + 1e-12*np.eye(J.shape[0]), -F)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J + 1e-9*np.eye(J.shape[0]), -F, rcond=None)[0]

        if not cfg.damping:
            x += delta
            continue

        # backtracking
        alpha = 1.0
        base = nF
        while alpha > 1e-4:
            x_try = x + alpha * delta
            F_try, _ = FJ_fun(x_try)
            if np.linalg.norm(F_try) < cfg.rho * base:
                x = x_try
                break
            alpha *= cfg.c1
        if alpha <= 1e-4:
            x += 1e-4 * delta
    return x  # restituisce l'ultimo tentativo (anche se non convergente)
