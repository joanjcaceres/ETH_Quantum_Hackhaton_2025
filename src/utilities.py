import numpy as np
import cvxpy as cp
import dynamiqs as dq
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class TomographyMetrics:
    fidelity: float
    trace_dist: float
    HS_dist: float
    purity_rec: float
    eig_true: np.ndarray
    eig_rec: np.ndarray
    
    def __repr__(self) -> str:
        return (
                f"  fidelity: {self.fidelity:.6f},\n"
                f"  trace_dist: {self.trace_dist:.6f},\n"
                f"  HS_dist: {self.HS_dist:.6f},\n"
                f"  purity_rec: {self.purity_rec:.6f},\n"
                f"  eig_true: {np.array2string(self.eig_true[:3], precision=4, separator=', ')},\n"
                f"  eig_rec: {np.array2string(self.eig_rec[:3], precision=4, separator=', ')}\n"
                )
    
    
def rho_cat_state(N: int, alpha: float, sign: str = '+') -> dq.QArray:
    """
    Generate a normalized cat state (|+⟩ or |−⟩) and return its density matrix.

    Parameters
    ----------
    N : int
        Hilbert space dimension.
    beta : float
        Coherent state amplitude.
    sign : str
        '+' for cat plus (|β⟩ + |−β⟩), '-' for cat minus (|β⟩ − |−β⟩).

    Returns
    -------
    rho_cat : dq.QArray
        Density matrix of the normalized cat state.
    """
    coh_plus = dq.coherent(N, alpha)
    coh_minus = dq.coherent(N, -alpha)

    if sign == '+':
        cat = dq.unit(coh_plus + coh_minus)
    elif sign == '-':
        cat = dq.unit(coh_plus - coh_minus)
    else:
        raise ValueError("sign must be '+' or '-'")

    return cat @ cat.dag()

    
def rho_reconstruction(
    W_grid: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    alpha_list: Sequence[complex],
    N_psi: int,
    rho_reference: dq.QArray,
    N_fit: Optional[int] = None, 
    solver: str ='SCS',
    objective: str ='sum_squares',
    **kwargs
    ) -> Tuple[dq.QArray, List[float], List[dq.QArray], TomographyMetrics]:

    """
    Perform quantum state tomography via Wigner sampling and convex optimization, 
    then compute standard fidelity and distance metrics.

    Parameters
    ----------
    W_grid : np.ndarray
        2D array of Wigner function values on a grid defined by xvec and pvec.
    xvec : np.ndarray
        1D array of x-coordinates for the Wigner grid.
    pvec : np.ndarray
        1D array of p-coordinates for the Wigner grid.
    alpha_list : sequence of complex
        Displacement parameters alpha_k used to probe the state.
    N_psi : int
        Dimension of the true state’s Hilbert space.
    rho_true : dynamiqs.QArray
        The true density matrix of the quantum state.
    N_fit : int, optional
        Dimension of the reconstruction Hilbert space (default: 2 * N_psi).
    solver : str, optional
        Name of the CVXPY solver to use (e.g. 'SCS', 'CVXOPT').
    objective : {'sum_squares', 'sum_absolute'}, optional
        Choice of CVXPY objective:
        - 'sum_squares': minimize ∑ (p_pred - w_k)²  
        - 'sum_absolute': minimize ∥p_pred - w_k∥₁  
    **kwargs : dict, optional
        Extra keyword args passed to `problem.solve()`, e.g. max_iters=int.

    Returns
    -------
    rho_rec_q : dynamiqs.QArray
        Reconstructed density matrix as a Dynamiqs QArray.
    w_k : list of float
        Expected probabilities from Wigner sampling at each alpha_k.
    E_ops_big : list of dynamiqs.QArray
        POVM elements E(alpha_k) = ½(I + D(alpha) P D(alpha)†) used in the fit.
    metrics : TomographyMetrics
        Evaluation metrics:
        - fidelity    : float, state fidelity between true and reconstructed.
        - trace_dist  : float, trace distance ½‖rho_true - rho_rec‖₁.
        - HS_dist     : float, Hilbert-Schmidt distance.
        - purity_rec  : float, purity of reconstructed state.
        - eig_true    : ndarray, sorted eigenvalues of rho_true.
        - eig_rec     : ndarray, sorted eigenvalues of rho_rec.
        

    Raises
    ------
    ValueError
        If `objective` is not one of {'sum_squares', 'sum_absolute'}.
    """
    max_iters = kwargs.get('max_iters', 1000)

    if W_grid.shape != (len(xvec), len(pvec)):
        raise ValueError(f"W_grid shape {W_grid.shape} does not match xvec and pvec lengths {len(xvec)}, {len(pvec)}")
        
    W_interp = RegularGridInterpolator((xvec, pvec), W_grid, bounds_error=False, fill_value=0.0)
        
    # Calculate expected measurement outcomes from Wigner
    alphas = np.array(alpha_list)
    x_coords = alphas.real
    p_coords = alphas.imag
    points = np.column_stack([x_coords, p_coords])  # shape (N, 2)
    w_values = W_interp(points)  # vectorized evaluation
    w_k = 0.5 * (1 + (np.pi / 2) * w_values)
    
    if N_fit is None:
        N_fit = 2 * N_psi  # ensure larger dimension
    # Build POVM elements E(alpha) vectorized via jax.vmap
    I_big = dq.eye(N_fit)
    P_big = dq.parity(N_fit)
    alphas_jax = jnp.array(alpha_list)
    def _build_E_op(alpha):
        D = dq.displace(N_fit, alpha)
        return 0.5 * (I_big + D @ P_big @ D.dag())
    # Vectorized map over all alphas without explicit Python loops
    E_ops_big = list(jax.vmap(_build_E_op)(alphas_jax))
    
    # Convert E_ops → NumPy matrices for CVXPY
    E_matrices = []
    for E in E_ops_big:
        E_np = np.array(E.data)[:N_psi, :N_psi]
        E_np = 0.5 * (E_np + E_np.conj().T)  # ensure Hermitian
        E_matrices.append(E_np)
        
    # 6) Solve least-squares tomography via CVXPY
    rho_var = cp.Variable((N_psi, N_psi), complex=True)
    constraints = [
        rho_var >> 0, # positive semidefinite
        cp.trace(rho_var) == 1, # trace = 1
        rho_var == rho_var.H # Hermitian
    ]
    
    p_preds = [cp.real(cp.trace(E_matrices[i] @ rho_var)) for i in range(len(E_matrices))]
    # Select optimization objective
    if objective == 'sum_squares':
        obj = cp.Minimize(cp.sum_squares(cp.hstack(p_preds) - np.array(w_k)))
    elif objective == 'sum_absolute':
        obj = cp.Minimize(cp.norm(cp.hstack(p_preds) - np.array(w_k), 1))
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    problem = cp.Problem(obj, constraints)
    problem.solve(solver=solver, max_iters=max_iters)
    rho_opt = rho_var.value
    rho_reconstructed = dq.asqarray(rho_opt)

    # Evaluate metrics
    F = dq.fidelity(rho_reference, rho_reconstructed)
    eig_true = np.sort(np.real(np.array(jnp.linalg.eigvals(rho_reference.data))))[::-1]
    eig_rec = np.sort(np.real(np.array(jnp.linalg.eigvals(rho_reconstructed.data))))[::-1]
    delta = rho_reference.data - rho_reconstructed.data
    Tdist = 0.5 * jnp.sum(jnp.abs(jnp.linalg.eigvals(delta)))
    HSdist = jnp.sqrt(jnp.trace(delta @ delta))
    purity_rec = jnp.trace(rho_reconstructed.data @ rho_reconstructed.data)

    metrics = TomographyMetrics(
        fidelity=float(F),
        trace_dist=float(Tdist),
        HS_dist=float(jnp.real(HSdist)),
        purity_rec=float(jnp.real(purity_rec)),
        eig_true=eig_true,
        eig_rec=eig_rec
    )

    return rho_reconstructed, w_k, E_ops_big, metrics

def integral_2d(dx: float, dy: float, z: np.ndarray) -> float:
    """Compute the 2D integral over a grid with spacing dx and dy, ignoring NaNs."""
    return float(dx * dy * np.nansum(z))


def wigner_affine_correction(
    x_values: np.ndarray,
    y_values: np.ndarray,
    wigner_values: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Rescale a Wigner function by removing a background offset and normalizing its integral.

    Parameters
    ----------
    x_values : np.ndarray
        1D array of x-coordinates.
    y_values : np.ndarray
        1D array of y-coordinates.
    wigner_values : np.ndarray
        2D array of Wigner function values.

    Returns
    -------
    corrected_wigner : np.ndarray
        Affine-rescaled Wigner function.
    a : float
        Normalization factor used in the rescaling.
    b : float
        Background offset estimated outside a defined radius.
    """
    # Create meshgrid for all x,y points
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    
    # Create a mask for points at the edge of the grid
    not_nan_mask = ~np.isnan(wigner_values)
    border_mask = np.zeros_like(wigner_values, dtype=bool)

    lines= 2
    border_mask[0:lines, :] = not_nan_mask[0:lines, :]  # Left edge (first two rows)
    border_mask[-lines:, :] = not_nan_mask[-lines:, :]  # Right edge (last two rows)
    border_mask[:, 0:lines] = not_nan_mask[:, 0:lines]  # Bottom edge (first two columns)
    border_mask[:, -lines:] = not_nan_mask[:, -lines:]  # Top edge (last two columns)

    # Remove the nan values of the edge mask
    border_mask[np.isnan(wigner_values)] = False

    offset = np.mean(wigner_values[border_mask])
    
    # Replace NaNs with zeros for normalization and correction
    wigner_clean = np.nan_to_num(wigner_values, nan=offset)
    
    # Compute normalization on the cleaned data
    normalization = np.trapezoid(
        np.trapezoid(wigner_clean - offset, x=y_values, axis=1),
        x=x_values
    )
    
    # Apply affine correction
    corrected_wigner = (wigner_clean - offset) / normalization
    return corrected_wigner, normalization, offset

def get_denoising_fidelity(
    wigner_data: np.ndarray,
    x_values : np.ndarray,
    y_values: np.ndarray,
    noiseless_quantum_state: dq.QArrayLike
    ):
    
    x_jax = jnp.asarray(x_values)
    y_jax = jnp.asarray(y_values)

    # Compute Wigner function using Dynamiqs
    _,_, true_wigner = dq.wigner(noiseless_quantum_state, xvec=x_jax, yvec=y_jax)
    true_wigner = jnp.nan_to_num(true_wigner)

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]
    fidelity = jnp.pi * jnp.sum(true_wigner* wigner_data) * dx * dy

    return float(fidelity)