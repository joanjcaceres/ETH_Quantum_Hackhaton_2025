import numpy as np
import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
from typing import Callable, Sequence, Optional, List, Tuple, Dict, Any

    
def tomography_and_evaluate(
    wigner_fn: Callable[[float, float], float],
    alpha_list: Sequence[complex],
    N_psi: int,
    rho_true: dq.QArray,
    N_fit: Optional[int] = None, 
    solver: str ='SCS',
    objective: str ='sum_squares',
    **kwargs
    ) -> Tuple[dq.QArray, List[float], List[dq.QArray], Dict[str, Any]]:

    """
    Perform quantum state tomography via Wigner sampling and convex optimization, 
    then compute standard fidelity and distance metrics.

    Parameters
    ----------
    wigner_fn : callable
        Function W(x, p) → float that evaluates the Wigner function at real
        phase-space coordinates x, p.
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
    metrics : dict
        Evaluation metrics with keys:
        - 'fidelity'    : float, state fidelity between true and reconstructed.
        - 'trace_dist'  : float, trace distance ½‖ρ_true − ρ_rec‖₁.
        - 'HS_dist'     : float, Hilbert-Schmidt distance.
        - 'purity_rec'  : float, purity of reconstructed state.
        - 'delta_purity': float, purity_true - purity_rec.
        - 'eig_true'    : ndarray, sorted eigenvalues of ρ_true.
        - 'eig_rec'     : ndarray, sorted eigenvalues of ρ_rec.

    Raises
    ------
    ValueError
        If `objective` is not one of {'sum_squares', 'sum_absolute'}.
    """
    max_iters = kwargs.get('max_iters', 1000)
    
    if N_fit is None:
        N_fit = 2 * N_psi  # ensure larger dimension
        
    # Calculate expected measurement outcomes from Wigner
    w_k = [0.5 * (1 + (np.pi / 2) * wigner_fn(alpha.real, alpha.imag))
           for alpha in alpha_list]
    
    # Build POVM elements E(alpha) = ½(I + D(alpha) P D(alpha)†)
    I_big = dq.eye(N_fit)
    P_big = dq.parity(N_fit)
    E_ops_big = [0.5 * (I_big + dq.displace(N_fit, alpha) @ P_big @ dq.displace(N_fit, alpha).dag())
             for alpha in alpha_list]
    
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
    rho_rec_q = dq.asqarray(rho_opt)

    # Evaluate metrics
    F = dq.fidelity(rho_true, rho_rec_q)
    eig_true = np.sort(np.real(np.array(jnp.linalg.eigvals(rho_true.data))))[::-1]
    eig_rec = np.sort(np.real(np.array(jnp.linalg.eigvals(rho_rec_q.data))))[::-1]
    delta = rho_true.data - rho_rec_q.data
    Tdist = 0.5 * jnp.sum(jnp.abs(jnp.linalg.eigvals(delta)))
    HSdist = jnp.sqrt(jnp.trace(delta @ delta))
    purity_true = jnp.trace(rho_true.data @ rho_true.data)
    purity_rec = jnp.trace(rho_rec_q.data @ rho_rec_q.data)
    d_purity = purity_true - purity_rec

    metrics = {
        'fidelity': float(F),
        'trace_dist': float(Tdist),
        'HS_dist': float(jnp.real(HSdist)),
        'purity_rec': float(jnp.real(purity_rec)),
        'delta_purity': float(jnp.real(d_purity)),
        'eig_true': eig_true,
        'eig_rec': eig_rec
    }

    return rho_rec_q, w_k, E_ops_big, metrics