import numpy as np
import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
    
def tomography_and_evaluate(
    psi,
    xvec,
    pvec, 
    alpha_list,
    N_fit = None, 
    sigma = 0.0,
    solver='SCS',
    objective='sum_squares'
    ):
    """
    Full pipeline: Wigner sampling → convex optimization → evaluation metrics.
    
    Inputs:
      psi        : pure state ket (DenseQArray)
      xvec, pvec : 1D numpy arrays defining the Wigner grid
      alpha_list : list of complex displacements alpha_k
      N_fit      : Hilbert dimension for fitting (default: max(2*N_psi, N_psi))
      sigma      : noise level for Wigner sampling (default: 0.0)
      solver     : CVXPY solver name
      objective  : CVXPY objective function type (default: 'sum_squares')
    
    Returns:
      rho_rec_q  : reconstructed density matrix (DenseQArray)
      w_k        : list of expected probabilities from Wigner data
      E_ops      : list of dynamiqs QArray observables E(alpha_k)
      metrics    : dict with fidelity, trace_dist, HS_dist, delta_purity, eig_true, eig_rec
    """
    
    N_psi = psi.shape[0]
    if N_fit is None:
        N_fit = max(N_psi, 2 * N_psi)  # ensure larger dimension
        
    # 1) Compute Wigner on (xvec,pvec) and convert to NumPy
    xg, pg, W_jax = dq.wigner(psi, xvec=xvec, yvec=pvec)
    W_grid = np.asarray(W_jax)
    # 2) Build interpolator
    W_interp = RegularGridInterpolator((xg, pg), W_grid,
                                       method='linear', bounds_error=False, fill_value=0.0)
    # 3) Observed click-rates w_k
    noise = np.random.normal(loc=0.0, scale=sigma, size=len(alpha_list))  # changed code
    w_k = [0.5 * (1 + (np.pi/2) * W_interp((alpha.real, alpha.imag)) + noise[i])
           for i, alpha in enumerate(alpha_list)]
    # 4) Build POVM elements E(alpha) = ½(I + D(alpha) P D(alpha)†)
    I = dq.eye(N_fit)
    P = dq.parity(N_fit)
    E_ops = [0.5 * (I + dq.displace(N_fit, alpha) @ P @ dq.displace(N_fit, alpha).dag())
             for alpha in alpha_list]
    
    # 5) Convert E_ops → NumPy matrices for CVXPY
    E_mats = []
    for E in E_ops:
        E_np = np.array(E.data)[:N_psi, :N_psi]
        E_np = 0.5 * (E_np + E_np.conj().T)  # ensure Hermitian
        E_mats.append(E_np)
        
    # 6) Solve least-squares tomography via CVXPY
    rho_var = cp.Variable((N_psi, N_psi), complex=True)
    cons = [
        rho_var >> 0,
        cp.trace(rho_var) == 1,
        rho_var == rho_var.H
    ]
    p_preds = [cp.real(cp.trace(E_mats[i] @ rho_var)) for i in range(len(E_mats))]
    # Select optimization objective
    if objective == 'sum_squares':
        obj = cp.Minimize(cp.sum_squares(cp.hstack(p_preds) - np.array(w_k)))
    elif objective == 'sum_absolute':
        obj = cp.Minimize(cp.norm(cp.hstack(p_preds) - np.array(w_k), 1))
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)
    rho_opt = rho_var.value
    rho_rec_q = dq.asqarray(rho_opt)
    # 7) Compute evaluation metrics
    rho_true = psi @ psi.dag()
    F = dq.fidelity(rho_true, rho_rec_q)
    eig_true = np.sort(np.real(np.array(jnp.linalg.eigvals(rho_true.data))))[::-1]
    eig_rec  = np.sort(np.real(np.array(jnp.linalg.eigvals(rho_rec_q.data))))[::-1]
    delta = rho_true.data - rho_rec_q.data
    Tdist = 0.5 * jnp.sum(jnp.abs(jnp.linalg.eigvals(delta)))
    HSdist = jnp.sqrt(jnp.trace(delta @ delta))
    purity_true = jnp.trace(rho_true.data @ rho_true.data)
    purity_rec  = jnp.trace(rho_rec_q.data  @ rho_rec_q.data)
    d_purity    = purity_true - purity_rec
    metrics = {
        'fidelity': float(F),
        'trace_dist': float(Tdist),
        'HS_dist': float(jnp.real(HSdist)),
        'delta_purity': float(jnp.real(d_purity)),
        'eig_true': eig_true,
        'eig_rec': eig_rec
    }
    return rho_rec_q, w_k, E_ops, metrics