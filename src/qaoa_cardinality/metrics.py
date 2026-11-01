"""Metrics that depend on measured/simulated probability mass, not just search space."""
import math
import numpy as np


def distribution_metrics(probabilities,energies,n,k,confidence=0.99):
    p=np.asarray(probabilities,float);e=np.asarray(energies,float)
    if p.shape!=(2**n,) or e.shape!=p.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(e)):
        raise ValueError('finite vectors of size 2**n required')
    if np.any(p<0) or not np.isclose(p.sum(),1) or not 0<confidence<1:
        raise ValueError('normalized probabilities and confidence in (0,1) required')
    mask=np.array([i.bit_count()==k for i in range(len(p))])
    if not mask.any():raise ValueError('empty feasible set')
    best=float(e[mask].max());worst=float(e[mask].min())
    p_feasible=float(p[mask].sum())
    p_opt=float(p[mask & np.isclose(e,best,rtol=0,atol=1e-10)].sum())
    expected=float(p[mask]@e[mask]/p_feasible) if p_feasible>0 else None
    quality=(float(np.clip((expected-worst)/(best-worst),0,1)) if best>worst+1e-12 else 1.) if expected is not None else None
    shots=(1 if p_opt>=1-1e-12 else math.ceil(math.log1p(-confidence)/math.log1p(-p_opt))) if p_opt>0 else None
    return {'p_feasible':p_feasible,'p_optimal':p_opt,'expected_feasible_energy':expected,
            'normalized_quality':quality,'normalized_regret':None if quality is None else 1-quality,
            'shots_to_confidence':shots,'confidence':confidence,'exact_optimum':best}


def upper_cvar(probabilities,energies,alpha=1.):
    if not 0<alpha<=1:raise ValueError('alpha must be in (0,1]')
    p=np.asarray(probabilities,float);e=np.asarray(energies,float);order=np.argsort(e)[::-1]
    used=np.clip(alpha-np.r_[0,np.cumsum(p[order])[:-1]],0,p[order])
    return float(used@e[order]/alpha)
