import numpy as np
import pytest
from qaoa_cardinality.qaoa_v2 import QAOAv2,QAOAConfig,build_cost_hamiltonian
from qaoa_cardinality.metrics import distribution_metrics,upper_cvar
from qaoa_cardinality.exact_milp import maximum_ising_energy


@pytest.mark.parametrize('seed',range(12))
def test_fortet_milp_matches_full_enumeration(seed):
    n=4+seed%3;k=n//2;rng=np.random.default_rng(seed);J=rng.normal(size=(n,n));J=(J+J.T)/2
    energies=build_cost_hamiltonian(J)
    feasible=[i for i in range(2**n) if i.bit_count()==k]
    assert maximum_ising_energy(J,k) == pytest.approx(energies[feasible].max(),abs=1e-7)


def test_uniform_and_optimal_distributions_do_not_get_the_same_quality():
    n=4;k=2;J=np.diag([0.1,0.4,0.9,1.7]);e=build_cost_hamiltonian(J)
    feasible=np.array([i.bit_count()==k for i in range(16)])
    optimal=np.flatnonzero(feasible)[np.argmax(e[feasible])]
    uniform=np.ones(16)/16;delta=np.zeros(16);delta[optimal]=1
    u=distribution_metrics(uniform,e,n,k);d=distribution_metrics(delta,e,n,k)
    assert u['p_feasible'] == pytest.approx(6/16)
    assert d['p_optimal']==1 and u['p_optimal']<1
    assert u['normalized_regret']>d['normalized_regret']==0
    shifted=distribution_metrics(uniform,e+200,n,k)
    assert shifted['normalized_quality']==pytest.approx(u['normalized_quality'])


def test_zero_feasible_mass_is_not_called_optimal():
    p=np.zeros(16);p[0]=1
    result=distribution_metrics(p,np.arange(16),4,2)
    assert result['normalized_quality'] is None
    assert result['p_optimal']==0 and result['shots_to_confidence'] is None


def test_cvar_and_seeded_dicke_fit():
    assert upper_cvar([0.5,0.5],[0.,2.],0.5)==2
    assert upper_cvar([0.5,0.5],[0.,2.],1.)==1
    J=np.diag([0.1,0.4,0.9,1.7])
    models=[QAOAv2(QAOAConfig(p=1,n_iter=15,target_k=2,seed=41,cvar_alpha=0.3)).fit(J) for _ in range(2)]
    assert np.allclose(models[0].get_state(),models[1].get_state())
    metrics=distribution_metrics(abs(models[0].get_state())**2,models[0].H_diag_,4,2)
    assert metrics['p_feasible']==pytest.approx(1,abs=1e-10)
