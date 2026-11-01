import json
import numpy as np
from qaoa_cardinality import QAOAv2,QAOAConfig
from qaoa_cardinality.qaoa_v2 import qubo_to_ising_matrix
from qaoa_cardinality.metrics import distribution_metrics
from qaoa_cardinality.exact_milp import maximum_ising_energy

rng=np.random.default_rng(2026);A=rng.normal(size=(5,5));cov=A@A.T/50;mu=rng.uniform(0.01,0.1,5)
Q=np.diag(mu)-cov # maximize synthetic return-minus-risk with exactly two selected
J=qubo_to_ising_matrix(Q)
model=QAOAv2(QAOAConfig(p=2,n_iter=35,target_k=2,seed=2026,cvar_alpha=0.5)).fit(J)
actual=distribution_metrics(abs(model.get_state())**2,model.H_diag_,5,2)
control=distribution_metrics(np.ones(32)/32,model.H_diag_,5,2)
print(json.dumps({'seed':2026,'execution':'NUMPY_STATEVECTOR','qaoa':actual,'uniform_control':control,
                 'independent_milp_optimum':maximum_ising_energy(J,2)},indent=2))
