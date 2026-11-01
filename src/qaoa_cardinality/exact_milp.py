"""Independent Fortet binary-product MILP for the cardinality-constrained Ising optimum."""
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint


def maximum_ising_energy(J,k):
    J=np.asarray(J,float);n=len(J);pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    h=np.diag(J);linear=-2*h.copy();constant=float(h.sum())
    for i,j in pairs:
        linear[i]-=2*J[i,j];linear[j]-=2*J[i,j];constant+=J[i,j]
    c=-np.r_[linear,[4*J[i,j] for i,j in pairs]]
    m=len(c);rows=[];lo=[];hi=[]
    row=np.zeros(m);row[:n]=1;rows.append(row);lo.append(k);hi.append(k)
    for t,(i,j) in enumerate(pairs):
        y=n+t
        for index in [i,j]:
            row=np.zeros(m);row[y]=1;row[index]=-1;rows.append(row);lo.append(-np.inf);hi.append(0)
        row=np.zeros(m);row[y]=1;row[i]=-1;row[j]=-1;rows.append(row);lo.append(-1);hi.append(np.inf)
    result=milp(c,integrality=np.r_[np.ones(n),np.zeros(len(pairs))],bounds=Bounds(np.zeros(m),np.ones(m)),
                constraints=LinearConstraint(np.array(rows),np.array(lo),np.array(hi)),options={'mip_rel_gap':0.})
    if not result.success:raise RuntimeError(result.message)
    return float(-result.fun+constant)
