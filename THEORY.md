# Mathematical scope

The retained NumPy statevector QAOA engine applies diagonal Ising cost phases
and XY mixing to a k-hot Dicke initialization. XY gates preserve Hamming weight.
The optimizer is finite-difference Adam ascent on expected energy; an optional
upper-tail CVaR objective explicitly maximizes the best alpha mass of the energy
distribution. Heuristic spin-glass angle schedules are retained without an
improvement claim. Objective convention is MAXIMIZATION.

Release metrics use the actual probability distribution: P(feasible), P(optimal),
conditional expected feasible energy, normalized regret/quality relative to the
feasible energy range, and shots-to-success. Degenerate ranges and zero feasible
mass are explicit. No best-over-all-bitstrings metric is reported as sampled
solution quality. Uniform plus-state and k-hot controls must differ from a
distribution concentrated on the optimum. Fortet-linearized SciPy MILP optima
are cross-checked against exact enumeration on small seeded instances.

This scoped first-party export excludes vendor solvers, commercial instances,
hardware paths, historical performance percentages and unsupported comparative
tests. Numerical bit ordering and normalization are regression-tested.

Exponential-memory NumPy simulation only; no hardware execution or quantum speedup claim. The normalized quality is gauge-invariant range scaling, not a raw energy ratio. CVaR tuning is a heuristic, not proof of practical advantage. General sector/turnover inequality encoding is not part of this cardinality-only release.
