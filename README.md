# SymNLPModels.jl
NLPModels.jl adapter for Symbolics.jl

## Usage
Minimize the function `(x - 3)^2` subject to `x^2 ≤ 1` using Ipopt.
```julia
using SymNLPModels: SymNLPModel
using NLPModelsIpopt: ipopt

@variables x

objective = (x - 3)^2
constraints = [x^2 ≲ 1]

model = SymNLPModels.SymNLPModel(objective, constraints)
stats = ipopt(model; tol=1e-4, print_level=0)

actual = stats.solution
```

## Note
This project makes little sense as you could use ModelingToolkit wotj OptimizationMOI to do the same.