# SymNLPModels.jl
NLPModels.jl adapter for Symbolics.jl

## Univariate Concave Example
Minimize the function `(x - 3)^2` subject to `x^2 ≤ 1` using Ipopt.
```julia
using SymNLPModels: SymNLPModel
using NLPModelsIpopt: ipopt
using Symbolics

@variables x

objective = (x - 3)^2
constraints = [x^2 ≲ 1]

model = SymNLPModel(objective, constraints)
stats = ipopt(model; tol=1e-4, print_level=0)

actual = stats.solution
```

## Multivariate Concave Example
Provide `variables` explicitly in the model, or use `parse_solution` and `value` if variable order is important.
```julia
using SymNLPModels: SymNLPModel, value, parse_solution
using NLPModelsIpopt: ipopt
using Symbolics

@variables x[1:5]
X = Symbolics.scalarize(x)
center = randn(length(X))

objective = sum((X .- center).^2)
constraints = X.^2 .≲ 1
tol = 1e-4

model = SymNLPModel(objective, constraints;)
stats = ipopt(model; )

expected = clamp.(center, -1, 1)
solution = parse_solution(model, stats.solution)
actual = value(solution, X)   
@show isapprox(expected, actual; atol=tol)
```

## Note
This project makes little sense as you could use _ModelingToolkit_ with _OptimizationMOI_ to do the same.