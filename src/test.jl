using Revise

include("SymNLPModels.jl")

using Symbolics

using NLPModelsIpopt: ipopt

@variables x[1:3] y[1:3]

obj = (x[1]-10)^2 + (y[2]+1)^2
constr1 = x[1]^2 - 1
constr2 = y[2]^2 - 1

model = SymNLPModels.SymNLPModel(obj)
stats = ipopt(model; print_level=0)
@show stats.solution


constr_model = SymNLPModels.SymNLPModel(obj, [constr1, constr2])
constr_stats = ipopt(constr_model; print_level=0)
@show constr_stats.solution
;