using SymNLPModels
using Test

@testset "end to end" begin
    include("e2e_ipopt_unconstrained.jl")
    include("e2e_ipopt_constrained.jl")
end