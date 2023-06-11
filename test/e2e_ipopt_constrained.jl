using Symbolics
using NLPModelsIpopt: ipopt

@testset "constrained" begin
    @testset "trivial single variable" for center in -2:1.3:2
        @variables x

        objective = (x - center)^2
        constraints = [x^2 ≲ 1]
        tol = 1e-4

        model = SymNLPModels.SymNLPModel(objective, constraints)
        stats = ipopt(model; tol, print_level=0)

        expected = clamp.([center], -1, 1)
        actual = stats.solution
        @test isapprox(expected, actual; atol=tol)
    end
end