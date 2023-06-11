using Symbolics
using NLPModelsIpopt: ipopt

@testset "unconstrained" begin
    @testset "concave univariate" begin
        @variables x
        center = randn()

        objective = (x - center)^2
        tol = 1e-4

        model = SymNLPModel(objective)
        stats = ipopt(model; tol, print_level=0)

        expected = [center]
        actual = stats.solution
        @test isapprox(expected, actual; atol=tol)
    end
end