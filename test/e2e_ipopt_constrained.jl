using Symbolics
using NLPModelsIpopt: ipopt

@testset "constrained" begin
    @testset "trivial single variable" begin
        @variables x

        center = randn()
        obj = (x - center)^2
        c1 = x^2 - 1

        model = SymNLPModels.SymNLPModel(obj, [c1])
        stats = ipopt(model; tol=1e-4, print_level=0)

        expected = clamp.([center], -1, 1)
        actual = stats.solution
        @test isapprox(expected, actual; atol=1e-4)
    end
end