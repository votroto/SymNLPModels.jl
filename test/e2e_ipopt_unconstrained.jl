using Symbolics
using NLPModelsIpopt: ipopt

@testset "unconstrained" begin
    @testset "trivial single variable" begin
        @variables x

        center = randn()
        obj = (x - center)^2

        model = SymNLPModels.SymNLPModel(obj)
        stats = ipopt(model; tol=1e-4, print_level=0)

        expected = [center]
        actual = stats.solution
        @test isapprox(expected, actual; atol=1e-4)
    end
end