using Symbolics
using NLPModelsIpopt: ipopt

@testset "constrained" begin
    @testset "concave univariate" begin
        vars = @variables x
        center = randn()

        objective = (x - center)^2
        constraints = [x^2 ≲ 1]
        tol = 1e-4

        model = SymNLPModel(objective, constraints)
        stats = ipopt(model; tol, print_level=0)

        expected = clamp.([center], -1, 1)
        actual = stats.solution
        @test isapprox(expected, actual; atol=tol)
    end

    @testset "concave multivariate" begin
        @variables x[1:5]
        X = Symbolics.scalarize(x)

        center = randn(length(X))
        objective = sum((X .- center).^2)
        constraints = X.^2 .≲ 1
        tol = 1e-4

        model = SymNLPModel(objective, constraints)
        stats = ipopt(model; tol, print_level=0)

        expected = clamp.(center, -1, 1)
        solution = SymNLPModels.parse_solution(model, stats.solution)
        actual = SymNLPModels.value(solution, X)   
        @test isapprox(expected, actual; atol=tol)
    end
end