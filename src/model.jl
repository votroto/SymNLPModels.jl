import Symbolics as Sym
using SparseArrays: SparseMatrixCSC, nnz
using NLPModels

mutable struct SymNLPModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters

    objective::Sym.Num
    variables::Vector
    jacobian::SparseMatrixCSC{Sym.Num, Int64}
    hessian::SparseMatrixCSC{Sym.Num, Int64}
    gradient::Vector{Sym.Num}

    _objective::Function
    _constraints!::Function
    _jacobian!::Function
    _lagrangian_hessian!::Function
    _gradient!::Function

    _jacobian_buffer::SparseMatrixCSC{T, Int}
end

"""
    SymNLPModel(
    objective::Sym.Num, 
    inequalities::AbstractVector{Sym.Inequality};
    variables = Sym.get_variables(objective),
    initial = randn(length(variables))
)

Nonlinear symbolic mathematical model with an objective and variables 
constrained by inequalities.
"""
function SymNLPModel(
    objective::Sym.Num, 
    inequalities::AbstractVector{Sym.Inequality};
    variables = Sym.get_variables(objective),
    initial = randn(length(variables))
)
    build(f) = Sym.build_function(f, variables; expression=false)
    
    constraints = map(inequality_to_expr, inequalities)
    hessian = Sym.sparsehessian(objective, variables)
    jacobian = Sym.sparsejacobian(constraints, variables)
    gradient = Sym.gradient(objective, variables)

    _objective = build(objective)
    _lagrangian_hessian! = build!_lagrangian_hessian(objective, constraints)
    _, _constraints! = build(constraints)
    _, _jacobian! = build(jacobian)
    _, _gradient! = build(gradient)

    nnzh = nnz(hessian)
    nnzj = nnz(jacobian)
    ncon = length(constraints)
    nvar = length(variables)
    ucon = zeros(ncon)
    meta = NLPModelMeta(nvar; x0=initial, nnzh, nnzj, ncon, ucon)
    counters = Counters()

    jacobian_buffer = similar(jacobian, Float64)

    SymNLPModel(meta, counters, objective, variables, jacobian, hessian, 
        gradient, _objective, _constraints!, _jacobian!, _lagrangian_hessian!, 
        _gradient!, jacobian_buffer)
end

function SymNLPModel(objective, inequalities; kwargs...)
    _inequalities = map(_to_inequality, inequalities)
    SymNLPModel(Sym.Num(objective), _inequalities; kwargs...)
end

function SymNLPModel(objective; kwargs...) 
    SymNLPModel(Sym.Num(objective), Sym.Inequality[]; kwargs...)
end
