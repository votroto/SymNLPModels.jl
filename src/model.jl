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
    _hessian_buffer::SparseMatrixCSC{T, Int}
end

SymNLPModel(objective) = SymNLPModel(objective, [])


function SymNLPModel(objective, constraints)
    variables = Sym.get_variables(objective)
    hessian = Sym.sparsehessian(objective, variables)
    jacobian = Sym.sparsejacobian(constraints, variables)
    gradient = Sym.gradient(objective, variables)
    
    build(f) = Sym.build_function(f, variables; expression=false)
    _objective = build(objective)
    _constraints! = build(constraints)[2]
    _jacobian! = build(jacobian)[2]
    _lagrangian_hessian! = build_lagrangian_hessian(objective, constraints)
    _gradient! = build(gradient)[2]

    nnzh = nnz(hessian)
    nnzj = nnz(jacobian)
    ncon = length(constraints)
    nvar = length(variables)
    ucon = zeros(ncon)
    meta = NLPModelMeta(nvar; nnzh, nnzj, ncon, ucon)
    counters = Counters()

    jacobian_buffer = similar(jacobian, Float64)
    hessian_buffer = similar(hessian, Float64)

    SymNLPModel(meta, counters, objective, variables, jacobian, hessian, 
        gradient, _objective, _constraints!, _jacobian!, _lagrangian_hessian!, 
        _gradient!, jacobian_buffer, hessian_buffer)
end
