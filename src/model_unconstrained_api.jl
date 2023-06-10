using NLPModels

import NLPModels: hess_structure!, hess_coord!, grad!, obj

function NLPModels.hess_structure!(
    nlp::SymNLPModel, 
    rows::AbstractVector, 
    cols::AbstractVector
)
    sa_structure!(rows, cols, nlp.hessian)
end

function NLPModels.grad!(
    nlp::SymNLPModel, 
    x::AbstractVector, 
    g::AbstractVector
)
    nlp._gradient!(g, x)
    g
end

function NLPModels.obj(
    nlp::SymNLPModel, 
    x::AbstractVector
)
    nlp._objective(x)
end

function NLPModels.hess_coord!(
    nlp::SymNLPModel,
    x::AbstractVector,
    hvals::AbstractVector;
    obj_weight=1
)
    nlp._lagrangian_hessian!(hvals, x, []; obj_weight)

    hvals
end