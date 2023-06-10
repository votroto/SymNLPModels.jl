import NLPModels: jac_structure!, jac_coord!, cons!

function NLPModels.jac_structure!(nlp::SymNLPModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T}
) where {T}
    sa_structure!(rows, cols, nlp.jacobian)
end

function NLPModels.jac_coord!(nlp::SymNLPModel,
    x::AbstractVector,
    jvals::AbstractVector
)
    nlp._jacobian!(nlp._jacobian_buffer, x)
    jvals .= nlp._jacobian_buffer.nzval

    jvals
end

function NLPModels.cons!(
    nlp::SymNLPModel,
    x::AbstractVector,
    c::AbstractVector
)
    nlp._constraints!(c, x)
    c
end

function NLPModels.hess_coord!(
    nlp::SymNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hvals::AbstractVector;
    obj_weight=1
)
    nlp._lagrangian_hessian!(hvals, x, y; obj_weight)

    hvals
end