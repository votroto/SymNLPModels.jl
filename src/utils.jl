using LinearAlgebra: mul!
import Symbolics as Sym

function build!_lagrangian_hessian(objective, constraints)
    variables = Sym.get_variables(objective)
    hess(f) = Sym.sparsehessian(f, variables) 
    build!(f) = Sym.build_function(f, variables; expression=false)[2]
    
    constraint_hessians! = map(build! ∘ hess, constraints)
    objective_hessian = hess(objective)
    objective_hessian! = build!(objective_hessian)

    sa_buf = similar(objective_hessian, Float64)
    vec_buf = similar(sa_buf.nzval, Float64)

    function lag!(v, x, y; obj_weight=1)
        objective_hessian!(sa_buf, x)
        mul!(vec_buf, sa_buf.nzval, obj_weight)
        v .= vec_buf

        for (i, hc!) in enumerate(constraint_hessians!)
            hc!(sa_buf, x)
            mul!(vec_buf, sa_buf.nzval, y[i])
            v .+= vec_buf
        end
    end

    lag!
end

function sa_structure!(rows, cols, sa)
    rows .= sa.rowval
    for i = 1:size(sa, 2)
        for j = sa.colptr[i]:(sa.colptr[i+1]-1)
            cols[j] = i
        end
    end

    rows, cols
end

function inequality_to_expr(ineq::Sym.Inequality)
    lhs, rhs, op = ineq.lhs, ineq.rhs, ineq.relational_op
    (op == Sym.geq) ? -lhs + rhs : lhs - rhs
end

function parse_solution(model, solution)
    Dict(model.variables .=> solution)
end

function value(solution_dict::AbstractDict, keys::AbstractArray)
    map(k -> get(solution_dict, k, NaN), keys)
end

function value(solution_dict::AbstractDict, key)
    get(solution_dict, key, NaN)
end

_to_inequality(ineq::Sym.Inequality) = ineq
_to_inequality(ineq) = Sym.Inequality(ineq)