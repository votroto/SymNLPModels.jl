using LinearAlgebra: mul!

function build_lagrangian_hessian(objective, constraints)
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