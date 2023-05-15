module DifferentialCorrection 

export differential_correction, get_extrema, get_poles

using LinearAlgebra, JuMP
using Quadmath
import GenericSchur

const TOL = 1e-4

function get_ϕ(xk, tk)
    ϕ = 1 ./ (xk .- tk')
    ϕ[ϕ .=== -0.0] .= -1.
    ϕ[ϕ .=== 0.0] .= 1.
    ϕ = ϕ .* prod(sign.(ϕ), dims=2)
    mask = isinf.(ϕ)
    sinf = sign.(ϕ[mask])
    ϕ = ϕ ./ maximum(abs.(ϕ), dims=2)
    ϕ[mask] = sinf
    ϕ
end

function dc_lp(ϕ::Matrix{Float64}, F::Vector{Float64}, W::Vector{Float64}, δ_prev, ϕβ_prev::Vector{Float64}, optimizer; forced_zeros=[])
    m = size(ϕ, 2)
    model = Model(optimizer);
    set_optimizer_attribute(model, "QUIET", true);
    set_optimizer_attribute(model, "INTPNT_SOLVE_FORM", 2)
    @variable(model, ϵ);
    @variable(model, α[1:m]);
    @variable(model, β[1:m]);
    ϕβ = ϕ*β
    ϕα = ϕ*α
    ϕβf = ϕβ .* F;
    ϕβw = ϕβ_prev .* W;
    rhs = ϵ .+ δ_prev * (ϕβ ./ ϕβ_prev)
    lhs = (ϕα - ϕβf) ./ ϕβw 
    @constraint(model, crefu, lhs .<= rhs);
    @constraint(model, crefl, -lhs .<= rhs);
    @constraint(model, β .<= 1);
    @constraint(model, β .>= -1);
    @constraint(model, ϕβ .>= 0);
    for ϕ0 in forced_zeros
        @constraint(model, ϕ0[1,:]' * α == 0);
    end
    @objective(model, Min, ϵ);
    optimize!(model)
    Int(termination_status(model))==1, value.(α), value.(β), -value(ϵ)
end

function differential_correction(xk, tk, fk, optimizer; wk=fk, error_ub=1, 
                    num_extrema=2*length(tk)-1, forced_zeros=[], max_iter=100, verbosity=1)
    ϕ = get_ϕ(xk, tk)
    ϕβ_prev = ones(eltype(ϕ), size(ϕ, 1))
    ϕ0s = map(x -> get_ϕ([x], tk), forced_zeros)
    δ_prev = error_ub
    δ = Nothing
    (α, β, errs) = (Nothing, Nothing, Nothing)
    iter = 0
    while iter < max_iter
        (flag, α, β, ϵ) = dc_lp(ϕ, fk, wk, δ_prev, ϕβ_prev, optimizer; forced_zeros=ϕ0s)
        flag || error("solver failed")
        ϵ > 0 || @warn "numerical difficulty: LP iteration failed to increase accuracy"

        rk = (ϕ*α)./(ϕ*β)
        errs = (rk - fk) ./ wk
        ϕβ_prev = ϕ*β

        δ = maximum(abs.(errs))
        
        extrema_indices = get_extrema(errs)
        δ_lb = error_lower_bound(errs[extrema_indices], num_extrema)

        current_accuracy = (δ - δ_lb) / δ
        verbosity > 1 && @info "current minimax error = $current_accuracy" 

        if current_accuracy < TOL break end
        δ_prev = δ
        iter += 1
    end
    iter >= max_iter && error("max iterations reached")
    return (δ, α, β, errs)
end

function get_extrema(errs)
    extrema_idxs = [1]
    current_extremum = 0
    for (i, err) in enumerate(errs)
        if err * current_extremum < 0 
            push!(extrema_idxs, i)
            current_extremum = err
        elseif abs(err) > abs(current_extremum)
            extrema_idxs[end] = i
            current_extremum = err
        end
    end
    return extrema_idxs
end

function error_lower_bound(extr, num_expected)
    if length(extr) < num_expected
        return 0
    elseif length(extr) == num_expected
        return minimum(abs.(extr))
    end
    idx = 1
    if length(extr) == num_expected + 1
        if abs(extr[1]) > abs(extr[end]) 
            idx = length(extr) 
        end
    else
        idx = findmin(extr)[2]
    end
    if idx == 1
        return error_lower_bound(extr[2:end], num_expected)
    elseif idx == length(extr)
        return error_lower_bound(extr[1:end-1], num_expected)
    else
        if abs(errs[idx+1]) > abs(errs[idx-1])
            idx -= 1
        end
        return error_lower_bound([errs[1:idx-1]; errs[idx+2:end]], num_expected)
    end
end

function get_poles(tk, β)
    A = Float128.([0 β'; ones(length(tk), 1) diagm(tk)])
    B = Float128.(diagm([0; ones(length(tk))]))
    S, T = GenericSchur.schur(Float128.(A) .+ 0im, B)
    poles = diag(S ./ T)
    poles = real.(poles)
    poles = poles[.~isinf.(poles)]
    poles = poles[.~isnan.(poles)]
    Float64.(poles)
end

end