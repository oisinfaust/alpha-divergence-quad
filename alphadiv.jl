module AlphaDivergenceQuad

export minimax_quad, QuadratureRule, weights, nodes

include("dc.jl")
using .DifferentialCorrection
using Interpolations

struct QuadratureRule
    weights::Vector{<:AbstractFloat}
    nodes::Vector{<:AbstractFloat}
    n::Int
    function QuadratureRule(t::Vector{<:AbstractFloat}, u::Vector{<:AbstractFloat})
        all(u .>= 0) || throw(DomainError(u, "weights must be nonnegative"))
        (all(t .>= 0) & all(t .<= 1)) ||  throw(DomainError(t, "nodes must be in [0,1]"))
        n = length(u)
        length(t) == n || throw(DomainError(nothing, "must be equal number of weights and nodes"))
        new(copy(u), copy(t), n)
    end
end
Base.length(q::QuadratureRule) = q.n
Base.show(io::IO, q::QuadratureRule) = print(io, "Quadrature rule on [0, 1] with $(length(q)) nodes")
Base.iterate(q::QuadratureRule) = q.n >= 1 ? ((q.nodes[1], q.weights[1]), 2) : nothing
Base.iterate(q::QuadratureRule, i) = q.n >= i ? ((q.nodes[i], q.weights[i]), i+1) : nothing
weights(q::QuadratureRule) = q.weights
nodes(q::QuadratureRule) = q.nodes

default_grid() = 10 .^ range(-20, 20, length=1000)

function get_g(α) 
    if α == 1
        g = x -> x != 1 ? (x*log(x) - (x-1))*(1+x) / (x-1)^2 : 1.
    elseif α == 0
        g = x -> x != 1 ? (x - 1 - log(x))*(1+x)/(x-1)^2 : 1.
    else
        g = x -> x != 1 ? (x^α - α*(x-1)-1)*(1+x)/(x-1)^2/α/(α-1) : 1.
    end
    return g
end

function get_nodes_weights(a, b, tk)
    p = get_poles(tk, b)
    numerator = sum(a ./ (p' .- tk), dims=1)[:]
    denom_prime = sum(-b ./ (p' .- tk).^2, dims=1)[:]
    u = numerator ./ denom_prime ./ (1 .- p.^2)
    t = 1 ./ (1 .- p)
    (t, u)
end

function next_tk(xk, interior_extrema, n)
    if length(interior_extrema) < 2
        return n == 0 ? [1.] : exp.(range(-n/2, n/2, length=n))
    end
    interp_linear = linear_interpolation(range(0,1,length=length(interior_extrema)), interior_extrema)
    tkidx = floor.(interp_linear.(range(0,1,length=n)))
    return xk[Int.(tkidx)]
end

function minimax_quad(α, β, degrees, optimizer, xk=default_grid(); verbosity=1, callback=nothing)
    (α > -1) & (α < 2) || throw(DomainError(α, "need α in (-1,2)"))
    (β >= -1) & (β <= 2) || throw(DomainError(β, "need β in [-1,2]"))
    (α <= 0) & (β >= α) && throw(DomainError((α, β), "for α <= 0 need β in [-1,α)"))
    (α >= 1) & (β <= α) && throw(DomainError((α, β), "for α >= 1 need β in (α,2]"))

    g = get_g(α); # function to be approximated
    gk = g.(xk);
    w = get_g(β); # weight function
    wk = w.(xk); 

    Δ = Float64[]
    Q = QuadratureRule[]

    δ = maximum(abs.(gk ./ wk))
    interior_extrema = []
    for degree in degrees
        try 
            verbosity > 0 && @info "Computing best $(degree) node quadrature rule... "
            tk = next_tk(xk, interior_extrema, degree+1)
            (δ, a, b, errs) = differential_correction(xk, tk, gk, optimizer; wk=wk, error_ub=δ, 
                num_extrema=2*length(tk)-1, forced_zeros=[-1.], verbosity=verbosity)
            verbosity > 0 && @info "Finished! Error = $δ"

            if !isnothing(callback)
                callback(xk, errs)
            end

            err = maximum(abs.(errs))
            push!(Δ, err)
            t, u = get_nodes_weights(a, b, tk)
            push!(Q, QuadratureRule(t, u))

            extrema_indices = get_extrema(errs)
            interior_extrema = copy(extrema_indices)
            if abs(abs(errs[1]) - err) < err * DifferentialCorrection.TOL
                interior_extrema = interior_extrema[2:end]
            end
            if abs(abs(errs[end]) - err) < err * DifferentialCorrection.TOL
                interior_extrema = interior_extrema[1:end-1]
            end
        catch e
            @error e.msg
            @warn "terminated early - failed to compute best degree $degree approximation"
            break
        end
    end
    return Δ, Q
end

end