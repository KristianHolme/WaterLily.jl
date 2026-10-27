# Full-coarsening (2× in every direction) maps between fine and coarse cells.
@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)

# Semi-coarsening up and down. `c[j]` flags which directions are coarsened (2×) between two levels
@inline up(I::CartesianIndex{n},c::NTuple{n,Bool}) where n = CartesianIndices(ntuple(j-> c[j] ? (2I.I[j]-2:2I.I[j]-1) : (I.I[j]:I.I[j]), n))
@inline down(I::CartesianIndex{n},c::NTuple{n,Bool}) where n = CI(ntuple(j-> c[j] ? (I.I[j]+2)÷2 : I.I[j], n))
# Fine faces (normal i) composing a coarse face, given the coarsening mask `c`.
@inline upL(I::CartesianIndex{n},i,c::NTuple{n,Bool}) where n =
    CartesianIndices(ntuple(j-> j==i ? (c[i] ? (2I.I[i]-2:2I.I[i]-2) : (I.I[i]:I.I[i])) :
                                       (c[j] ? (2I.I[j]-2:2I.I[j]-1) : (I.I[j]:I.I[j])), n))

@fastmath @inline function restrict(I::CartesianIndex,b,c)
    s = zero(eltype(b))
    for J ∈ up(I,c)
        s += @inbounds(b[J])
    end
    return s
end
@fastmath @inline function restrictL(I::CartesianIndex,i,b,c)
    s = zero(eltype(b))
    for J ∈ upL(I,i,c)
        s += @inbounds(b[J,i])
    end
    return c[i] ? s/2 : s  # halve only if the face-normal direction is coarsened
end

# coarsening mask: coarsen every direction that is still divisible
@inline coarsen_mask(N::NTuple) = map(divisible,N)
# mask used to build level `coarse` from `fine` (recovered from their sizes)
@inline coarsen_mask(fine,coarse) = ntuple(j-> size(coarse,j)<size(fine,j), ndims(fine))

function restrictML(b::Poisson)
    N,n = size_u(b.L)
    c = coarsen_mask(N)
    Na = ntuple(j-> c[j] ? 1+N[j]÷2 : N[j], n)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,c,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
function restrictL!(a::AbstractArray{T,M},b,c;perdir=()) where {T,M}
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b,c) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zero(SVector{M-1,T}),false,perdir)  # correct μ₀ @ boundaries
end
restrict!(a,b,c) = @inside a[I] = restrict(I,b,c)
prolongate!(a,b,c) = @inside a[I] = b[down(I,c)]

@inline divisible(N::Integer) = mod(N,2)==0 && N>4
# keep coarsening while ANY direction is still divisible (semi-coarsening)
@inline divisible(l::Poisson) = any(size(l.x) .|> divisible)
"""
    MultiLevelPoisson{T::Float, S<:AbstractArray{T,D}, V<:AbstractArray{T,D+1}}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The main field is `levels`, a vector of nested `Poisson` systems from fine to coarse.
"""
struct MultiLevelPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    x::S
    L::V
    z::S
    levels :: Vector{Poisson{T,S,V}}
    n :: Vector{Int16}
    perdir :: NTuple # direction of periodic boundary condition
    function MultiLevelPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};maxlevels=10,perdir=()) where T
        levels = Poisson[Poisson(x,L,z;perdir)]
        while divisible(levels[end]) && length(levels) <= maxlevels
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        new{T,typeof(x),typeof(L)}(x,L,z,levels,[],perdir)
    end
end

function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        c = coarsen_mask(ml.levels[l-1].x,ml.levels[l].x)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,c,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])
    end
end

function Vcycle!(ml::MultiLevelPoisson;l=1,ω=1,fix=nothing)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    c = coarsen_mask(fine.x,coarse.x)
    isgpu(fine.x) || return Vcycle_cpu!(ml,fine,coarse,c;l,ω)
    r,ϵ,x,L,D,iD,cr,cx = fine.r,fine.ϵ,fine.x,fine.L,fine.D,fine.iD,coarse.r,coarse.x
    N,per = size(x),perarg(x,fine.perdir)
    # set up coarse level: Jacobi! with ϵ=r⊙iD evaluated on the fly, and its residual stored in ϵ. With `fix=s`,
    # the residual is first corrected to `iD==0 ? 0 : r-s` (as in `residual!`) and its norms are returned
    norms = if isnothing(fix)
        @loop (ϵ[I] = r[I]-mult(I,L,D,jacobi,r,iD,N,per);
               x[I] += jacobi(I,r,iD,N,per)) over I ∈ inside(x)
        nothing
    else
        loop_reduce(_jacobi_fix,(+,max),inside(x),fine.z,ϵ,r,x,L,D,iD,fix,N,per)
    end
    @loop (cr[I] = restrict(I,ϵ,c); cx[I] = zero(eltype(cx))) over I ∈ inside(cx)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1; ω)
    smooth!(coarse;ω)
    # correct fine: increment! with ϵ=prolongate!(coarse.x) evaluated on the fly, from the residual in ϵ
    @loop (r[I] = ϵ[I]-ω*mult(I,L,D,prolong,cx,c,N,per);
           x[I] += ω*prolong(I,cx,c,N,per)) over I ∈ inside(x)
    return norms
end
# The V-cycle with stored increments: on CPUs, evaluating them on the fly costs more than the passes it saves
function Vcycle_cpu!(ml,fine,coarse,c;l,ω)
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r,c)
    fill!(coarse.x,0)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1; ω)
    smooth!(coarse;ω)
    # correct fine
    prolongate!(fine.ϵ,coarse.x,c)
    increment!(fine; ω)
end
# Jacobi! and prolongate! increments at J, including the periodic copies in the ghosts that perBC! would fill
@inline jacobi(J,r,iD,N,per) = (J = wrap(J,N,per); @inbounds r[J]*iD[J])
@inline prolong(J,cx,c,N,per) = @inbounds cx[down(wrap(J,N,per),c)]
# `jacobi` of the corrected residual, which is zero where iD==0
@inline jacobi_fix(J,r,iD,s,N,per) = (J = wrap(J,N,per); @inbounds (r[J]-s)*iD[J])
@fastmath @inline function _jacobi_fix(I,ϵ,r,x,L,D,iD,s,N,per)
    v = iszero(@inbounds iD[I]) ? zero(s) : @inbounds(r[I])-s
    @inbounds ϵ[I] = v-mult(I,L,D,jacobi_fix,r,iD,s,N,per)
    @inbounds x[I] += v*iD[I]
    return abs(v),abs(v)
end

mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson,x) = residual!(ml.levels[1],x)
diagonal(ml::MultiLevelPoisson) = diagonal(ml.levels[1])

const smooth! = GaussSeidelRB!

function solver!(ml::MultiLevelPoisson{T};tol=default_tol,itmx=32,src=nothing) where T
    p = ml.levels[1]
    r₁tol = l1n_tol(p, tol); r∞tol = tol
    # on GPUs with the source evaluated on the fly, the first fine Jacobi of the first V-cycle
    # corrects the residual and returns its norms
    fix = isgpu(p.x) && src isa Tuple && itmx ≥ 1 ? residual_mean!(p,src) : nothing
    ω = T(1); nᵖ=0
    if isnothing(fix)
        r₁,r∞ = residual_L₁∞!(p,src)
        @log ", $nᵖ, $r∞, $r₁, $ω\n"
    end
    while nᵖ<itmx
        if nᵖ==0 && !isnothing(fix)
            r₁,r∞ = Vcycle!(ml; ω, fix)
            @log ", $nᵖ, $r∞, $r₁, $ω\n"
        else
            Vcycle!(ml; ω)
        end
        rnew,r∞ = smooth!(p; ω, norms=true); nᵖ+=1
        @log ", $nᵖ, $r∞, $rnew, $ω\n"
        if     rnew ≥ r₁
            ω = max(0.2, 0.9ω) |> T
        elseif rnew < r₁
            ω = min(1.0, 1.02ω) |> T
        end
        r₁ = rnew
        (r₁<r₁tol && r∞<r∞tol) && break
    end
    perBC!(p.x,p.perdir)
    push!(ml.n,nᵖ);
end