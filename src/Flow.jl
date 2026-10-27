@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])/2
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)
@fastmath cds(u,c,d) = (c+d)/2

@inline ϕu(a,I,f,u,λ) = (p = u>0; @inbounds u*λ(ifelse(p,f[I-2δ(a,I)],f[I+δ(a,I)]),ifelse(p,f[I-δ(a,I)],f[I]),ifelse(p,f[I],f[I-δ(a,I)])))
@inline ϕuP(a,Ip,I,f,u,λ) = (p = u>0; @inbounds u*λ(ifelse(p,f[Ip],f[I+δ(a,I)]),ifelse(p,f[I-δ(a,I)],f[I]),ifelse(p,f[I],f[I-δ(a,I)])))
@inline ϕuL(a,I,f,u,λ) = @inbounds u*ifelse(u>0,ϕ(a,I,f),λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)]))
@inline ϕuR(a,I,f,u,λ) = @inbounds u*ifelse(u<0,ϕ(a,I,f),λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]))

@fastmath @inline function div(I::CartesianIndex{m},u) where {m}
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
end
@fastmath @inline function μddn(I::CartesianIndex{np1},μ,f) where np1
    s = zero(eltype(f))
    for j ∈ 1:np1-1
        s+= @inbounds μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])
    end
    return s/2
end
@fastmath median(a,b,c) = max(min(a,b),min(max(a,b),c))

function conv_diff!(r,u,Φ,λ::F;ν=0.1,perdir=()) where {F}
    fill!(r,zero(eltype(r)))
    N,n = size_u(u)
    for j ∈ 1:n
        for i ∈ 1:n # inner cells
            @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν*∂(j,CI(I,i),u);
                   r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
            @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        end
        boundaries!(r,u,ν,j,N,λ,Val{j in perdir}())
    end
end

# Fluxes on domain boundaries normal to `j`
function boundaries!(r,u,ν,j,N,λ,per)
    R = CartesianIndices((ntuple(k -> k==j ? (2:2) : (2:N[k]), length(N))...,1:length(N)))
    Nj = N[j]
    @loop boundary!(r,Ii,u,ν,j,Nj,λ,per) over Ii ∈ face(R,j)
end
# Neumann BC
@fastmath @inline function boundary!(r,Ii,u,ν,j,N,λ,::Val{false})
    I,i,IN = Base.front(Ii),last(Ii),CIj(j,Ii,N)
    @inbounds r[Ii] += ϕuL(j,Ii,u,ϕ(i,CI(I,j),u),λ) - ν*∂(j,Ii,u)
    @inbounds r[CIj(j,Ii,N-1)] -= ϕuR(j,IN,u,ϕ(i,CI(Base.front(IN),j),u),λ) - ν*∂(j,IN,u)
end
# Periodic BC
@fastmath @inline function boundary!(r,Ii,u,ν,j,N,λ,::Val{true})
    I,i = Base.front(Ii),last(Ii)
    Φ = ϕuP(j,CIj(j,Ii,N-2),Ii,u,ϕ(i,CI(I,j),u),λ) - ν*∂(j,Ii,u)
    @inbounds r[Ii] += Φ; @inbounds r[CIj(j,Ii,N-1)] -= Φ
end

"""
    accelerate!(r,t,g,U)

Accounts for applied and reference-frame acceleration using `rᵢ += g(i,x,t)+dU(i,x,t)/dt`
"""
accelerate!(r,t,::Nothing,::Union{Nothing,Tuple}) = nothing
accelerate!(r,t,f::Function) = @loop r[I,i] += f(i,loc(i,I,eltype(r)),t) over I ∈ CartesianIndices(Base.front(size(r))), i ∈ 1:last(size(r))
accelerate!(r,t,g::Function,::Union{Nothing,Tuple}) = accelerate!(r,t,g)
accelerate!(r,t,::Nothing,U::Function) = accelerate!(r,t,(i,x,t)->derivative(τ->U(i,x,τ),t))
accelerate!(r,t,g::Function,U::Function) = accelerate!(r,t,(i,x,t)->g(i,x,t)+derivative(τ->U(i,x,τ),t))
# the function `(i,x,t)->gᵢ` that accelerate! adds, or nothing
acceleration(::Nothing,::Union{Nothing,Tuple}) = nothing
acceleration(g::Function,::Union{Nothing,Tuple}) = g
acceleration(::Nothing,U::Function) = (i,x,t)->derivative(τ->U(i,x,τ),t)
acceleration(g::Function,U::Function) = (i,x,t)->g(i,x,t)+derivative(τ->U(i,x,τ),t)
@inline gᵢ(::Nothing,Ii,t) = zero(t)
@inline gᵢ(g,Ii,t) = g(last(Ii),loc(Ii,typeof(t)),t)

# Flux of uᵢ through the lower j-face of the cell, as `conv_diff!` stores it in Φ: the interior flux reads
# the far upwind value only, and the boundary fluxes are those of `boundaries!`
@fastmath @inline function Φ(j,Ii::CartesianIndex,u,ν,λ::F,N,per) where {F}
    I,i = Base.front(Ii),last(Ii)
    2<I[j]<N[j] && return Φᵢ(j,Ii,u,ν,λ)
    per[j] && I[j]==N[j] && (Ii = CIj(j,Ii,2); I = Base.front(Ii)) # the upper periodic face is the lower one
    uf = ϕ(i,CI(I,j),u)
    I[j]==N[j] && return ϕuR(j,Ii,u,uf,λ)-ν*∂(j,Ii,u)
    return (per[j] ? ϕuP(j,CIj(j,Ii,N[j]-2),Ii,u,uf,λ) : ϕuL(j,Ii,u,uf,λ))-ν*∂(j,Ii,u)
end
# `Φ` of an interior face, 2<I[j]<N[j]
@fastmath @inline function Φᵢ(j,Ii::CartesianIndex,u,ν,λ::F) where {F}
    uf = ϕ(last(Ii),CI(Base.front(Ii),j),u); c,b = @inbounds(u[Ii]),@inbounds(u[Ii-δ(j,Ii)]); p = uf>0
    return uf*λ(@inbounds(u[ifelse(p,Ii-2δ(j,Ii),Ii+δ(j,Ii))]),ifelse(p,b,c),ifelse(p,c,b))-ν*(c-b)
end
# `r[Ii]` of conv_diff!: the net flux into the cell, also in the ghosts (zero in the lower ones, and without
# the faces normal to j in the upper ones of j)
@inline function rhs(Ii::CartesianIndex{m},u,ν,λ::F,N,per) where {m,F}
    z = zero(eltype(u))
    any(ntuple(@inline(j->Ii[j]==1),Val(m-1))) && return z
    sum(ntuple(@inline(j->Ii[j]<N[j] ? Φ(j,Ii,u,ν,λ,N,per)-Φ(j,Ii+δ(j,Ii),u,ν,λ,N,per) : z),Val(m-1)))
end
# `bdim` of a fluid face (μ₀=1, μ₁=0) whose fluxes normal to j>1 are interior, 3≤I[j]≤N[j]-2
@fastmath @inline function fluid(Ii::CartesianIndex{m},u,u⁰,Δt,ν,λ::F,N,per,g,t) where {m,F}
    r = Φ(1,Ii,u,ν,λ,N,per)-Φ(1,Ii+δ(1,Ii),u,ν,λ,N,per)
    r += sum(ntuple(@inline(j->Φᵢ(j+1,Ii,u,ν,λ)-Φᵢ(j+1,Ii+δ(j+1,Ii),u,ν,λ)),Val(m-2)))
    return @inbounds(u⁰[Ii])+Δt*(r+gᵢ(g,Ii,t))
end
# `f[Ii]` after the first loop of BDIM!: u⁰+Δt(r+g)-V
@fastmath @inline f̃(Ii,u,u⁰,V,Δt,ν,λ,N,per,g,t) = @inbounds(u⁰[Ii])+Δt*(rhs(Ii,u,ν,λ,N,per)+gᵢ(g,Ii,t))-@inbounds(V[Ii])

"""
    apply!(f, c)

Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c` or
a function `f(x)` to the center of a uniform array `c`.
"""
apply!(f,c) = hasmethod(f,Tuple{Int,CartesianIndex}) ? applyV!(f,c) : applyS!(f,c)
applyV!(f,c) = @loop c[I,i] = f(i,loc(i,I,eltype(c))) over I ∈ CartesianIndices(Base.front(size(c))), i ∈ 1:last(size(c))
applyS!(f,c) = @loop c[I] = f(loc(0,I,eltype(c))) over I ∈ CartesianIndices(c)

ic_function(uBC::Function) = (i,x)->uBC(i,x,0)
ic_function(uBC) = (i,x)->uBC[i]

"""
    ic_kwarg(u0, uλ)

Resolve the initial-condition keyword argument. `uλ` is the deprecated name for `u0`:
if it is supplied a deprecation warning is emitted and its value is used only when `u0`
is not given. Returns the effective initial condition (possibly `nothing`).

Remove this function in WL 2.0
"""
function ic_kwarg(u0, uλ)
    isnothing(uλ) && return u0
    @warn "The `uλ` keyword argument is deprecated and will be removed in a future release. Use `u0` instead." maxlog=1
    isnothing(u0) ? uλ : u0
end

abstract type AbstractFlow{D,T} end
# The inside faces where the BDIM is not the identity: `band` with μ₀≠1 and μ₁=0, and `shell` with μ₁≠0.
# `measure!` marks them stale, and the fused `BDIM!` finds them again when it needs them.
mutable struct BDIMFaces{A,C}
    band :: A
    shell :: A
    n :: C # the list lengths, counted on the device
    valid :: Bool
end
BDIMFaces(μ₀) = (e = similar(μ₀,CartesianIndex{ndims(μ₀)},0); BDIMFaces(e,copy(e),similar(μ₀,Int32,2),false))

"""
    Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}, Lf, Bf}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `D`)
and the velocity vector field `u` (an array of dimension `D+1`).
"""
struct Flow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}, Lf, Bf} <: AbstractFlow{D,T}
    # Fluid fields
    u :: Vf # velocity vector field
    u⁰:: Vf # previous velocity
    f :: Vf # force vector
    p :: Sf # pressure scalar field
    σ :: Sf # divergence scalar
    # BDIM fields
    V :: Vf # body velocity vector
    μ₀:: Vf # zeroth-moment vector
    μ₁:: Tf # first-moment tensor field
    # Non-fields
    uBC :: Union{NTuple{D,Number},Function} # domain boundary values/function
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    g :: Union{Function,Nothing} # acceleration field funciton
    exitBC :: Bool # Convection exit
    perdir :: NTuple # tuple of periodic direction
    λ :: Lf # convective scheme λ(u,c,d); a type parameter so conv_diff!'s kernel specializes on it
    faces :: Bf # faces where the BDIM is not the identity, for the fused `BDIM!`
    function Flow(N::NTuple{D}, uBC; mem=Array, Δt=0.25, ν=0., g=nothing,
            u0=nothing, uλ=nothing, perdir=(), exitBC=false, λ=quick, T=Float32) where D
        u0 = ic_kwarg(u0, uλ) # to be removed in v2.0
        uBC isa Tuple && (uBC = T.(uBC)) # same type for all components: `uBC[i]` must be type-stable in GPU kernels
        Ng = N .+ 2
        Nd = (Ng..., D)
        isnothing(u0) && (u0 = ic_function(uBC))
        u = Array{T}(undef, Nd...) |> mem
        isa(u0, Function) ? apply!(u0, u) : apply!(ic_function(T.(u0)), u)
        BC!(u,uBC,exitBC,perdir); exitBC!(u,u,zero(T))
        u⁰ = copy(u)
        fv, p, σ = zeros(T, Nd) |> mem, zeros(T, Ng) |> mem, zeros(T, Ng) |> mem
        V, μ₀, μ₁ = zeros(T, Nd) |> mem, ones(T, Nd) |> mem, zeros(T, Ng..., D, D) |> mem
        BC!(μ₀,ntuple(zero, D),false,perdir)
        faces = BDIMFaces(μ₀)
        new{D,T,typeof(p),typeof(u),typeof(μ₁),typeof(λ),typeof(faces)}(u,u⁰,fv,p,σ,V,μ₀,μ₁,uBC,T[Δt],T(ν),g,exitBC,perdir,λ,faces)
    end
    # field-wise constructor: lets `setproperties(flow; uBC)` swap non-field values without reallocating the fields
    function Flow(u,u⁰,f,p,σ,V,μ₀,μ₁,uBC,Δt,ν,g,exitBC,perdir,λ,faces=BDIMFaces(μ₀))
        uBC isa Tuple && (uBC = eltype(p).(uBC)) # same type-stability requirement as above
        new{ndims(p),eltype(p),typeof(p),typeof(u),typeof(μ₁),typeof(λ),typeof(faces)}(u,u⁰,f,p,σ,V,μ₀,μ₁,uBC,Δt,ν,g,exitBC,perdir,λ,faces)
    end
end

stale!(a::AbstractFlow) = nothing
stale!(a::Flow) = (a.faces.valid = false; nothing)
# The lists are compacted with atomic counters: one pass counts the faces, a second stores them (in any order)
function bdimfaces(a::Flow)
    f = a.faces; f.valid && return f
    R,μ₀,μ₁,n = inside_u(size(a.p)),a.μ₀,a.μ₁,f.n
    fill!(n,0); @loop countface!(n,Ii,μ₀,μ₁) over Ii ∈ R
    nb,ns = Array(n)
    length(f.band)==nb || (f.band = similar(f.band,nb))
    length(f.shell)==ns || (f.shell = similar(f.shell,ns))
    band,shell = f.band,f.shell
    fill!(n,0); @loop addface!(n,band,shell,Ii,μ₀,μ₁) over Ii ∈ R
    f.valid = true; return f
end
@inline function countface!(n,Ii,μ₀,μ₁)
    @inbounds μ₀[Ii]==1 && return
    @atomic n[isshell(Ii,μ₁) ? 2 : 1] += one(eltype(n))
    return
end
@inline function addface!(n,band,shell,Ii,μ₀,μ₁)
    @inbounds μ₀[Ii]==1 && return
    if isshell(Ii,μ₁)
        k = @atomic n[2] += one(eltype(n)); @inbounds shell[k] = Ii
    else
        k = @atomic n[1] += one(eltype(n)); @inbounds band[k] = Ii
    end
    return
end
@inline isshell(Ii::CartesianIndex{m},μ₁) where m = any(ntuple(@inline(j->!iszero(@inbounds μ₁[Ii,j])),Val(m-1)))

"""
    mom_step!(a::AbstractFlow,b::AbstractPoisson;udf=nothing,kwargs...)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
`kwargs` are passed to `udf` and `poisson_kwargs` to `solver!`.
"""
@fastmath function mom_step!(a::AbstractFlow,b::AbstractPoisson;udf=nothing,poisson_kwargs=(;),kwargs...)
    copyto!(a.u⁰,a.u); fused(a,udf) || fill!(a.u,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    mom_predict!(a,t₀,t₁;udf,kwargs...)
    mom_project!(a,b,1,t₁;poisson_kwargs...)
    # corrector u → u¹
    @log "c"
    mom_correct!(a,t₁;udf,kwargs...)
    mom_project!(a,b,2,t₁;poisson_kwargs...)
    push!(a.Δt,CFL(a))
end

"""
    time(a::AbstractFlow)

Current flow time.
"""
time(a::AbstractFlow) = sum(@view(a.Δt[1:end-1]))

function BDIM!(a::AbstractFlow,s=1)
    dt = a.Δt[end]
    @loop a.f[I,i] = a.u⁰[I,i]+dt*a.f[I,i]-a.V[I,i] over I ∈ CartesianIndices(a.p), i ∈ 1:ndims(a.p)
    @loop a.u[I,i] = (a.u[I,i]+(μddn(CI(I,i),a.μ₁,a.f)+a.V[I,i]+a.μ₀[I,i]*a.f[I,i]))/s over I ∈ inside(a.p), i ∈ 1:ndims(a.p)
end

"""
    mom_predict!(a::AbstractFlow, t₀, t₁; udf=nothing, kwargs...)

Predictor phase of `mom_step!`: advect under `u⁰` with the convective scheme `a.λ` and apply BDIM.
`t₀` and `t₁` are the start and end times of the step; the domain BCs are enforced at `t₁`
by `mom_project!`.
"""
function mom_predict!(a::AbstractFlow, t₀, t₁; udf=nothing, kwargs...)
    fused(a,udf) && return BDIM!(a,a.u,a.u⁰,t₀)
    conv_diff!(a.f,a.u⁰,a.σ,a.λ;ν=a.ν,perdir=a.perdir)
    udf!(a,udf,a.u⁰,t₀; kwargs...) # advect with u⁰
    accelerate!(a.f,t₀,a.g,a.uBC)
    BDIM!(a)
end

"""
    mom_correct!(a::AbstractFlow, t; udf=nothing, kwargs...)

Corrector phase of `mom_step!`: advect under the projected `u` with the convective scheme `a.λ`,
apply BDIM and blend with the trapezoidal weight.
"""
function mom_correct!(a::AbstractFlow, t; udf=nothing, kwargs...)
    if fused(a,udf) # the fluxes read a.u, so it is overwritten after the update
        BDIM!(a,a.f,a.u,t,2)
        return @loop a.u[I,i] = a.f[I,i] over I ∈ inside(a.p), i ∈ 1:ndims(a.p)
    end
    conv_diff!(a.f,a.u,a.σ,a.λ;ν=a.ν,perdir=a.perdir)
    udf!(a,udf,a.u,t; kwargs...) # advect with projected a.u
    accelerate!(a.f,t,a.g,a.uBC)
    BDIM!(a,2)
end
scale_u!(a::AbstractFlow{D,T},s) where {D,T} = (s=T(s); @loop a.u[I,i] *= s over I ∈ inside(a.p), i ∈ 1:D)

"""
    fused(a::AbstractFlow,udf)

Whether the predictor and corrector of `a` use the fused `BDIM!(a,out,u,t,s)`. This is the case for a `Flow`
on a GPU without a `udf`, which needs `r` in `a.f`.
"""
fused(a::Flow,::Nothing) = isgpu(a.u)
fused(a,udf) = false

"""
    BDIM!(a::Flow,out,u,t,s=1)

`out=((s-1)u+v)/s` over the inside faces, where `v` is the BDIM velocity of `conv_diff!(u)`,
`accelerate!` and `BDIM!` at time `t`. `r` is not stored in `a.f`, nor the fluxes in `a.σ`: each flux is
evaluated by both of its cells, which pays where memory traffic dominates. The fluid faces, where the BDIM
is the identity, are updated by a loop without the BDIM terms and with the general fluxes only where they
are needed; two loops over the listed faces of `a.faces` then overwrite those with μ₀≠1. This keeps the
register use of the main loop low, and the BDIM terms are only evaluated where they are not the identity.
"""
function BDIM!(a::Flow{D,T},out,u,t,s=1) where {D,T}
    u⁰,V,μ₀,μ₁,Δt,ν,λ = a.u⁰,a.V,a.μ₀,a.μ₁,a.Δt[end],a.ν,a.λ
    N,per,g,t,s = size(a.p),pertuple(a.perdir,Val(D)),acceleration(a.g,a.uBC),T(t),T(s)
    # fluid faces (μ₀=1, μ₁=0), where the BDIM is the identity. The core faces have interior fluxes
    # normal to j>1, and run over all of j=1 so that the general flux is only needed on the x faces;
    # the one-face shell next to the other domain boundaries uses the general fluxes.
    core = CartesianIndices((2:N[1]-1,map(n->3:n-2,Base.tail(N))...,1:D))
    @loop out[Ii] = ((s-1)*u[Ii]+fluid(Ii,u,u⁰,Δt,ν,λ,N,per,g,t))/s over Ii ∈ core
    for j ∈ 2:D, k ∈ (2,N[j]-1)
        R = CartesianIndices((ntuple(d->d<j ? (d==1 ? (2:N[d]-1) : (3:N[d]-2)) : d==j ? (k:k) : (2:N[d]-1),D)...,1:D))
        @loop out[Ii] = ((s-1)*u[Ii]+@inbounds(u⁰[Ii])+Δt*(rhs(Ii,u,ν,λ,N,per)+gᵢ(g,Ii,t)))/s over Ii ∈ R
    end
    # the faces in or near the body are overwritten with the full BDIM, the shell with its μ₁ terms
    faces = bdimfaces(a); band,shell = faces.band,faces.shell
    @loop bdim_band!(out,band[k],s,u,u⁰,V,μ₀,Δt,ν,λ,N,per,g,t) over k ∈ CartesianIndices(band)
    @loop bdim_shell!(out,shell[k],s,u,u⁰,V,μ₀,μ₁,Δt,ν,λ,N,per,g,t) over k ∈ CartesianIndices(shell)
end
# `out[Ii]` of BDIM! on a `band` face (μ₁=0) and on a `shell` face
@fastmath @inline function bdim_band!(out,Ii,s,u,u⁰,V,μ₀,Δt,ν,λ,N,per,g,t)
    μ = @inbounds μ₀[Ii]
    v = @inbounds(V[Ii])+(iszero(μ) ? μ : μ*f̃(Ii,u,u⁰,V,Δt,ν,λ,N,per,g,t))
    @inbounds out[Ii] = ((s-1)*u[Ii]+v)/s
end
@fastmath @inline bdim_shell!(out,Ii,s,u,args...) = @inbounds out[Ii] = ((s-1)*u[Ii]+bdim(Ii,u,args...))/s
# `μddn(Ii,μ₁,f)+V[Ii]+μ₀[Ii]*f[Ii]` of BDIM!, with f evaluated on the fly. Faces with μ₀=1 have μ₁=0 (it is
# O(ϵ eps^⅔) where μ₀ rounds to 1), so V cancels and only f[Ii] is needed: the BDIM is one read of μ₀ in the fluid
@fastmath @inline function bdim(Ii::CartesianIndex{m},u,u⁰,V,μ₀,μ₁,Δt,ν,λ,N,per,g,t) where {m}
    μ = @inbounds μ₀[Ii]
    μ==1 && return @inbounds(u⁰[Ii])+Δt*(rhs(Ii,u,ν,λ,N,per)+gᵢ(g,Ii,t))
    s = sum(ntuple(@inline(j->μ₁∂f̃(j,Ii,μ₁,u,u⁰,V,Δt,ν,λ,N,per,g,t)),Val(m-1)))
    return s/2+@inbounds(V[Ii])+(iszero(μ) ? μ : μ*f̃(Ii,u,u⁰,V,Δt,ν,λ,N,per,g,t)) # μ₀=0 in the body and on the domain faces
end
@fastmath @inline function μ₁∂f̃(j,Ii,μ₁,args...) # a term of μddn
    μ = @inbounds μ₁[Ii,j]
    iszero(μ) ? zero(μ) : μ*(f̃(Ii+δ(j,Ii),args...)-f̃(Ii-δ(j,Ii),args...))
end

"""
    mom_project!(a::AbstractFlow, b::AbstractPoisson, w::Int, t; kwargs...)

Projection phase of `mom_step!`: solve the pressure Poisson equation, correct
the velocity by `Δt/w·∇p`, and enforce BCs at time `t`. `w=1` in the predictor and `w=2`
in the corrector. On return `a.u` is divergence-free and BC-consistent. `kwargs` are passed to `solver!`.
"""
function mom_project!(a::AbstractFlow, b::AbstractPoisson, w::Int, t; kwargs...)
    BC!(a.u,a.uBC,a.exitBC,a.perdir,t)              # BC MUST be at t₁
    w==1 && a.exitBC && exitBC!(a.u,a.u⁰,a.Δt[end]) # convective exit, predictor (w=1) only
    pressure!(a,b,a.Δt[end]/w; kwargs...)
    BC!(a.u,a.uBC,a.exitBC,a.perdir,t)
end

"""
    pressure!(a::AbstractFlow, b::AbstractPoisson, dt; kwargs...)

Solve for the pressure `b.x` and correct the velocity `a.u` by `dt⋅L∇p`. A general `AbstractPoisson`
solves for `dt⋅p`. WaterLily's `Poisson` and `MultiLevelPoisson` solve for `p` directly, with the
source and the tolerance `tol` divided by `dt`, so `tol` keeps its meaning. `kwargs` are passed to `solver!`.
"""
function pressure!(a::AbstractFlow, b::AbstractPoisson, dt; kwargs...)
    b.x .*= dt                                      # initial pressure solution
    Dp = diagonal(b)                                # Poisson matrix diagonal
    @inside b.z[I] = div(I,a.u)-δv(Dp,I)*div(I,a.V) # flow divergence, not BDIM-ϵ stretching
    solver!(b; kwargs...)
    project!(a.u,b.L,b.x)  # apply solution
    b.x ./= dt             # and unscale to recover pressure
end
function pressure!(a::AbstractFlow, b::Union{Poisson,MultiLevelPoisson}, dt; tol=default_tol, kwargs...)
    idt = inv(dt); tol = tol/primal(dt) # the tolerance is a number, also when dt is a `Dual`
    if isgpu(b.x) # the residual evaluates the source, which saves a pass
        solver!(b; tol, src=(a.u,a.V,idt), kwargs...)
    else
        Dp = diagonal(b)
        @inside b.z[I] = idt*(div(I,a.u)-δv(Dp,I)*div(I,a.V))
        solver!(b; tol, kwargs...)
    end
    project!(a.u,b.L,b.x,dt,regs(b))
end
# u -= s⋅L∇x
project!(u,L,x,s=1,reg=nothing) = @loop u[I,i] -= s*Lface(reg,L,I,i)*∂(i,I,x) over I ∈ inside(x), i ∈ 1:ndims(x)
# L[I,i], which is ℓ on the faces of the regular cells
@inline Lface(reg,L,I,i) = (ℓ = ℓof(reg,I); iszero(ℓ) ? @inbounds(L[I,i]) : ℓ)

function CFL(a::AbstractFlow;Δt_max=10)
    # on GPUs the maximum is reduced in the loop, with σ as the scratch array of the partials
    isgpu(a.σ) && return min(Δt_max,inv(first(loop_reduce(_flux_out,(max,max),inside(a.p),a.σ,a.u))+5a.ν))
    fill!(a.σ,0) # the ghosts hold scratch values (such as the fluxes of conv_diff!)
    @inside a.σ[I] = flux_out(I,a.u)
    min(Δt_max,inv(maximum(a.σ)+5a.ν))
end
@inline _flux_out(I,u) = (s = flux_out(I,u); (s,s))
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    s = z = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(z,u[I+δ(i,I),i])+max(z,-u[I,i]))
    end
    return s
end

"""
    udf!(flow::AbstractFlow,udf::Function,u,t)

User defined function using `udf::Function` to operate on `flow::AbstractFlow` during the
predictor and corrector step, in sync with time `t`. `u` is the velocity field the convective
flux is evaluated on in that phase (`a.u⁰` in the predictor, `a.u` in the corrector).
A `udf` that needs the advecting velocity (eg. SGS) should use `force!(flow,u,t; kwargs...)`.
Keyword arguments must be passed to `sim_step!` for them to be carried over the actual call.
"""
udf!(flow,::Nothing,u,t; kwargs...) = nothing
udf!(flow,force!::Function,u,t; kwargs...) =
    applicable(force!,flow,u,t) ? force!(flow,u,t; kwargs...) : force!(flow,t; kwargs...)
