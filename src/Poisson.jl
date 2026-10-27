abstract type AbstractPoisson{T,S,V} end

"""
    Poisson{T::Float, S<:AbstractArray{T,D}, V<:AbstractArray{T,D+1}}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = z

where A is symmetric, block-tridiagonal and extremely sparse. Moreover,
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])`. This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

To help iteratively solve the system above, the Poisson structure holds
helper arrays for `inv(D)`, the error `ϵ`, and residual `r=z-Ax`. An iterative
solution method then estimates the error `ϵ=̃A⁻¹r` and increments `x+=ϵ`, `r-=Aϵ`.

On GPUs, a byte `reg[I]≠0` flags a regular cell, where `L=ℓ` on all faces (the fluid away from a body),
and the stencils of these cells use `ℓ` instead of reading `L`, `D` and `iD`. `ℓ` is the largest `L`
when the level is made.
"""
struct Poisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    L :: V # Lower diagonal coefficients
    D :: S # Diagonal coefficients
    iD :: S # 1/Diagonal
    reg :: S # regular-cell flags, one byte per cell packed in an array like x (set on GPUs only)
    ℓ :: T # L on all faces of a regular cell
    x :: S # approximate solution
    ϵ :: S # increment/error
    r :: S # residual
    z :: S # source
    n :: Vector{Int16} # pressure solver iterations
    perdir :: NTuple # direction of periodic boundary condition
    function Poisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=()) where T
        @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
        r = similar(x); fill!(r,0)
        ϵ,D,iD = copy(r),copy(r),copy(r)
        reg = similar(x,(cld(size(x,1),sizeof(T)),Base.tail(size(x))...)); fill!(reg,0)
        ℓ = isgpu(x) ? maximum(L) : one(T)
        isgpu(x) ? set_diag!(D,iD,flags(reg),ℓ,L) : set_diag!(D,iD,L)
        new{T,typeof(x),typeof(L)}(L,D,iD,reg,ℓ,x,ϵ,r,z,[],perdir)
    end
end

using ForwardDiff: Dual,Tag
Base.eps(::Type{D}) where D<:Dual{Tag{G,T}} where {G,T} = eps(T)
function set_diag!(D,iD,L)
    @inside D[I] = diag(I,L)
    @inside iD[I] = iszero(D[I]) ? D[I] : inv(D[I])
end
set_diag!(D,iD,reg,ℓ,L) = @loop (D[I] = diag(I,L); iD[I] = iszero(D[I]) ? D[I] : inv(D[I]);
                                 reg[I] = regular(I,L,ℓ)) over I ∈ inside(D)
update!(p::Poisson) = isgpu(p.x) ? set_diag!(p.D,p.iD,flags(p.reg),p.ℓ,p.L) : set_diag!(p.D,p.iD,p.L) # reg is used on GPUs only
flags(reg) = reinterpret(UInt8,reg)

# the flag of cell `I`: whether `L=ℓ` on all its faces
@inline function regular(I::CartesianIndex{d},L,ℓ) where {d}
    b = true
    for i in 1:d
        b &= @inbounds(L[I,i]==ℓ && L[I+δ(i,I),i]==ℓ)
    end
    return UInt8(b)
end
regular(I::CartesianIndex,L::AbstractArray{<:Dual},ℓ) = 0x00 # equal values can have different partials

@fastmath @inline function diag(I::CartesianIndex{d},L) where {d}
    s = zero(eltype(L))
    for i in 1:d
        s -= @inbounds(L[I,i]+L[I+δ(i,I),i])
    end
    return s
end
@fastmath @inline δv(D,I::CartesianIndex{d}) where {d} = 1+@inbounds(D[I])/2d
diagonal(p::Poisson) = p.D

"""
    mult!(p::Poisson,x)

Efficient function for Poisson matrix-vector multiplication.
Fills `p.z = p.A x` with 0 in the ghost cells.
"""
function mult!(p::Poisson,x)
    @assert axes(p.z)==axes(x)
    perBC!(x,p.perdir)
    fill!(p.z,0)
    reg = regs(p)
    @inside p.z[I] = mult(I,p.L,p.D,reg,x)
    return p.z
end
"""
    mult(I,L,D,x,args...)

`(Ax)[I]`. `x` is an array, or a function `x(J,args...)` evaluated on the fly.
"""
@fastmath @inline function mult(I::CartesianIndex{d},L,D,x,args...) where {d}
    s = at(x,I,args...)*@inbounds(D[I])
    for i in 1:d
        s += at(x,I-δ(i,I),args...)*@inbounds(L[I,i])+at(x,I+δ(i,I),args...)*@inbounds(L[I+δ(i,I),i])
    end
    return s
end
# `x[J]`, or the value `x(J,args...)` of a field that is evaluated on the fly
@inline at(x::AbstractArray,J,args...) = @inbounds x[J]
@inline at(x::Function,J,args...) = x(J,args...)

# `ℓ` of a regular cell or 0. `reg=nothing` treats every cell as irregular
@inline ℓof((f,ℓ)::Tuple,I) = ifelse(@inbounds(f[I])==0x00,zero(ℓ),ℓ)
@inline ℓof(::Nothing,I) = false
regs(p::Poisson) = isgpu(p.x) ? (flags(p.reg),p.ℓ) : nothing # the branch costs more than the memory traffic it saves on CPUs
regs(p::AbstractPoisson) = nothing

"""
    mult(I,L,D,reg,x)

`(Ax)[I]`, using `reg` to skip reading `L` and `D` in regular cells.
"""
@fastmath @inline function mult(I::CartesianIndex{d},L,D,reg::Union{Tuple,Nothing},x) where {d}
    ℓ = ℓof(reg,I)
    iszero(ℓ) && return mult(I,L,D,x)
    s = -2d*@inbounds(x[I])
    for i in 1:d
        s += @inbounds(x[I-δ(i,I)]+x[I+δ(i,I)])
    end
    return ℓ*s
end

"""
    residual!(p::Poisson,src=nothing)

Computes the residual `r = z-Ax` and corrects it such that
`r = 0` if `iD==0` which ensures local satisfiability
    and
`sum(r) = 0` which ensures global satisfiability.

The global correction is done by adjusting all points uniformly,
minimizing the local effect. Other approaches are possible.

Note: These corrections mean `x` is not strictly solving `Ax=z`, but
without the corrections, no solution exists.

With `src=(u,V,idt)`, the source `z = idt*(div(u)-δv(D)*div(V))` is evaluated in the
residual kernel, and `p.z` is neither read nor written.
"""
function residual!(p::Poisson,src=nothing)
    perBC!(p.x,p.perdir)
    reg = regs(p)
    @inside p.r[I] = ifelse(p.iD[I]==0,zero(eltype(p.r)),source(I,p.z,p.D,src)-mult(I,p.L,p.D,reg,p.x))
    s = meanr(p.r,p.iD) # only solved rows absorb the imbalance
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = ifelse(p.iD[I]==0,zero(eltype(p.r)),p.r[I]-s)
end
@inline source(I,z,D,::Nothing) = @inbounds z[I]
@inline source(I,z,D,(u,V,idt)::Tuple) = idt*(div(I,u)-δv(D,I)*div(I,V))

function increment!(p::Poisson{T};ω=1) where {T}
    perBC!(p.ϵ,p.perdir); reg = regs(p)
    @loop (p.r[I] = p.r[I]-ω*mult(I,p.L,p.D,reg,p.ϵ);
           p.x[I] = p.x[I]+ω*p.ϵ[I]) over I ∈ inside(p.x)
end

using KernelAbstractions: @localmem, @synchronize
"""
    loop_reduce(f,(op₁,op₂),R,buf,args...)

GPU loop over `I ∈ R` that runs `f(I,args...)` and reduces the pair of values it returns with
`op₁` and `op₂`, the starting values being zero. Each work-group reduces its pairs in shared
memory and writes them to the scratch array `buf`, which `f` must not use; the partials are then
reduced in one small `mapreduce`.
"""
function loop_reduce(f::F,(op₁,op₂),R,buf,args...) where F
    T = eltype(buf); ng = cld(length(R),REDUCEGROUP)
    @assert 2ng ≤ length(buf) "loop_reduce: the scratch array is too small for the partials"
    s,m = view(vec(buf),1:ng),view(vec(buf),ng+1:2ng)
    _loop_reduce!(get_backend(buf),REDUCEGROUP)(f,op₁,op₂,s,m,R,args;ndrange=ng*REDUCEGROUP)
    return mapreduce((a,b)->(a,b),((a,b),(c,d))->(op₁(a,c),op₂(b,d)),s,m;init=(zero(T),zero(T)))
end
const REDUCEGROUP = 256 # work-items per work-group of loop_reduce
@kernel function _loop_reduce!(f,op₁,op₂,s,m,R,args)
    i = @index(Global,Linear); t = @index(Local,Linear); g = @index(Group,Linear)
    a = @localmem eltype(s) (REDUCEGROUP,); b = @localmem eltype(s) (REDUCEGROUP,)
    v₁ = v₂ = zero(eltype(s))
    i ≤ length(R) && ((v₁,v₂) = f(@inbounds(R[i]),args...))
    @inbounds a[t] = v₁; @inbounds b[t] = v₂
    @synchronize
    k = REDUCEGROUP÷2
    while k ≥ 1
        t ≤ k && (@inbounds a[t] = op₁(a[t],a[t+k]); @inbounds b[t] = op₂(b[t],b[t+k]))
        @synchronize
        k ÷= 2
    end
    t == 1 && (@inbounds s[g] = a[1]; @inbounds m[g] = b[1])
end

"""
    increment_L₁∞!(p::Poisson;ω=1)

`increment!(p;ω)` followed by `L₁∞(p)`. On GPUs, the norms are reduced in the increment kernel,
with `p.z` as the scratch array.
"""
function increment_L₁∞!(p::Poisson{T};ω=1) where {T}
    isgpu(p.x) || (increment!(p;ω); return L₁∞(p))
    perBC!(p.ϵ,p.perdir)
    loop_reduce(_increment,(+,max),inside(p.x),p.z,p.r,p.x,p.ϵ,p.L,p.D,T(ω))
end
@fastmath @inline function _increment(I,r,x,ϵ,L,D,ω)
    @inbounds v = r[I]-ω*mult(I,L,D,ϵ); @inbounds r[I] = v
    @inbounds x[I] += ω*ϵ[I]
    return abs(v),abs(v)
end

"""
    residual_L₁∞!(p::Poisson,src=nothing)

`residual!(p,src)` followed by `L₁∞(p)`. On GPUs, the residual kernel also sums `r` and counts
the solved cells, and the kernel that subtracts the mean also reduces the norms, with `p.ϵ` as
the scratch array.
"""
function residual_L₁∞!(p::Poisson{T},src=nothing) where {T}
    isgpu(p.x) || (residual!(p,src); return L₁∞(p))
    s = residual_mean!(p,src)
    iszero(s) && return L₁∞(p)
    loop_reduce(_fix_residual,(+,max),inside(p.x),p.ϵ,p.r,p.iD,s)
end
# The residual kernel of `residual_L₁∞!` (GPUs): `r` without the correction, and the mean `s` that the
# correction subtracts from it (zero when negligible)
function residual_mean!(p::Poisson{T},src=nothing) where {T}
    perBC!(p.x,p.perdir)
    Σr,n = loop_reduce(_residual,(+,+),inside(p.x),p.ϵ,p.r,p.x,p.z,p.L,p.D,p.iD,src)
    s = Σr/n # only solved rows absorb the imbalance
    return abs(s) <= 2eps(T) ? zero(T) : s
end
@fastmath @inline function _residual(I,r,x,z,L,D,iD,src)
    solved = !iszero(@inbounds iD[I])
    v = solved ? source(I,z,D,src)-mult(I,L,D,x) : zero(eltype(r))
    @inbounds r[I] = v
    return v,eltype(r)(solved)
end
@inline function _fix_residual(I,r,iD,s)
    v = iszero(@inbounds iD[I]) ? zero(eltype(r)) : @inbounds(r[I])-s
    @inbounds r[I] = v
    return abs(v),abs(v)
end
"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother. Runs `it` iterations with relaxation parameter `ω` scaling the deferred corrections in `increment!`.
Note: This runs for general backends but converges _very_ slowly.
"""
@fastmath Jacobi!(p;it=1,ω=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    increment!(p;ω)
end

@fastmath @inline function gauss(I::CartesianIndex{d},r,L,iD,x,args...) where {d}
    s = @inbounds(r[I])
    for i in 1:d
        s -= at(x,I-δ(i,I),args...)*@inbounds(L[I,i]) + at(x,I+δ(i,I),args...)*@inbounds(L[I+δ(i,I),i])
    end
    return s*@inbounds(iD[I])
end
@fastmath @inline function gauss(I::CartesianIndex{d},r,L,iD,reg::Union{Tuple,Nothing},x) where {d}
    ℓ = ℓof(reg,I)
    iszero(ℓ) && return gauss(I,r,L,iD,x)
    s = @inbounds(r[I])
    for i in 1:d
        s -= ℓ*@inbounds(x[I-δ(i,I)]+x[I+δ(i,I)])
    end
    return s*inv(-2d*ℓ) # iD of a regular cell
end

# x[I] = gauss(I,…) on colour k₀; the neighbours are read from `y`, which is `x` unless
# given as a function `y(J,args...)`
@inline function gauss_rb(x,r,L,iD,k₀,Iv::CartesianIndex{d},y=x,args...) where {d}
    I,ok = rb(x,k₀,Iv)
    ok && (x[I] = gauss(I,r,L,iD,y,args...))
end
@inline function gauss_rb(x,r,L,iD,reg,k₀,Iv::CartesianIndex{d}) where {d}
    I,ok = rb(x,k₀,Iv)
    ok && (x[I] = gauss(I,r,L,iD,reg,x))
end
@inline function rb(x,k₀,Iv::CartesianIndex{d}) where {d}
    k = 2*Iv.I[end] - 1 - (sum(Base.front(Iv.I)) + k₀) % 2 # double the k-index and shift for red-black indexing
    return CartesianIndex(ntuple( i-> i==d ? k : Iv.I[i], d)), k < size(x,d) # with an odd size, the last k is a ghost for one colour
end

@inline function half_rangek(x::AbstractArray{T,N}) where{T,N}
    return CartesianIndices(ntuple( i-> i==N ? (2:cld(size(x,i),2)) : (2:size(x,i)-1), N)) # every interior k of both colours
end

"""
    GaussSeidelRB!(p::Poisson;it=4, ω=1, norms=false)

Red-black Gauss-Seidel smoother. Runs `it` iterations; a complete red-black cycle requires `it` to be even.
`ω` under-/over-relaxes the solution through scaling the deferred corrections in `increment!`.
With `norms=true`, it returns `L₁∞(p)` of the new residual (see `increment_L₁∞!`).
Note: This performs best on GPU configurations and is the default smoother.
"""
function GaussSeidelRB!(p::Poisson{T};it=4, ω=1, norms=false) where {T}
    reg = regs(p)
    if isgpu(p.x) && isempty(p.perdir) && it ≥ 2
        # the first half sweep reads the initial ϵ=r⊙iD on the fly, and the second sets the rest of ϵ
        @loop gauss_rb(p.ϵ,p.r,p.L,p.iD,1,I,jacobi0,p.r,p.iD) over I ∈ half_rangek(p.ϵ)
        i₀ = 2
    else
        @inside p.ϵ[I] = p.r[I]*p.iD[I]  # initialize ϵ
        perBC!(p.ϵ,p.perdir); i₀ = 1
    end
    for i ∈ i₀:it
        @loop gauss_rb(p.ϵ,p.r,p.L,p.iD,reg,i,I) over I ∈ half_rangek(p.ϵ)
    end
    norms && return increment_L₁∞!(p;ω)
    increment!(p;ω) # increment solution and residual
end
@inline jacobi0(J,r,iD) = @inbounds r[J]*iD[J]

using LinearAlgebra: ⋅
using KernelAbstractions: Backend, GPU
"""
    perdot(a,b,perdir)

Apply dot product to the inner cells of two _scalar_ fields, assuming zero values in ghost cell when using Neumann BC.
"""
perdot(a,b,::Tuple{}) = a⋅b
perdot(a,b,perdir,R=inside(a)) = perdot(get_backend(a),a,b,R)
perdot(::Backend,a,b,R) = sum(@inbounds(a[I]*b[I]) for I ∈ R)
perdot(::GPU,a,b,R) = mapreduce(*,+,view(a,R.indices...),view(b,R.indices...))

"""
    pcg!(p::Poisson; it=6)

Conjugate-Gradient smoother with Jacobi predictioning. Runs at most `it` iterations,
but will exit early if the Gram-Schmidt update parameter `|α| < 1%` or `|r D⁻¹ r| < 1e-8`.
Note: This runs for general backends.
"""
function pcg!(p::Poisson{T};it=6,kwargs...) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z
    abs(rho)<10eps(T) && return
    reg = regs(p)
    for i in 1:it
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,reg,ϵ)
        alpha = rho/perdot(z,ϵ,p.perdir)
        (abs(alpha)<1e-2 || abs(alpha)>1e2) && return # alpha should be O(1)
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        i==it && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        abs(rho2)<10eps(T) && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
    end
end

"""
    L₂(a,R=inside(a))

Squared L₂ norm of array `a` over `R`, which excludes the ghost cells by default.
"""
L₂(a,R::CartesianIndices=inside(a)) = L₂(get_backend(a),a,R)
L₂(backend,a,R) = sum(abs2,@inbounds(a[I]) for I ∈ R)
L₂(::GPU,a,R) = mapreduce(abs2,+,view(a,R.indices...))
L₂(p::Poisson) = p.r ⋅ p.r # special method since outside(p.r)≡0
L₁(p::Poisson) = sum(abs,p.r) # special method since outside(p.r)≡0
L∞(p::Poisson) = maximum(abs,p.r)
"""
    L₁∞(p::Poisson)

`(L₁(p),L∞(p))`, in a single reduction on GPUs.
"""
L₁∞(p::Poisson) = L₁∞(get_backend(p.r),p.r)
L₁∞(::Backend,r) = (sum(abs,r),maximum(abs,r))
L₁∞(::GPU,r) = mapreduce(a->(abs(a),abs(a)),((s,m),(t,n))->(s+t,max(m,n)),r;init=(zero(eltype(r)),zero(eltype(r))))
# mean of `r` over the solved cells (`iD≠0`), in a single reduction on GPUs
meanr(r,iD) = meanr(get_backend(r),r,iD)
meanr(::Backend,r,iD) = sum(r)/count(!iszero,iD)
function meanr(::GPU,r,iD)
    s,n = mapreduce((a,d)->(a,Int32(!iszero(d))),((s,n),(t,m))->(s+t,n+m),r,iD;init=(zero(eltype(r)),Int32(0)))
    return s/n
end

# default tolerance of `solver!`
const default_tol = 2e-3

# mean residual  Σ|r|/N < tol/10   ⟺   L₁(p)=Σ|r| < (tol/10)·N
l1n_tol(p::AbstractPoisson, tol) = (Float64(tol)/10) * length(inside(p.r))

"""
    solver!(A::Poisson;tol=2e-3,itmx=1e3)

Approximate iterative solver for the Poisson matrix equation `Ax=b`.

  - `A`: Poisson matrix with working arrays.
  - `A.x`: Solution vector. Can start with an initial guess.
  - `A.z`: Right-Hand-Side vector. Will be overwritten!
  - `A.n[end]`: stores the number of iterations performed.
  - `tol`: Grid-independent max-norm (worst-cell) tolerance `max|r| < tol`. This is the
        knob to tune: on refined grids the mean residual clears `tol/10` with margin, so
        the max-norm is the binding constraint — lower `tol` for tighter divergence.
        Convergence also requires the mean residual `Σ|r|/N < tol/10` (same units as the
        max-norm, no hidden exponents: the bulk sits 10× below the cap).
  - `itmx`: Maximum number of iterations.
  - `src`: `nothing` to use `A.z`, or `(u,V,idt)` for the source of `residual!(A,src)`.
"""
function solver!(p::Poisson;tol=default_tol,itmx=1e3,src=nothing)
    r₁tol = l1n_tol(p, tol); r∞tol = tol
    r₁,r∞ = residual_L₁∞!(p,src)
    nᵖ=0; @log ", $nᵖ, $r∞, $r₁\n"
    while nᵖ<itmx
        pcg!(p); r₁,r∞ = L₁∞(p); nᵖ+=1
        @log ", $nᵖ, $r∞, $r₁\n"
        (r₁<r₁tol && r∞<r∞tol) && break
    end
    perBC!(p.x,p.perdir)
    push!(p.n,nᵖ)
end