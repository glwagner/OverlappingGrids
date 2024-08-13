using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Grids: halo_size
using Oceananigans.Fields: interpolate!, interpolate, _node
using GLMakie

function hydrostatic_model(grid; boundary_conditions=NamedTuple())
    momentum_advection = VectorInvariant(vorticity_scheme = WENO(order=5),
                                         vertical_scheme = Centered(),
                                         divergence_scheme = WENO(order=5))
    buoyancy = nothing
    tracers = nothing
    free_surface = ExplicitFreeSurface(gravitational_acceleration=10)
    model_args = (; momentum_advection, buoyancy, tracers,
                  free_surface, boundary_conditions)
    
    model = HydrostaticFreeSurfaceModel(; grid, model_args...)

    return model
end

outer_grid = RectilinearGrid(size = (128, 128, 1),
                             halo = (5, 5, 5),
                             x = (-5, 5),
                             y = (-5, 5),
                             z = (0, 1),
                             topology = (Periodic, Periodic, Bounded))

outer_model = hydrostatic_model(outer_grid)

uᵢ(x, y, z) = 2rand() - 1 + 2
vᵢ(x, y, z) = 2rand() - 1
set!(outer_model, u=uᵢ, v=vᵢ)

inner_grid = RectilinearGrid(size = (128, 128, 1),
                             halo = (5, 5, 5),
                             x = (-1.5, 1.5),
                             y = (-1.5, 1.5),
                             z = (0, 1),
                             topology = (Bounded, Bounded, Bounded))

island(x, y) = (x^2 + y^2) < 1
inner_grid = ImmersedBoundaryGrid(inner_grid, GridFittedBottom(island))

u_bcs = FieldBoundaryConditions(west=nothing, east=nothing, north=nothing, south=nothing)
v_bcs = FieldBoundaryConditions(west=nothing, east=nothing, north=nothing, south=nothing)
inner_boundary_conditions = (u=u_bcs, v=v_bcs)
inner_model = hydrostatic_model(inner_grid, boundary_conditions=inner_boundary_conditions)

ui, vi, wi = inner_model.velocities
ηi = inner_model.free_surface.η

uo, vo, wo = outer_model.velocities
ηo = outer_model.free_surface.η

interpolate!(ui, uo)
interpolate!(vi, vo)
interpolate!(ηi, ηo)

mask_immersed_field!(ui, NaN) 
mask_immersed_field!(vi, NaN) 
mask_immersed_field!(ηi, NaN) 

function makeplot()
    ui, vi, wi = inner_model.velocities
    ηi = inner_model.free_surface.η

    uo, vo, wo = outer_model.velocities
    ηo = outer_model.free_surface.η

    mask_immersed_field!(ui, NaN) 
    mask_immersed_field!(vi, NaN) 

    fig = Figure()
    axui = Axis(fig[1, 1], aspect=1)
    axvi = Axis(fig[1, 2], aspect=1)
    axηi = Axis(fig[1, 3], aspect=1)
    axuo = Axis(fig[2, 1], aspect=1)
    axvo = Axis(fig[2, 2], aspect=1)
    axηo = Axis(fig[2, 3], aspect=1)

    heatmap!(axui, ui, colormap=:balance)
    heatmap!(axvi, vi, colormap=:balance)
    heatmap!(axηi, ηi, colormap=:balance)
    heatmap!(axuo, uo, colormap=:balance)
    heatmap!(axvo, vo, colormap=:balance)
    heatmap!(axηo, ηo, colormap=:balance)

    return fig
end

const C = Center()
const F = Face()

@inline function _interpolate_u!(i, j, k, grid1, u1, grid2, u2)
    Xu = _node(i, j, 1, grid1, F, C, C)
    @inbounds u1[i, j, 1] = interpolate(Xu, u2, (F, C, C), grid2)
    return nothing
end

@inline function _interpolate_v!(i, j, k, grid1, v1, grid2, v2)
    Xv = _node(i, j, 1, grid1, C, F, C)
    @inbounds v1[i, j, 1] = interpolate(Xv, v2, (C, F, C), grid2)
    return nothing
end

@inline function _interpolate_η!(i, j, k, grid1, η1, grid2, η2)
    Xη = _node(i, j, 1, grid1, C, C, C)
    @inbounds η1[i, j, 1] = interpolate(Xη, η2, (C, C, C), grid2)
    return nothing
end

function interpolate_solution!(outer_model, inner_model)
    ui, vi, wi = inner_model.velocities
    ηi = inner_model.free_surface.η
    
    uo, vo, wo = outer_model.velocities
    ηo = outer_model.free_surface.η

    Nix, Niy, Niz = size(inner_model.grid)
    Hix, Hiy, Hiz = halo_size(inner_model.grid)

    xi = xnodes(inner_grid, F)
    xo = xnodes(outer_grid, F)
    x1 = xi[1]
    x2 = xi[Nix]

    yi = ynodes(inner_grid, F)
    yo = ynodes(outer_grid, F)
    y1 = yi[1]
    y2 = yi[Niy]

    # Fill inner solution halos and boundary values from outer solution
    # Loop in x first
    for j = -Hiy+1:Niy+Hiy
        for i = Nix+1:Nix+Hix
            _interpolate_u!(i, j, 1, inner_grid, ui, outer_grid, uo)
        end

        for i = -Hix+1:1
            _interpolate_u!(i, j, 1, inner_grid, ui, outer_grid, uo)
        end
    end

    for j = -Hiy+1:Niy+Hiy
        for i = Nix:Nix+Hix
            _interpolate_v!(i, j, 1, inner_grid, vi, outer_grid, vo)
            _interpolate_η!(i, j, 1, inner_grid, ηi, outer_grid, ηo)
        end

        for i = -Hix+1:0
            _interpolate_v!(i, j, 1, inner_grid, vi, outer_grid, vo)
            _interpolate_η!(i, j, 1, inner_grid, ηi, outer_grid, ηo)
        end
    end

    # y-halo
    for i = -Hix+1:Nix+Hix
        for j = Niy+1:Niy+Hiy
            _interpolate_u!(i, j, 1, inner_grid, ui, outer_grid, uo)
            _interpolate_η!(i, j, 1, inner_grid, ηi, outer_grid, ηo)
        end

        for j = -Hiy+1:0
            _interpolate_u!(i, j, 1, inner_grid, ui, outer_grid, uo)
            _interpolate_η!(i, j, 1, inner_grid, ηi, outer_grid, ηo)
        end
    end

    for i = -Hix+1:Nix+Hix
        for j = Niy+1:Niy+Hiy
            _interpolate_v!(i, j, 1, inner_grid, vi, outer_grid, vo)
        end

        for j = -Hiy+2:1
            _interpolate_v!(i, j, 1, inner_grid, vi, outer_grid, vo)
        end
    end

    # interpolate outer solution from inner grid
    i1 = findfirst(x -> x >= x1, xo)
    i2 = findfirst(x -> x >= x2, xo) - 1

    j1 = findfirst(y -> y >= y1, yo)
    j2 = findfirst(y -> y >= y2, yo) - 1

    for i = i1:i2-1, j = j1:j2
        _interpolate_u!(i, j, 1, outer_grid, uo, inner_grid, ui)
    end

    for i = i1:i2, j = j1:j2-1
        _interpolate_v!(i, j, 1, outer_grid, vo, inner_grid, vi)
    end

    for i = i1:i2, j = j1:j2
        _interpolate_η!(i, j, 1, outer_grid, ηo, inner_grid, ηi)
    end

    return nothing
end

interpolate_solution!(outer_model, inner_model)
makeplot()

g = inner_model.free_surface.gravitational_acceleration
c = sqrt(g)
Δx = xspacings(inner_grid.underlying_grid, Center())
Δt = 0.1 * Δx / c
inner_simulation = Simulation(inner_model; Δt) 
outer_simulation = Simulation(outer_model; Δt) 

pop!(inner_simulation.callbacks, :nan_checker)
pop!(outer_simulation.callbacks, :nan_checker)

for n = 1:100
    time_step!(inner_simulation)
    time_step!(outer_simulation)
    interpolate_solution!(outer_model, inner_model)
end

mask_immersed_field!(ui, NaN) 
mask_immersed_field!(vi, NaN) 
interpolate_solution!(outer_model, inner_model)

makeplot()

