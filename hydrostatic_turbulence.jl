using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using GLMakie

grid = RectilinearGrid(size = (128, 128, 1),
                       halo = (5, 5, 5),
                       x = (-5, 5),
                       y = (-5, 5),
                       z = (0, 1),
                       topology = (Periodic, Periodic, Bounded))

island(x, y) = (x^2 + y^2) < 1
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(island))

momentum_advection = VectorInvariant(vorticity_scheme = WENO(order=5),
                                     vertical_scheme = Centered(),
                                     divergence_scheme = WENO(order=5))
buoyancy = nothing
tracers = nothing
free_surface = ExplicitFreeSurface(gravitational_acceleration=10)
model_args = (; momentum_advection, buoyancy, tracers, free_surface)
model = HydrostaticFreeSurfaceModel(; grid, model_args...)

uᵢ(x, y, z) = 2rand() - 1 + 2
vᵢ(x, y, z) = 2rand() - 1
set!(model, u=uᵢ, v=vᵢ)

g = free_surface.gravitational_acceleration
c = sqrt(g)
Δx = xspacings(grid.underlying_grid, Center())
Δt = 0.1 * Δx / c
simulation = Simulation(model; Δt) 

for n = 1:2000
    time_step!(simulation)
end

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)
mask_immersed_field!(ζ, NaN) 

heatmap(ζ, axis=(; aspect=1), colormap=:balance, nan_color=:gray)
display(current_figure())

