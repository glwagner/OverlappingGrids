using Oceananigans
using Oceananigans: time_step!
using Oceananigans.Fields: ConstantField
using GLMakie

U = 1
κ = 1
Nx = 100

closure = ScalarDiffusivity(; κ)
velocities = PrescribedVelocityFields(u=ConstantField(U))
tracers = :c
buoyancy = nothing
model_args = (; closure, velocities, tracers, buoyancy)

left_bcs = FieldBoundaryConditions(east=nothing)
right_bcs = FieldBoundaryConditions(west=nothing)

left_grid = RectilinearGrid(size=Nx, x=(0, 1), topology=(Bounded, Flat, Flat))
left_model = HydrostaticFreeSurfaceModel(grid=left_grid; boundary_conditions=(; c=left_bcs), model_args...)

right_grid = RectilinearGrid(size=Nx, x=(1, 2), topology=(Bounded, Flat, Flat))
right_model = HydrostaticFreeSurfaceModel(grid=right_grid; boundary_conditions=(; c=right_bcs), model_args...)

x₀ = 0.5
dx = 0.1
cᵢ(x) = exp(-(x-x₀)^2 / 2dx^2)
set!(left_model, c=cᵢ)

Δx = 1 / Nx
Δt = 1e-1 * Δx^2 # κ=1
left_simulation = Simulation(left_model; Δt)
right_simulation = Simulation(right_model; Δt)

for n = 1:10000
    time_step!(left_simulation)
    time_step!(right_simulation)

    cL = left_model.tracers.c
    cR = right_model.tracers.c

    @inbounds begin
        cL[Nx + 1, 1, 1] = cR[1, 1, 1]
        cR[0, 1, 1] = cL[Nx, 1, 1]
    end
end

cL = left_model.tracers.c
cR = right_model.tracers.c
lines(cL)
lines!(cR)
display(current_figure())
