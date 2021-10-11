import IPython
import socialforce
import torch

from social_force.utils import create_square

def initial_state_corridor(n):
    _ = torch.manual_seed(42)

    # first n people go right, second n people go left
    state = torch.zeros((n * 2, 6))

    # positions
    state[:n, 0:2] = ((torch.rand((n, 2)) - 0.5) * 2.0) * torch.tensor([25.0, 4.5])
    state[n:, 0:2] = ((torch.rand((n, 2)) - 0.5) * 2.0) * torch.tensor([25.0, 4.5])

    # velocity
    state[:n, 2] = torch.normal(torch.full((n,), 1.34), 0.26)
    state[n:, 2] = torch.normal(torch.full((n,), -1.34), 0.26)

    # x destination
    state[:n, 4] = 100.0
    state[n:, 4] = -100.0

    return state


initial_state = initial_state_corridor(60)

upper_wall = torch.stack([torch.linspace(-30, 30, 600), torch.full((600,), 5)], -1)
lower_wall = torch.stack([torch.linspace(-30, 30, 600), torch.full((600,), -5)], -1)
mid_square = create_square(0, 0)
arr = [upper_wall, lower_wall]
arr.extend(mid_square)
ped_space = socialforce.potentials.PedSpacePotential(arr)

ped_ped = socialforce.potentials.PedPedPotential()
simulator = socialforce.Simulator(ped_ped=ped_ped, ped_space=ped_space,
                                  oversampling=2, delta_t=0.08)
simulator.integrator = socialforce.simulator.PeriodicBoundary(
    simulator.integrator, x_boundary=[-25.0, 25.0])

with torch.no_grad():
    states_sf = simulator.run(initial_state, 250)

# with socialforce.show.track_canvas(ncols=2, figsize=(12, 2), tight_layout=False) as (ax1, ax2):
#     socialforce.show.states(ax1, states_sf[0:1], monochrome=True)
#     socialforce.show.space(ax1, ped_space)
#     ax1.text(0.1, 0.1, '$t = 0s$', transform=ax1.transAxes)
#     ax1.set_xlim(-25, 25)
#
#     socialforce.show.states(ax2, states_sf[249:250], monochrome=True)
#     socialforce.show.space(ax2, ped_space)
#     ax2.text(0.1, 0.1, '$t = 20s$', transform=ax2.transAxes)
#     ax2.set_xlim(-25, 25)

with socialforce.show.track_canvas(figsize=(6, 2), tight_layout=False, show=False, dpi=130) as ax:
    ax.set_xlim(-25, 25)
    socialforce.show.space(ax, ped_space)
    video = socialforce.show.state_animation(ax, states_sf, delta_t=0.08)

video.save("pedmove.gif")

