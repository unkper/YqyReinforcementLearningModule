import socialforce
import torch

from social_force.utils import *
from ped_env.utils.maps import map_05, map_06, map_07


def simulate_map(map:Map):
    peds = parse_map_for_people(map, 60, desired_speed=1.0)
    space = parse_map_for_space(map)

    ped_ped = socialforce.potentials.PedPedPotential()
    simulator = socialforce.Simulator(ped_ped=ped_ped, ped_space=space,
                                      oversampling=2, delta_t=0.08)
    simulator.integrator = socialforce.simulator.PeriodicBoundary(
        simulator.integrator)

    with torch.no_grad():
        states_sf = simulator.run(peds, 600)

    with socialforce.show.track_canvas(figsize=(10, 10), tight_layout=False, show=False, dpi=130) as ax:
        ax.set_xlim(0, 25)
        socialforce.show.space(ax, space)
        video = socialforce.show.state_animation(ax, states_sf, delta_t=0.08)

    video.save("pedmove{}.gif".format(str(map)))

simulate_map(map_05)
simulate_map(map_06)
simulate_map(map_07)