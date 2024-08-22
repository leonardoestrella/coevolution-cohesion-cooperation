import numpy as np
from boid import Boid

def initialize_data(filename):
	with open (filename, "w") as file:
		pass
	with open(filename,"a") as f:
		f.write("id,generation,prop_cooperators,avg_cohesion,avg_cohesion_cooperators,avg_cohesion_cheaters,avg_fit_coops,avg_fit_cheat,speed\n")
	
def record_data_string(id,generation,flockMates,speed):
	
	coops = np.array([mate.coop for mate in flockMates])

	prop_coops = np.sum(coops) / len(flockMates)

	cohesive_ten = np.array([mate.radius for mate in flockMates]) - flockMates[0].r_tau

	avg_coh = np.mean(cohesive_ten)
	avg_coh_coops = np.mean(cohesive_ten[coops])
	avg_coh_cheat = np.mean(cohesive_ten[np.invert(coops)])

	fitness = np.array([mate.fitness for mate in flockMates])

	avg_fit_coops = np.mean(fitness[coops])
	avg_fit_cheat = np.mean(fitness[np.invert(coops)])

	data = [prop_coops,avg_coh,avg_coh_coops,avg_coh_cheat,avg_fit_coops,avg_fit_cheat, speed]

	entry = f"{id},{generation}," + ",".join(map(str,data)) + "\n"

	return entry