from boid import Boid #define the boids and their behavior
from movement import * #define the movement algorithm
from fitnessalgorithms import * #define fitness algorithms and calculations
from newgeneration import * #chooses parents and generates offspring
from filemanager import * #records data
import threading #allows parallel processing
import numpy as np #numpy 
import math #math
import random #random
from math import pi #constant

# 1. Initializes the flock (boid.py)
# 2. Allows the flock to move (movement.py)
# 3. Calculates fitness (fitnessalgorithms.py)
# 4. Decides offspring and its characteristics (offspring.py)
# 5. Initializes offspring flock (main.py)
# 6. Repeats from 2

# Paremeters for the model
generations = 100
steps_generation = 100 #2000
population = 10 #1024!

#initialization values (constant)
cohesion = 0.4
alignment = 0.4
r_tau = 1
speed = 1

#initialization values (traits)
initial_coop = 0.5 #Proportion of the cooperative trait in the population
initial_radius = r_tau+2 #This evolves to the cohesive tendencies

#Movement parameters
max_turning = 0.872665 # 50 degree in radians
error_turn = .05 # copying error
dt = 0.2 #stepsize
Width = 300
Height = 300

#Fitness parameters
group_radius = 5 #distance at which groups are formed
benefit = 100 #benefit from cooperators
c_cop = 1 #cost of cooperating
c_coh = 2 #cost of cohesive tendencies
base_fitness = 1 #base fitness

#Mutation rates
mutation_coh = 0.01
mutation_coop = 0.005

output_file = "output_file.txt"

params = {"id":1,
		"generations":generations,
		"steps_generation":steps_generation,
		"population":population,
		"cohesion":cohesion,
		"alignment": alignment,
		"r_tau": r_tau,
		"speed":speed,
		"initial_radius":initial_radius,
		"initial_coop":initial_coop,
		"max_turning":max_turning,
		"error_turn":error_turn,
		"group_radius": group_radius,
		"benefit": benefit,
		"c_cop": c_cop,
		"c_coh": c_coh,
		"base_fitness": base_fitness,
		"mutation_coh": mutation_coh,
		"mutation_coop": mutation_coop,
		"dt" : dt,
		"Width" : Width,
		"Height" : Height
		}

def run_simulation (parameters, fitness_alg):
	
	generations = parameters["generations"]
	steps_generation = parameters["steps_generation"]
	population = parameters["population"]
	prop_coops = parameters["initial_coop"]
	speed = parameters["speed"]
	Width = parameters["Width"]
	Height = parameters["Height"]

	#Determine the behavior of the agents
	coops = int(population*prop_coops)
	cheaters = population - coops
	behaviors = np.array([True]*coops + [False]*cheaters)
	np.random.shuffle(behaviors)

	#Initialize position and velocities
	temp = init_pos_vel(population, Width, Height, speed)
	positions = temp[0]
	velocities = temp[1]
	#Initialize flock
	flock = [Boid(parameters,behaviors[i]) for i in range(population)] #boid.py

	#used to record data
	data_string = ''

	for generation in range(generations):

		final_state = move_flock(flock,steps_generation, parameters, positions,velocities) #Moves the flock by steps_generation steps (movement.py)
		positions = final_state[0] #updates the final positions
		velocities = final_state[1] #updates the final velocitieis
		fitness_alg(flock, positions, parameters) #calculate fitness (fitnessalgorithms.py)
		data_string += record_data_string(parameters["id"],generation,flock,speed) #record data in the string

		update_flock(flock, parameters) #generate offspring (newgeneration.py)
		
		temp = init_pos_vel(population, Width, Height, speed) #restart positions and velocities
		positions = temp[0]
		velocities = temp[1]

	with file_lock: #avoids problems with threads
		with open(output_file, "a") as f:
			f.write(data_string)

file_lock = threading.Lock()
initialize_data(output_file)

#Run for many steps (specify range of speeds and repetitions!)
"""
speeds = 10**np.linspace(-4,2,7)
i_count= 0
repetitions = 5

for i in range(repetitions):
	data_store = []
	threads = []
	for speed in speeds:
		input_par = params.copy()
		input_par["id"] = i_count
		input_par["speed"] = speed
		thread = threading.Thread(target=run_simulation, args=(input_par,joshi_fitness))
		threads.append(thread)
		thread.start()
		i_count+=1
	for thread in threads:
		thread.join()
"""
#run once
run_simulation(params,joshi_fitness)