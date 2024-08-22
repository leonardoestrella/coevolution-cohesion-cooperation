from random import uniform
import colorsys
from math import pi,sin,cos
import numpy as np
import random

class Boid:
	def __init__(self, parameters, behavior):

		self.cohesion = parameters["cohesion"] #strength of cohesion 
		self.alignment = parameters["alignment"] #strength of alignment
		self.radius = parameters["initial_radius"] #cohesive tendency
		self.r_tau = parameters["r_tau"] # repulsion radius
		self.coop = behavior  #cooperator or cheater
		self.fitness = 0 #initial fitness

	def offspring_gen(self, traits, params):
		
		self.radius = traits["radius"] #inherits the parent's cohesion
		self.radius += np.random.normal(0,params["mutation_coh"]) #mutation

		if self.radius < self.r_tau: #cohesive tendencies cannot be less than 0
			self.radius = self.r_tau

		if random.random() > params["mutation_coop"]: #Mutation in coopeartion trait
			self.coop = traits["coop"] #no mutation
		else:
			self.coop = not traits["coop"] #mutation

		self.fitness = 0 #restart fitness