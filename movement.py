
from boid import Boid
import numpy as np
import math
import random
from math import pi



def init_pos_vel(population,Width, Height, speed): #initializes random positions and velocities
	positions = np.random.rand(population,2) * np.array([Width,Height])
	angles = np.random.rand(population) * 2*pi
	cos_angles = np.cos(angles)
	sin_angles = np.sin(angles)
	velocities = np.stack((cos_angles,sin_angles),axis=-1)
	velocities*=speed
	return positions,velocities

def normalize_not0(matrix,axis): #normalizes an entry in a matrix if its not 0
  if axis == 1:
    norm = np.linalg.norm(matrix, axis=1)
    norm[norm==0] = 1
    return matrix / norm[:, np.newaxis]
  elif axis == 2:
    norm = np.linalg.norm(matrix, axis=2)
    norm[norm==0] = 1
    return matrix / norm[:, :, np.newaxis]

def move_flock(flock, steps,parameters, initial_pos, initial_vel): #returns the final positions and velocities of the agents

	positions = initial_pos
	velocities = initial_vel

	r_tau = parameters["r_tau"]
	dt = parameters["dt"]
	max_turn = parameters["max_turning"]

	Width = parameters["Width"]
	Height = parameters["Height"]

	error_turn = parameters["error_turn"]
	speed = parameters["speed"]

	radii_vector = np.array([mate.radius for mate in flock])
	coh_tend = np.array([mate.cohesion for mate in flock])
	align_tend = np.array([mate.alignment for mate in flock])

	for ticks_num in range(steps):
		next_step = run_step(positions,velocities,r_tau,radii_vector,coh_tend,align_tend, dt, max_turn, Width, Height, error_turn,speed)
		positions = next_step[0] #Updates the positions
		velocities = next_step[1] #Updates the velocities

	return positions, velocities

def run_step(position_matrix,velocities,r_tau,radii_vector, coh_tend,align_tend,dt, max_turn, Width, Height, error_turn,speed):
  population = len(velocities)
  differences = position_matrix[:, np.newaxis, :] - position_matrix[np.newaxis, :, :] #calculate the difference in positions  
  distances = np.linalg.norm(differences, axis=2) #calculate the distances between every two points

  close_matrix = distances < radii_vector[:, np.newaxis] #check which points are close enough 
  np.fill_diagonal(close_matrix,False) #do not inclue itself! 
  close_matrix = close_matrix.astype(int) #change bool to int

  avoid_vector = avoid_calc(differences,distances,r_tau) #calculate the avoidance vectors
  cohesion_vector = cohesion_calc(differences,close_matrix) #calculate the cohesion vectors
  alignment_vector = alignment_calc(velocities,close_matrix) #calculate the alignment vectors

  new_dir = decide_dir(velocities,avoid_vector,cohesion_vector,alignment_vector,coh_tend,align_tend) #decide whether to avoid or to aggregate
  error = np.random.normal(0, error_turn, size=(population,2)) #copying error
  new_dir += error

  new_vel = turning(new_dir,velocities, max_turn)*speed #Place limits into the angle at which the mate can turn

  new_positions = position_matrix + new_vel*dt # p_n+1 = p_n + v_n * dt
  new_positions = limits(new_positions, Width, Height) #check limits

  return new_positions, new_vel

def avoid_calc(differences,distances,r_tau): #avoid other mates
  repulsion_matrix = distances <= r_tau #check which are too close to this boid 
  np.fill_diagonal(repulsion_matrix, False) #do not include itself!
  repulsion_matrix = repulsion_matrix.astype(int) #change bool to int

  eval_diff = differences*repulsion_matrix[:, :, np.newaxis] #Multiply to only have nonzero values if too close
  eval_diff = normalize_not0(eval_diff,axis=2) #Normalize those that are not 0

  avoid_vecs = np.sum(eval_diff, axis=1) #add all the vectors that avoid!
  avoid_vecs = normalize_not0(avoid_vecs,axis=1) #normalize such vectors

  return avoid_vecs

def cohesion_calc(differences, close_matrix):
  cohesion_diff = - differences #take the negative of the differences
  cohesion_diff = normalize_not0(cohesion_diff,axis=2)

  eval_diff = cohesion_diff*close_matrix[:, :, np.newaxis] #use only those that are close enough
  cohesion_vecs = np.sum(eval_diff, axis=1) #sum each row
  cohesion_vecs = normalize_not0(cohesion_vecs,axis=1) #normalize

  return cohesion_vecs

def alignment_calc(velocities, close_matrix):
  vel_calc = normalize_not0(velocities,axis=1) #turn into direction vectors
  align_vec = np.matmul(close_matrix,vel_calc) #use only those that are close enough (dot product)
  align_vec = normalize_not0(align_vec,axis=1) #normalize the sums
  return align_vec

def decide_dir(velocities,avoid_vector,cohesion_vector,alignment_vector,coh_tend,align_tend):

  result_vec = avoid_vector #start with the avoid_vectors first
  coh_tend = coh_tend[:,np.newaxis] #shape (N,1)
  align_tend = align_tend[:,np.newaxis] #shape (N,1)

  new_vel = coh_tend * cohesion_vector + align_tend * avoid_vector + (1-coh_tend-align_tend) * velocities 
  #calculate velocities given by cohesion and alignment

  norms = np.linalg.norm(avoid_vector,axis=1)
  zero_indices = np.where(norms==0)[0] #check where there is no repulsion
  result_vec[zero_indices] = new_vel[zero_indices] #substitute those with the cohesion, alignment, and velocities vectors
  result_vec = normalize_not0(result_vec,axis=1) #normalize the resultant vectors

  return result_vec

def turning(new_dir,velocities,max_turn):
  result_dir = new_dir #Start with the new direction (which is already normal)
  prev_dir = velocities / np.linalg.norm(velocities, axis=1)[:, np.newaxis] #get the previous direction vectors
  prev_angles = np.arctan2(velocities[:,1],velocities[:,0]) #calculate the angles of the new directions

  diff_angles = np.arccos(np.clip(np.sum(prev_dir * new_dir, axis=1), -1.0, 1.0)) #calculate the difference in angles for new and previous directions
  redirect_mask = diff_angles > max_turn #check which turns require a large angle turn

  cross_products = np.cross(prev_dir, new_dir) #check the direction of the angle turn
  rotation_signs = np.sign(cross_products)

  rotation_angles = rotation_signs * max_turn + prev_angles #calculate the angle of the rotated vectors (if turning angle is too large)

  cos_angles = np.cos(rotation_angles)
  sin_angles = np.sin(rotation_angles)
  rotation_vecs = np.stack((cos_angles,sin_angles),axis=-1) #get the resultant vectors of the maximum turn


  result_dir[redirect_mask] = rotation_vecs [redirect_mask] #change the resultant vectors only if the turn is too large
  return(result_dir)

def limits(positions, Width, Height): #Limits the positions
  positions[positions[:,0]<0,0] = Width
  positions[positions[:,0]>Width,0] = 0
  positions[positions[:,1]<0,1] = Height
  positions[positions[:,1]>Height,1] = 0
  return positions