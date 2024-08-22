import numpy as np

class UnionFind:
  def __init__(self, size):
    self.parent = list(range(size))
    self.rank = [0] * size

  def find(self, p):
    if self.parent[p] != p:
      self.parent[p] = self.find(self.parent[p])
    return self.parent[p]

  def union(self, p, q):
    rootP = self.find(p)
    rootQ = self.find(q)
    if rootP != rootQ:
      if self.rank[rootP] > self.rank[rootQ]:
        self.parent[rootQ] = rootP
      elif self.rank[rootP] < self.rank[rootQ]:
        self.parent[rootP] = rootQ
      else:
        self.parent[rootQ] = rootP
        self.rank[rootP] += 1

def form_groups(distances, group_radius, flockMates):
  n = len(flockMates)
  uf = UnionFind(n)
  for i in range(n):
    for j in range(i + 1, n):
      distance = distances[i][j]
      if distance < group_radius:
        uf.union(i, j)
  groups = {}
  for i in range(n):
    root = uf.find(i)
    if root not in groups:
      groups[root] = []
    groups[root].append(flockMates[i])
  return list(groups.values())

def joshi_fitness(flockMates, positions, params): 

  group_radius = params["group_radius"]
  benefit = params["benefit"]
  c_cop = params["c_cop"]
  c_coh = params["c_coh"]
  base_fitness = params["base_fitness"]

  differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # calculate the difference in positions  
  distances = np.linalg.norm(differences, axis=2)  # calculate the distances between every two points
  
  groups = form_groups(distances, group_radius, flockMates)  #list that contains the groups. Each group contains mates

  for group in groups:  # adds the payoff from all group interactions
    n_g = len(group)
    k_g = sum(mate.coop for mate in group)

    if n_g > 1:
      coop_ben = (k_g - 1) / (n_g - 1) * benefit - c_cop
      cheat_ben = k_g / (n_g - 1) * benefit
    else:
      coop_ben = -c_cop
      cheat_ben = 0

    for mate in group:
      cohesive_cost = mate.radius**2 * c_coh
      if mate.coop:
        mate.fitness = coop_ben - cohesive_cost
      else:
        mate.fitness = cheat_ben - cohesive_cost

  flock_fitness = np.array([mate.fitness for mate in flockMates], dtype=float)
  
  min_fit = np.min(flock_fitness)
  
  for mate in flockMates:
    mate.fitness -= min_fit
    mate.fitness += base_fitness