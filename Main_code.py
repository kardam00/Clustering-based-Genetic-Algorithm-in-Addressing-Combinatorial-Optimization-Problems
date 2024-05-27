
import os
import re
import sys
import math
import time
import pandas
import random
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from multiprocessing import Lock

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from collections import defaultdict
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

NFE = None

# Solutions
def GeneateSolution(n):
    x = list(np.random.randint(0, 2, size=n))
    return x

# Cost Function
def KnapsakCost(x, v, w, W):
    global NFE
    if NFE == None:
        NFE=0

    NFE += 1

    GanedValue = np.dot(v, x)
    LostValue = np.dot(v, (1-np.array(x)))
    GanedWeight = np.dot(w, x)
    LostWeight = np.dot(w, (1-np.array(x)))

    if GanedWeight > W:
      GanedValue = -np.inf
      violation = True
    else:
      violation = False

    sol = {"GanedValue": GanedValue,
           "LostValue": LostValue,
           "GanedWeight": GanedWeight,
           "LostWeight": LostWeight,
           "isFeasible": violation,
           "chromosome": x
    }

    return GanedValue, sol

# Sort the population and cost (based on the cost)
def pop_sort(p, c):
    li = []
    for i in range(len(c)):
        li.append([c[i],i])

    li.sort(reverse=True)
    sort_index = []

    for x in li:
        sort_index.append(x[1])

    positions, cost = [], []
    for i in sort_index:
        positions.append(p[i])
        cost.append(c[i])

    return positions, cost

# Select Method
def rouletteWheelSelection(p):
    r = random.random()
    c = np.cumsum(p)
    indexes = [
    index for index in range(len(c))
    if c[index] > r
    ]

    return indexes[0]

# Crossover
def singlePoint_crossover(x1, x2):
    index = int(np.random.randint(1, len(x1)-1))
    y1 = x1[:index] + x2[index:]
    y2 = x2[:index] + x1[index:]
    return y1, y2

def doublePoint_crossover(x1, x2):
    ind = random.sample(range(1, len(x1)-1), 2)
    index1 = min(ind)
    index2 = max(ind)

    y1 = x1[:index1] + x2[index1:index2] + x1[index2:]
    y2 = x2[:index1] + x1[index1:index2] + x2[index2:]

    return y1, y2

def uniform_crossover(x1, x2):
    alpha = list(np.random.randint(2, size=len(x1)))
    y1 = list(np.multiply(alpha, x1) + (1-np.array(alpha)) * np.array(x2))
    y2 = list(np.multiply(alpha, x2) + (1-np.array(alpha)) * np.array(x1))

    return y1, y2

def CrossOver(x1, x2):
    pSinglePoint = 0.1
    pDoublePoint = 0.2
    pUniform = 1 - pSinglePoint - pDoublePoint

    METHOD = rouletteWheelSelection([pSinglePoint, pDoublePoint, pUniform])

    if METHOD == 0:
      y1, y2 = singlePoint_crossover(x1, x2)
    elif METHOD == 1:
      y1, y2 = doublePoint_crossover(x1, x2)
    elif METHOD == 2:
      y1, y2 = uniform_crossover(x1, x2)

    return y1, y2

# Mutation
def singleSwap_Mutation(x):
  temp = 0
  index = int(np.random.randint(0, len(x)))
  y = x.copy()
  if index != len(y)-1:
    temp = y[index]
    y[index] = y[index + 1]
    y[index + 1] = temp
  else:
    temp = y[index]
    y[index] = y[0]
    y[0] = temp
  return y

def doubleSwap_Mutation(x):
  temp = 0
  index = random.sample(range(1, len(x)-1), 2)

  y = x.copy()
  temp = y[index[0]]
  y[index[0]] = y[index[1]]
  y[index[1]] = temp
  return y

def uniform_Mutation(x):
  index = int(np.random.randint(0, len(x)))
  y = x.copy()
  y[index] = 1-x[index]
  return y

def inverseSwap_Mutation(x):
  index1, index2 = sorted(random.sample(range(len(x)), 2))
  if index1 == index2:
    index2 = (index2 + 1) % len(x)  # Ensure index2 is different
  index1, index2 = sorted([index1, index2])
  y = x[:index1] + list(reversed(x[index1:index2])) + x[index2:]
  return y

def Mutation(x):
    pSingleSwap = 0.1
    pDoubleSwap = 0.2
    pInverseSwap = 0.2
    pUniform = 1 - pSingleSwap - pDoubleSwap - pInverseSwap

    METHOD = rouletteWheelSelection([pSingleSwap, pDoubleSwap, pInverseSwap, pUniform])

    if METHOD == 0:
      y = singleSwap_Mutation(x)
    elif METHOD == 1:
      y = doubleSwap_Mutation(x)
    elif METHOD == 2:
      y = inverseSwap_Mutation(x)
    elif METHOD == 3:
      y = uniform_Mutation(x)

    return y

# determine clusters for Kmeans
def determine_n_clusters(solutions, max_clusters_threshold=0.1):
    """
    Determine the number of clusters based on the characteristics of the current solutions.

    Parameters:
    - solutions (list): List of solutions where each solution is represented as a list or array.
    - max_clusters_threshold (float): Maximum percentage of solutions allowed per cluster.

    Returns:
    - n_clusters (int): Number of clusters to be used in K-means clustering.
    """
    # Calculate the maximum number of clusters based on the threshold
    max_clusters = int(len(solutions) * max_clusters_threshold)

    # Ensure a minimum number of clusters
    min_clusters = 2
    max_clusters = max(min_clusters, max_clusters)

    # Determine the number of clusters based on the maximum allowed
    n_clusters = min(max_clusters, len(solutions))

    return n_clusters


def refine_clusters(clusters):
    refined_clusters = {}
    assigned_objects = set()

    for extent, intent in clusters.items():
        unique_objects = [obj for obj in extent if obj not in assigned_objects]
        if len(unique_objects) > 1:
            refined_clusters[tuple(unique_objects)] = intent
            assigned_objects.update(unique_objects)

    return refined_clusters

def perform_concept_clustering(population):
    data = pandas.DataFrame(population)
    binary_data = pandas.get_dummies(data, drop_first=True)
    binary_data = binary_data.astype(bool)
    binary_data.index = binary_data.index.astype(str)
    binary_data.columns = binary_data.columns.astype(str)

    # Generate Formal Context and Concept Lattice
    K = FormalContext.from_pandas(binary_data)
    L = ConceptLattice.from_context(K, algo='Sofia', L_max=100)

    # Initialize dictionary to store cluster assignments
    clusters = defaultdict(list)

    # Convert concepts to dictionary format
    concepts = {tuple(c.extent): tuple(c.intent) for c in L}

    # Step 1: Remove concepts based on specified conditions
    concepts = {extent: intent for extent, intent in concepts.items() if 
                len(extent) != len(binary_data.columns) and  # Remove concepts with extent length equal to attribute length
                len(extent) > 1 and                           # Remove concepts with extent length less than or equal to 1
                len(extent) <= len(binary_data) * 0.4}       # Remove concepts with extent length greater than 40% of total objects

    # Step 2: Initialize clusters with concepts having maximum intent size
    while concepts:
        max_extent = max(concepts.keys(), key=lambda x: len(x))
        if 2 <= len(max_extent) <= len(binary_data) * 0.4:
            clusters[max_extent] = list(concepts[max_extent])
        del concepts[max_extent]

    # Step 3: Iterate over remaining concepts to assign to clusters
    objects_left = set(binary_data.index)
    noise_objects = []
    for extent, intent in concepts.items():
        intersected_objects = set(extent).intersection(objects_left)
        if intersected_objects:
            new_cluster = list(intersected_objects)
            if 2 <= len(new_cluster) <= len(binary_data) * 0.4:
                clusters[tuple(new_cluster)] = list(extent)
            objects_left -= intersected_objects
        else:
            noise_objects.extend(list(extent))

    # Step 4: Create noise cluster
    if noise_objects:
        clusters[-1] = noise_objects

    # Step 5: Refine clusters to remove duplicate objects and clusters with fewer than two solutions
    refined_clusters = refine_clusters(clusters)

    # Formatting the clusters with solutions
    formatted_clusters = {}
    for i, (extent, intent) in enumerate(refined_clusters.items()):
        formatted_extent = [[int(value) for value in binary_data.loc[obj]] for obj in extent]
        formatted_clusters[i] = formatted_extent

    return formatted_clusters


# Run Functions
def run_GA(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
  filename = f'{alg}_Iterations_{_}.txt'
  with open(filename, 'a') as f:
    f.write(f'Run: {_}\n')

  pc = 0.7
  nc = math.ceil(pc * nPop)

  pm = 0.2
  nm = math.ceil(pm * nPop)

  bestSolutionG = []
  bestCostsG = []
  timeG = []
  nfe = []
  
  start_time = 0
  end_time = 0

  pop = Ipop.copy()
  costs = Icosts.copy()

  for it in range(1, maxIt):
    start_time = time.time()

    # Crossover
    popc, popc_cost = [], []
    for k in range(1, int(nc/2)):
      rand1, rand2 = [int(np.random.randint(nPop)) for _ in range(2)]
      p1 = pop[rand1]
      p2 = pop[rand2]

      y1, y2 = CrossOver(p1, p2)

      popc.append(y1)
      popc.append(y2)

      c1, _ = KnapsakCost(y1, v, w, W)
      c2, _ = KnapsakCost(y2, v, w, W)

      popc_cost.append(c1)
      popc_cost.append(c2)

    # Mutation
    popm, popm_cost = [], []
    for k in range(1, nm):
      rand = int(np.random.randint(nPop))
      p = pop[rand]
      mutated_p = Mutation(p)

      popm.append(mutated_p)

      c, _ = KnapsakCost(mutated_p, v, w, W)
      popm_cost.append(c)


    # After generating new populations, append them to pop
    pop = pop + popm + popc
    costs = costs + popm_cost + popc_cost

    pop, costs = pop_sort(pop, costs)

    pop = pop[:nPop]
    costs = costs[:nPop]

    bestSolutionG.append(pop[0])
    bestCostsG.append(costs[0])
    nfe.append(NFE)

    end_time = time.time()
    timeG.append(end_time - start_time)

    if it % 10 == 0:
      with lock:
        with open(filename, 'a') as f:
          f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsG[it-1]}, Time = {end_time - start_time} seconds')
      
  return bestSolutionG, bestCostsG, timeG,

def run_DBSCAN1(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
  
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')

    n_crossover = math.ceil(int(0.7 * nPop))
    n_mutation = math.ceil(int(0.2 * nPop))
    n_dbscan = math.ceil(int(0.2 * nPop))

    bestSolutionD1 = []
    bestCostsD1 = []
    timeD1 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        popc, popc_cost = [], []
        popm, popm_cost = [], []
        popd, popd_cost = [], []

        for k in range(1, math.ceil(n_crossover//2)):
            rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

            p1 = pop[rand1]
            p2 = pop[rand2]

            #if p1 == p2:
            #    rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

            #    p1 = pop[rand1]
            #    p2 = pop[rand2]

            #else:
            #    continue

            y1, y2 = CrossOver(p1, p2)
            popc.append(y1)
            popc.append(y2)

            c1, _ = KnapsakCost(y1, v, w, W)
            c2, _ = KnapsakCost(y2, v, w, W)

            popc_cost.append(c1)
            popc_cost.append(c2)
            

        for k in range(1, math.ceil(n_mutation)):
            rand = int(np.random.randint(nPop))
            p = pop[rand]
            mutated_p = Mutation(p)

            popm.append(mutated_p)
            c, _ = KnapsakCost(mutated_p, v, w, W)
            popm_cost.append(c)
            

        solutions = pop

        def euclidean_distance(solution1, solution2):
            return np.abs(np.linalg.norm(np.array(solution1) - np.array(solution2)))


        distances = np.zeros((len(solutions), len(solutions)))
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                distances[i, j] = euclidean_distance(solutions[i], solutions[j])
        
        eps = max(np.mean(distances), 0.1)
        dbscan = DBSCAN(eps = eps, min_samples=2, metric='precomputed')

        labels = dbscan.fit_predict(distances)

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(solutions[i])

        # if it % 10 == 0:
        #   with lock:
        #     with open(filename, 'a') as f:
        #         for cluster_label, cluster_solutions in clusters.items():
        #             f.write(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        # Iterate for some iterations
        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            # CrossOver
            for iteration in range(1, math.ceil(n_dbscan // 2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)
                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popd.append(y1)
                popd_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popd.append(y2)
                popd_cost.append(c2)
                

        elif n_clusters > 1:
            # CrossOver
            for iteration in range(1, math.ceil(n_dbscan // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popd.append(y1)
                popd_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popd.append(y2)
                popd_cost.append(c2)

        #pop.extend(popc)
        #pop.extend(popm)
        #pop.extend(popd)

        #costs.extend(popc_cost)
        #costs.extend(popm_cost)
        #costs.extend(popd_cost)

        pop = pop + popm + popc
        costs = costs + popm_cost + popc_cost
        
        pop, costs = pop_sort(pop, costs)
        
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionD1.append(pop[0])
        bestCostsD1.append(costs[0])
        nfe.append(NFE)

        end_time = time.time()
        timeD1.append(end_time - start_time)

        if it % 10 == 0:
          with lock:
            with open(filename, 'a') as f:
                f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsD1[it-1]}, Time = {end_time - start_time} seconds')
      
    return bestSolutionD1, bestCostsD1, timeD1

def run_DBSCAN2(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
  
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')
  
    bestSolutionD2 = []
    bestCostsD2 = []
    timeD2 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        nc = math.ceil(int(0.7 * nPop))
        nm = math.ceil(int(0.2 * nPop))

        popc = []
        popm = []
        popc_cost = []
        popm_cost = []

        solutions = pop

        def euclidean_distance(solution1, solution2):
            return np.abs(np.linalg.norm(np.array(solution1) - np.array(solution2)))


        distances = np.zeros((len(solutions), len(solutions)))
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                distances[i, j] = euclidean_distance(solutions[i], solutions[j])

	
        dbscan = DBSCAN(eps = max(np.mean(distances), 0.1), min_samples=2, metric='precomputed')

        labels = dbscan.fit_predict(distances)

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(solutions[i])

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 f.write(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)


        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            # CrossOver
            for iteration in range(math.ceil(nc//2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)
            


        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(nc // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                cluster_indice = random.sample(range(n_clusters), 1)

                cluster_label = clusters_list[cluster_indice[0]]

                cluster = clusters[cluster_label]

                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)

        pop.extend(popc)
        pop.extend(popm)

        costs.extend(popc_cost)
        costs.extend(popm_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionD2.append(pop[0])
        bestCostsD2.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeD2.append(end_time - start_time)

        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsD2[it-1]}, Time = {end_time - start_time} seconds')
            
    return bestSolutionD2, bestCostsD2, timeD2

def run_KMEANS1(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')

    bestSolutionK1 = []
    bestCostsK1 = []
    timeK1 = []
    nfe = []

    start_time = 0
    end_time = 0


    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        popc, popc_cost = [], []
        popm, popm_cost = [], []
        popk, popk_cost = [], []

        n_crossover = math.ceil(int(0.7 * nPop))
        n_mutation = math.ceil(int(0.2 * nPop))
        n_kmeans = math.ceil(int(0.2 * nPop))

        for k in range(math.ceil(n_crossover//2)):
            rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

            p1 = pop[rand1]
            p2 = pop[rand2]

            if p1 == p2:
                rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

                p1 = pop[rand1]
                p2 = pop[rand2]

            else:
                continue

            y1, y2 = CrossOver(p1, p2)
            popc.append(y1)
            popc.append(y2)

            c, _ = KnapsakCost(y1, v, w, W)
            popc_cost.append(c)
            
            c, _ = KnapsakCost(y2, v, w, W)
            popc_cost.append(c)
            

        for k in range(math.ceil(n_mutation)):
            rand = int(np.random.randint(n_mutation))

            p = pop[rand]

            popm.append(Mutation(p))
            c, _ = KnapsakCost(popm[-1], v, w, W)
            popm_cost.append(c)
            
        # Determine the number of clusters for this iteration
        n_clusters = determine_n_clusters(pop)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # Specify the number of clusters
        kmeans.fit(pop)
        labels = kmeans.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pop[i])

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 f.write(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        # Iterate for some iterations
        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            # CrossOver
            for iteration in range(math.ceil(n_kmeans // 2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)
                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popk.append(y1)
                popk_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popk.append(y2)
                popk_cost.append(c2)
                

        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(n_kmeans // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                  obj1 = random.choice(cluster1)
                  obj2 = random.choice(cluster2)

                else:
                  continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popk.append(y1)
                popk_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popk.append(y2)
                popk_cost.append(c2)

        pop.extend(popc)
        pop.extend(popm)
        pop.extend(popk)

        costs.extend(popc_cost)
        costs.extend(popm_cost)
        costs.extend(popk_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionK1.append(pop[0])
        bestCostsK1.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeK1.append(end_time - start_time)
        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsK1[it-1]}, Time = {end_time - start_time} seconds')
            
    return bestSolutionK1, bestCostsK1, timeK1

def run_KMEANS2(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
    
  filename = f'{alg}_Iterations_{_}.txt'
  with open(filename, 'a') as f:
    f.write(f'Run: {_}\n')
    
    bestSolutionK2 = []
    bestCostsK2 = []
    timeK2 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        nc = math.ceil(int(0.7 * nPop))
        nm = math.ceil(int(0.2 * nPop))

        popc = []
        popm = []
        popc_cost = []
        popm_cost = []

        solutions = pop

        # Determine the number of clusters for this iteration
        n_clusters = determine_n_clusters(solutions)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(solutions)
        labels = kmeans.labels_
        clusters = {}

        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(solutions[i])

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 print(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        #print('clusters: ', clusters)
        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)


        if n_clusters == 1:
          cluster_label = clusters_list[0]
          cluster = clusters[cluster_label]

          # CrossOver
          for iteration in range(math.ceil(nc//2)):
            obj1 = random.choice(cluster)
            obj2 = random.choice(cluster)

            if obj1 == obj2:
              obj1 = random.choice(cluster)
              obj2 = random.choice(cluster)

            else:
              continue

            # Perform crossover operation
            y1, y2 = CrossOver(obj1, obj2)

            # Calculate the cost for offspring solutions
            c1, _ = KnapsakCost(y1, v, w, W)
            popc.append(y1)
            popc_cost.append(c1)

            c2, _ = KnapsakCost(y2, v, w, W)
            popc.append(y2)
            popc_cost.append(c2)

          # Mutation
          for iteration in range(math.ceil(nm)):
            p = random.choice(cluster)

            popm.append(Mutation(p))
            c, _ = KnapsakCost(popm[-1], v, w, W)
            popm_cost.append(c)
            


        elif n_clusters > 1:
          # CrossOver
          for iteration in range(math.ceil(nc // 2)):
            # Choose random clusters and objects
            cluster_indices = random.sample(range(n_clusters), 2)

            cluster1_label = clusters_list[cluster_indices[0]]
            cluster2_label = clusters_list[cluster_indices[1]]

            cluster1 = clusters[cluster1_label]
            cluster2 = clusters[cluster2_label]

            obj1 = random.choice(cluster1)
            obj2 = random.choice(cluster2)

            if obj1 == obj2:
              obj1 = random.choice(cluster1)
              obj2 = random.choice(cluster2)

            else:
              continue

            # Perform crossover operation
            y1, y2 = CrossOver(obj1, obj2)

            # Calculate the cost for offspring solutions
            c1, _ = KnapsakCost(y1, v, w, W)
            popc.append(y1)
            popc_cost.append(c1)

            c2, _ = KnapsakCost(y2, v, w, W)
            popc.append(y2)
            popc_cost.append(c2)


          # Mutation
          for iteration in range(math.ceil(nm)):
            cluster_indice = random.sample(range(n_clusters), 1)

            cluster_label = clusters_list[cluster_indice[0]]

            cluster = clusters[cluster_label]

            p = random.choice(cluster)

            popm.append(Mutation(p))
            c, _ = KnapsakCost(popm[-1], v, w, W)
            popm_cost.append(c)

        pop.extend(popc)
        pop.extend(popm)

        costs.extend(popc_cost)
        costs.extend(popm_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionK2.append(pop[0])
        bestCostsK2.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeK2.append(end_time - start_time)
        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsK2[it-1]}, Time = {end_time - start_time} seconds')
            
    return bestSolutionK2, bestCostsK2, timeK2

def run_AP1(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
    
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')
    
    bestSolutionAP1 = []
    bestCostsAP1 = []
    timeAP1 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        popc, popc_cost = [], []
        popm, popm_cost = [], []
        popap, popap_cost = [], []

        n_crossover = math.ceil(int(0.7 * nPop))
        n_mutation = math.ceil(int(0.2 * nPop))
        n_affinity_propagation = math.ceil(int(0.2 * nPop))

        for k in range(math.ceil(n_crossover // 2)):
            rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

            p1 = pop[rand1]
            p2 = pop[rand2]

            if p1 == p2:
                rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

                p1 = pop[rand1]
                p2 = pop[rand2]

            else:
                continue

            y1, y2 = CrossOver(p1, p2)
            popc.append(y1)
            popc.append(y2)

            c, _ = KnapsakCost(y1, v, w, W)
            popc_cost.append(c)
            
            c, _ = KnapsakCost(y2, v, w, W)
            popc_cost.append(c)
            

        for k in range(math.ceil(n_mutation)):
            rand = int(np.random.randint(nPop))

            p = pop[rand]

            popm.append(Mutation(p))
            c, _ = KnapsakCost(popm[-1], v, w, W)
            popm_cost.append(c)
            
        # Perform Affinity Propagation clustering
        affinity_propagation = AffinityPropagation(damping=0.9, preference=None, max_iter=1000, convergence_iter=15, random_state=0)
        affinity_propagation.fit(pop)  # Fit the clustering algorithm
        cluster_centers_indices = affinity_propagation.cluster_centers_indices_
        n_clusters = len(cluster_centers_indices) if cluster_centers_indices is not None else 0

        clusters = {}
        for i, label in enumerate(affinity_propagation.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pop[i])

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 print(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        # Iterate for some iterations
        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]
            
            # CrossOver
            for iteration in range(math.ceil(n_affinity_propagation // 2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)
                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popap.append(y1)
                popap_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popap.append(y2)
                popap_cost.append(c2)
                
        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(n_affinity_propagation // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popap.append(y1)
                popap_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popap.append(y2)
                popap_cost.append(c2)

        pop.extend(popc)
        pop.extend(popm)
        pop.extend(popap)

        costs.extend(popc_cost)
        costs.extend(popm_cost)
        costs.extend(popap_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionAP1.append(pop[0])
        bestCostsAP1.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeAP1.append(end_time - start_time)

        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it}: NFE = {nfe[-1]}, Best Cost = {bestCostsAP1[it - 1]}, Time = {end_time - start_time} seconds')

    return bestSolutionAP1, bestCostsAP1, timeAP1

def run_AP2(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
  
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')

    bestSolutionAP2 = []
    bestCostsAP2 = []
    timeAP2 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        nc = math.ceil(int(0.7 * nPop))
        nm = math.ceil(int(0.2 * nPop))

        popc = []
        popm = []
        popc_cost = []
        popm_cost = []

        solutions = pop

        # Perform Affinity Propagation clustering
        affinity_propagation = AffinityPropagation(damping=0.9, preference=None, max_iter=1000, convergence_iter=15, random_state=0)
        affinity_propagation.fit(solutions)
        cluster_centers_indices = affinity_propagation.cluster_centers_indices_
        n_clusters = len(cluster_centers_indices) if cluster_centers_indices is not None else 0

        clusters = {}
        for i, label in enumerate(affinity_propagation.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(solutions[i])

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 print(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)


        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            # CrossOver
            for iteration in range(math.ceil(nc//2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)
            


        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(nc // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                cluster_indice = random.sample(range(n_clusters), 1)

                cluster_label = clusters_list[cluster_indice[0]]

                cluster = clusters[cluster_label]

                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)

        pop.extend(popc)
        pop.extend(popm)

        costs.extend(popc_cost)
        costs.extend(popm_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionAP2.append(pop[0])
        bestCostsAP2.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeAP2.append(end_time - start_time)

        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it}: NFE = {nfe[-1]}, Best Cost = {bestCostsAP2[it-1]}, Time = {end_time - start_time} seconds')

    return bestSolutionAP2, bestCostsAP2, timeAP2

def run_MS1(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):

    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')
    
    n_crossover = math.ceil(int(0.7 * nPop))
    n_mutation = math.ceil(int(0.2 * nPop))
    n_mean_shift = math.ceil(int(0.2 * nPop))

    bestSolutionMS1 = []
    bestCostsMS1 = []
    timeMS1 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
    
        start_time = time.time()

        popc, popc_cost = [], []
        popm, popm_cost = [], []
        popms, popms_cost = [], []

        for k in range(1, math.ceil(n_crossover // 2)):
            rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

            p1 = pop[rand1]
            p2 = pop[rand2]

            y1, y2 = CrossOver(p1, p2)
            popc.append(y1)
            popc.append(y2)

            c1, _ = KnapsakCost(y1, v, w, W)
            c2, _ = KnapsakCost(y2, v, w, W)

            popc_cost.append(c1)
            popc_cost.append(c2)
            
        for k in range(1, math.ceil(n_mutation)):
            rand = int(np.random.randint(nPop))
            p = pop[rand]
            mutated_p = Mutation(p)

            popm.append(mutated_p)
            c, _ = KnapsakCost(mutated_p, v, w, W)
            popm_cost.append(c)
            
        # Perform Mean Shift clustering
        bandwidth = estimate_bandwidth(pop, quantile=0.2, n_samples=500)
        min_bandwidth = 0.1
        bandwidth = max(bandwidth, min_bandwidth)  # Ensure bandwidth is not zero
        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit(pop)  # Fit the clustering algorithm
        cluster_centers = mean_shift.cluster_centers_
        n_clusters = len(cluster_centers)

        clusters = {}
        for i, label in enumerate(mean_shift.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pop[i])

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 print(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        # Iterate for some iterations
        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            #CrossOver
            for iteration in range(1, math.ceil(n_mean_shift // 2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)
                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popms.append(y1)
                popms_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popms.append(y2)
                popms_cost.append(c2)

        elif n_clusters > 1:
            # CrossOver
            for iteration in range(1, math.ceil(n_mean_shift // 2)):
        
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popms.append(y1)
                popms_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popms.append(y2)
                popms_cost.append(c2)
            

        #pop.extend(popc)
        #pop.extend(popm)
        #pop.extend(popms)

        #costs.extend(popc_cost)
        #costs.extend(popm_cost)
        #costs.extend(popms_cost)

        pop = pop + popm + popc
        costs = costs + popm_cost + popc_cost

        pop, costs = pop_sort(pop, costs)
        
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionMS1.append(pop[0])
        bestCostsMS1.append(costs[0])
        nfe.append(NFE)

        end_time = time.time()
        timeMS1.append(end_time - start_time)

        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it}: NFE = {nfe[-1]}, Best Cost = {bestCostsMS1[it - 1]}, Time = {end_time - start_time} seconds')

    return bestSolutionMS1, bestCostsMS1, timeMS1

def run_MS2(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')

    bestSolutionMS2 = []
    bestCostsMS2 = []
    timeMS2 = []
    nfe = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        nc = math.ceil(int(0.7 * nPop))
        nm = math.ceil(int(0.2 * nPop))

        popc = []
        popm = []
        popc_cost = []
        popm_cost = []

        solutions = pop

        # Perform Mean Shift clustering
        bandwidth = estimate_bandwidth(pop, quantile=0.2, n_samples=500)
        min_bandwidth = 0.1
        bandwidth = max(bandwidth, min_bandwidth)  # Ensure bandwidth is not zero
        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit(solutions)  # Fit the clustering algorithm
        cluster_centers = mean_shift.cluster_centers_
        n_clusters = len(cluster_centers)

        clusters = {}
        for i, label in enumerate(mean_shift.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(solutions[i])
        
        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        # Iterate for some iterations
        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            for iteration in range(1, math.ceil(nc//2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)
            
        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(nc // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                cluster_indice = random.sample(range(n_clusters), 1)

                cluster_label = clusters_list[cluster_indice[0]]

                cluster = clusters[cluster_label]

                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)

        pop.extend(popc)
        pop.extend(popm)

        costs.extend(popc_cost)
        costs.extend(popm_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionMS2.append(pop[0])
        bestCostsMS2.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeMS2.append(end_time - start_time)

        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsMS2[it-1]}, Time = {end_time - start_time} seconds')
        
    return bestSolutionMS2, bestCostsMS2, timeMS2

def run_FCA1(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
  
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')

    bestSolutionF1 = []
    bestCostsF1 = []
    timeF1 = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        popc, popc_cost = [], []
        popm, popm_cost = [], []
        popf, popf_cost = [], []

        n_crossover = math.ceil(int(0.7 * nPop))
        n_mutation = math.ceil(int(0.2 * nPop))
        n_fca = math.ceil(int(0.2 * nPop))

        for k in range(math.ceil(n_crossover//2)):
            rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

            p1 = pop[rand1]
            p2 = pop[rand2]

            if p1 == p2:
                rand1, rand2 =  [int(np.random.randint(nPop)) for _ in range(2)]

                p1 = pop[rand1]
                p2 = pop[rand2]

            else:
                continue

            y1, y2 = CrossOver(p1, p2)
            popc.append(y1)
            popc.append(y2)

            c, _ = KnapsakCost(y1, v, w, W)
            popc_cost.append(c)
            
            c, _ = KnapsakCost(y2, v, w, W)
            popc_cost.append(c)
            

        for k in range(math.ceil(n_mutation)):
            rand = int(np.random.randint(n_mutation))

            p = pop[rand]

            popm.append(Mutation(p))
            c, _ = KnapsakCost(popm[-1], v, w, W)
            popm_cost.append(c)
            
        clusters = perform_concept_clustering(pop)

        # if it % 10 == 0:
        #   with lock:
        #     with open(filename, 'a') as f:
        #         for cluster_label, cluster_solutions in clusters.items():
        #             f.write(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        #print('clusters: ', clusters)
        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        # Iterate for some iterations
        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            # CrossOver
            for iteration in range(math.ceil(n_fca // 2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)
                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popf.append(y1)
                popf_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popf.append(y2)
                popf_cost.append(c2)

        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(n_fca // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popf.append(y1)
                popf_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popf.append(y2)
                popf_cost.append(c2)

        pop.extend(popc)
        pop.extend(popm)
        pop.extend(popf)

        costs.extend(popc_cost)
        costs.extend(popm_cost)
        costs.extend(popf_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]


        bestSolutionF1.append(pop[0])
        bestCostsF1.append(costs[0])


        nfe.append(NFE)

        end_time = time.time()
        timeF1.append(end_time - start_time)
        if it % 10 == 0:
          with lock:
            with open(filename, 'a') as f:
                f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsF1[it-1]}, Time = {end_time - start_time} seconds')
      
    return bestSolutionF1, bestCostsF1, timeF1

def run_FCA2(_, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W):
  
    filename = f'{alg}_Iterations_{_}.txt'
    with open(filename, 'a') as f:
        f.write(f'Run: {_}\n')
  
    bestSolutionF2 = []
    bestCostsF2 = []
    timeF2 = []

    start_time = 0
    end_time = 0

    pop = Ipop.copy()
    costs = Icosts.copy()

    # Main Loop
    for it in range(1, maxIt):
        start_time = time.time()

        nc = math.ceil(int(0.7 * nPop))
        nm = math.ceil(int(0.2 * nPop))

        popc = []
        popm = []
        popc_cost = []
        popm_cost = []

        clusters = perform_concept_clustering(pop)

        # if it % 10 == 0:
        #     with lock:
        #         with open(filename, 'a') as f:
        #             for cluster_label, cluster_solutions in clusters.items():
        #                 f.write(f"\nCluster {cluster_label}: {len(cluster_solutions)} solutions")

        #print('clusters: ', clusters)
        clusters_list = list(clusters.keys())
        n_clusters = len(clusters_list)

        if n_clusters == 1:
            cluster_label = clusters_list[0]
            cluster = clusters[cluster_label]

            # CrossOver
            for iteration in range(math.ceil(nc//2)):
                obj1 = random.choice(cluster)
                obj2 = random.choice(cluster)

                if obj1 == obj2:
                    obj1 = random.choice(cluster)
                    obj2 = random.choice(cluster)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)
            
        elif n_clusters > 1:
            # CrossOver
            for iteration in range(math.ceil(nc // 2)):
                # Choose random clusters and objects
                cluster_indices = random.sample(range(n_clusters), 2)

                cluster1_label = clusters_list[cluster_indices[0]]
                cluster2_label = clusters_list[cluster_indices[1]]

                cluster1 = clusters[cluster1_label]
                cluster2 = clusters[cluster2_label]

                obj1 = random.choice(cluster1)
                obj2 = random.choice(cluster2)

                if obj1 == obj2:
                    obj1 = random.choice(cluster1)
                    obj2 = random.choice(cluster2)

                else:
                    continue

                # Perform crossover operation
                y1, y2 = CrossOver(obj1, obj2)

                # Calculate the cost for offspring solutions
                c1, _ = KnapsakCost(y1, v, w, W)
                popc.append(y1)
                popc_cost.append(c1)

                c2, _ = KnapsakCost(y2, v, w, W)
                popc.append(y2)
                popc_cost.append(c2)

            # Mutation
            for iteration in range(math.ceil(nm)):
                cluster_indice = random.sample(range(n_clusters), 1)

                cluster_label = clusters_list[cluster_indice[0]]

                cluster = clusters[cluster_label]

                p = random.choice(cluster)

                popm.append(Mutation(p))
                c, _ = KnapsakCost(popm[-1], v, w, W)
                popm_cost.append(c)
                

        pop.extend(popc)
        pop.extend(popm)

        costs.extend(popc_cost)
        costs.extend(popm_cost)

        pop, costs = pop_sort(pop, costs)
        pop = pop[:nPop]
        costs = costs[:nPop]

        bestSolutionF2.append(pop[0])
        bestCostsF2.append(costs[0])

        nfe.append(NFE)

        end_time = time.time()
        timeF2.append(end_time - start_time)

        if it % 10 == 0:
            with lock:
                with open(filename, 'a') as f:
                    f.write(f'\nIteration {it} : NFE = {nfe[-1]}, Best Cost = {bestCostsF2[it-1]}, Time = {end_time - start_time} seconds')
            
    return bestSolutionF2, bestCostsF2, timeF2

def run_process(args):
  process_num, Ipop, Icosts, nPop, maxIt, alg, v, w, W = args
  lock = Lock()
  
  if alg == 'GA': 
    bestSolution, bestCosts, besttime = run_GA(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'DBSCAN1': 
    bestSolution, bestCosts, besttime = run_DBSCAN1(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'DBSCAN2': 
    bestSolution, bestCosts, besttime = run_DBSCAN2(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'KMEANS1': 
    bestSolution, bestCosts, besttime = run_KMEANS1(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'KMEANS2': 
    bestSolution, bestCosts, besttime = run_KMEANS2(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'AP1': 
    bestSolution, bestCosts, besttime = run_AP1(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'AP2': 
    bestSolution, bestCosts, besttime = run_AP2(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'MS1': 
    bestSolution, bestCosts, besttime = run_MS1(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'MS2': 
    bestSolution, bestCosts, besttime = run_MS2(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'FCA1': 
    bestSolution, bestCosts, besttime = run_FCA1(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  elif alg == 'FCA2': 
    bestSolution, bestCosts, besttime = run_FCA2(process_num, alg, Ipop, Icosts, nPop, maxIt, lock, v, w, W)
    return bestSolution, bestCosts, besttime, process_num
  else:
    print('No function given')
    return None, None, None, None

def merge_iteration_files(alg, processes):
    with open(f'{alg}_Iterations.txt', 'w') as merged_file:
        for i in range(processes):
            filename = f'{alg}_Iterations_{i}.txt'
            with open(filename, 'r') as f:
                merged_file.write(f.read())
            os.remove(filename)


if __name__ == '__main__':
    
    n_samples_arr = [100, 300, 500, 800]  # number of Genes
    all_maxIt = [2000]  # number of iterations
    R = 1000  # range of values (v and w)
    nPop = 200  # number of chromosomes
    processes = 5 # number of runs
    algorithms = ['GA', 'DBSCAN1', 'DBSCAN2', 'KMEANS1', 'KMEANS2', 'AP1', 'AP2', 'FCA1', 'FCA2','MS1', 'MS2'] 
    
    for n_samples in n_samples_arr:
        sys.stdout = open(f'{n_samples}_Initial_Population.txt', 'w')
        v = list(np.random.randint(1, R, size=n_samples)) 
        w = list(np.random.randint(1, R, size=n_samples))     

        W = math.ceil(sum(w) * 0.75)
        n = len(v)
        nVar = len(w)
            
        NFE = 0

        Ipop, Icosts, Isol = [], [], []
        for i in range(nPop):
            Ipop.append(GeneateSolution(nVar))
            c, s = KnapsakCost(Ipop[i], v, w, W)
            Icosts.append(c)
            Isol.append(s)

        Ipop, Icosts = pop_sort(Ipop, Icosts)

        nfe = [NFE]

        print('n_samples: ', n_samples)
        print('v: ', v)
        print('w: ', w)
        print('W: ', W)
        print('Maximum v = ', sum(v))
        print('Maximum w= ', sum(w))
        print('population size: ', len(Ipop), len(Ipop[0]))
        print('Initial population: ', Ipop)
        print('Initial costs: ', Icosts)
        
        sys.stdout.close()
        sys.stdout = sys.__stdout__

        all_output_data = []

        for maxIt in all_maxIt:
            
            #print(f'\nmaxIt: {maxIt}')
            output_data = []
            all_best_solutions = []
            all_best_costs = []
            all_times = []

            # Second table construction
            df_second_cost = pandas.DataFrame()
            df_second_time = pandas.DataFrame()

            for alg in algorithms:

                sys.stdout = open(f'output.txt', 'a')
                df_first_cost = pandas.DataFrame()
                df_first_time = pandas.DataFrame()

                #print(f'\n{alg}')
                best_solutions = []
                best_costs = []
                times = []

                args = [(i, Ipop, Icosts, nPop, maxIt, alg, v, w, W) for i in range(processes)]

                #file_name = alg + '_Iterations.txt'
                #with open(file_name, 'a') as f:
                #    print('\nmaxIT: ', maxIt, file=f)
                with Pool(processes) as pool:
                    results = pool.map(run_process, args)
                        
                for result in results:
                    bestSolution, bestCosts, besttime, process_num = result
                    best_solutions.append(bestSolution)
                    best_costs.append(bestCosts)
                    times.append(besttime)
                
                # Transpose and convert to numpy array for consistent data manipulation
                best_costs_array = np.transpose(np.array(best_costs))
                best_time_array = np.transpose(np.array(times))

                # Store best_costs_array in df_first_cost
                for idx, col_data in enumerate(best_costs_array.T):
                    clm = f'run_{idx + 1}'
                    df_first_cost[clm] = col_data

                # Store best_time_array in df_first_time
                for idx, col_data in enumerate(best_time_array.T):
                    clm = f'run_{idx + 1}'
                    df_first_time[clm] = col_data

                merge_iteration_files(alg, processes)


                # Store first table in CSV file
                print('df_first_cost\n', df_first_cost)
                df_first_cost.to_csv(
                    f'{alg}_nsample_{n_samples}_maxIt_{maxIt}_cost.csv', index_label='Iterations')

                # Store first table in CSV file
                print('df_first_time\n', df_first_time)
                df_first_time.to_csv(
                    f'{alg}_nsample_{n_samples}_maxIt_{maxIt}_time.csv', index_label='Iterations')

                df_second_cost[alg] = df_first_cost.iloc[-1]  
                df_second_time[alg] = df_first_time.iloc[-1]  
                
                        
            # Store second table in CSV file
            print('df_second_cost\n', df_second_cost)
            df_second_cost.to_csv(f'nsample_{n_samples}_maxIt_{maxIt}_combined_cost.csv', index_label='Runs')

            print('df_second_time\n', df_second_time)
            df_second_time.to_csv(f'nsample_{n_samples}_maxIt_{maxIt}_combined_time.csv', index_label='Runs')

            sys.stdout.close()
            sys.stdout = sys.__stdout__
            
