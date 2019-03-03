import random
import mlrose
import numpy as np
import time
def tsp(length, algo, coords):
    ones_fit = mlrose.TravellingSales(coords = coords)
    prob_fit = mlrose.TSPOpt(length=length, fitness_fn = ones_fit, maximize=False)

    if algo == 'rhc':
        start_time = time.time()
        best_state, best_fitness = mlrose.random_hill_climb(prob_fit, max_iters = 100000)
        end_time = time.time() - start_time
    elif algo == 'sa':
        start_time = time.time()
        best_state, best_fitness = mlrose.simulated_annealing(prob_fit, max_iters = 100000)
        end_time = time.time() - start_time
    elif algo == 'ga':
        start_time = time.time()
        best_state, best_fitness = mlrose.genetic_alg(prob_fit, max_iters = 100000)
        end_time = time.time() - start_time
    else:
        start_time = time.time()
        best_state, best_fitness = mlrose.mimic(prob_fit, max_iters = 100000)
        end_time = time.time() - start_time
    return best_fitness, end_time

def createNewCoord():
    return (random.randint(1,20), random.randint(1,20))

def runprob(prob):
    best_fitness_dict = {}
    for alg in ['mimic']:
        best_fitness_array = []
        times = []
        print(alg)
        for i in range(30, 101):
            if alg == 'mimic':
                print(i)
            if prob == 'ff':
                best_fitness, time = flipflop(i, alg)
                best_fitness_array.append(best_fitness)
                times.append(time)
            elif prob == 'fp':
                best_fitness, time = fourpeaks(i, alg)
                best_fitness_array.append(best_fitness)
                times.append(time)
            elif prob == 'ts':
                coords_list = []
                while len(coords_list) < i:
                    new_coord = createNewCoord()
                    if new_coord not in coords_list:
                        coords_list.append(new_coord)
                best_fitness, time = tsp(i, alg, coords_list)
                best_fitness_array.append(best_fitness)
                times.append(time)


        best_fitness_dict[alg] = {'fitness': best_fitness_array, 'times': times}
    return best_fitness_dict


results_tsp = runprob('ts')
plot_fitness(results_tsp)
plot_times(results_tsp)
