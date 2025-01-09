import random
import numpy as np
import torch
import torch.nn.functional as F
from deap import base, creator, tools
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)

from vae_metric.metric import Metric

# --- 1. Define the fitness and individual ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize probability
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

IMAGE_HEIGHT = 128 // 2
IMAGE_WIDTH = 128 // 2
CHANNELS = 3

# Function to create a random individual (image) of shape (H, W, C)
def create_individual():
    # Pixel values in [0,1] (or [0,255], adjust accordingly)
    return np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS).astype(np.float32)

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

model_path = "../results/vqvae_20250109_180050_4900.pth"

# --- 2. Fitness function ---
def evaluate(individual):
    # Convert individual to tensor
    image = torch.tensor(individual).permute(2, 0, 1).unsqueeze(0)
    # Calculate fitness
    metric = Metric(model_path)
    prob_carpet = metric.evo_metric(image)

    # prob_carpet = np.random.rand()  # placeholder
    
    return (prob_carpet,)  # DEAP expects a tuple


toolbox.register("evaluate", evaluate)

# --- 3. Genetic operators ---
# For crossover, example: 2D single-point
def cxTwoPoint2D(ind1, ind2):
    h, w, c = ind1.shape
    # Flatten to 1D
    arr1 = ind1.flatten()
    arr2 = ind2.flatten()
    cxpoint1 = random.randint(1, len(arr1) - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, len(arr1) - 1)
    arr1[cxpoint1:cxpoint2], arr2[cxpoint1:cxpoint2] = arr2[cxpoint1:cxpoint2], arr1[cxpoint1:cxpoint2]
    # reshape back
    ind1[:] = arr1.reshape(h, w, c)
    ind2[:] = arr2.reshape(h, w, c)
    return ind1, ind2

toolbox.register("mate", cxTwoPoint2D)

# For mutation, example: add noise
def mutGaussian(ind, mu=0.0, sigma=0.1, indpb=0.05):
    # Flatten
    arr = ind.flatten()
    for i in range(len(arr)):
        if random.random() < indpb:
            arr[i] += random.gauss(mu, sigma)
            arr[i] = np.clip(arr[i], 0, 1)  # keep in [0,1]
    ind[:] = arr.reshape(ind.shape)
    return (ind,)

toolbox.register("mutate", mutGaussian)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selRoulette)

# --- 4. Evolve ---
def main():
    pop = toolbox.population(n=500)
    NGEN = 100
    CXPB, MUTPB = 0.5, 0.2
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for gen in range(NGEN):
        print(f"--- Generation {gen} ---")
        # Select
        offspring = toolbox.select(pop, len(pop))
        # Clone
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # invalidate fitness
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Re-evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replacement
        pop[:] = offspring
    
    # Return best
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind

if __name__ == "__main__":
    best_ind = main()
    # best_image is the evolved array with highest "carpet-likeness"
    best_image = torch.tensor(best_ind).cpu().numpy()
    print("Best individual (image):", best_image)

    import matplotlib.pyplot as plt
    plt.imshow(best_image)
    plt.savefig("best_image.png")
    plt.show()
