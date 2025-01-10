import random
import multiprocessing
import numpy as np
import torch
import torch.nn.functional as F
from deap import base, creator, tools
import os, sys
import matplotlib.pyplot as plt
from datetime import datetime

from fontTools.colorLib.builder import populateCOLRv0
from torch.distributed.tensor.parallel import loss_parallel

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)

from vae_metric.metric import Metric


def build_symmetric_image(img_64):
    """
    Given a 64x64x3 image in [H,W,C] format,
    build a 128x128 symmetrical image:
      - top-left: original img_64
      - top-right: horizontally flipped
      - bottom-left: vertically flipped
      - bottom-right: horizontally + vertically flipped
    Returns: a 128x128x3 NumPy array.
    """
    H, W, C = img_64.shape  # should be (64, 64, 3)
    bigger_size = 2 * H     # 128
    big_img = np.zeros((bigger_size, bigger_size, C), dtype=img_64.dtype)

    # 1) Top-left: original
    big_img[0:H, 0:W, :] = img_64

    # 2) Top-right: horizontal flip of original
    #    np.flip(..., axis=1) flips width dimension
    big_img[0:H, W:2*W, :] = np.flip(img_64, axis=1)

    # 3) Bottom-left: vertical flip
    #    np.flip(..., axis=0) flips height dimension
    big_img[H:2*H, 0:W, :] = np.flip(img_64, axis=0)

    # 4) Bottom-right: both horizontal + vertical flip
    big_img[H:2*H, W:2*W, :] = np.flip(img_64, axis=(0, 1))

    return big_img

toolbox = base.Toolbox()

# individuals are white canvases,
def create_individual():
    return np.ones((SIZE, SIZE, CHANNELS), np.float32)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize probability
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


SIZE = 64
CHANNELS = 3


model_path = os.path.join("..", "results", "vqvae_small_vae_48000.pth")
metric = Metric(model_path)

def evaluate(individual):
    img = build_symmetric_image(individual)

    # basically, reorder the array so that RGB column is first then coordinates, then add a batch dimension
    image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return metric.evo_metric(image)

def blend_crossover(m, d, alpha=0.5):
    if m.shape != d.shape:
        raise ValueError("mom and dad must have the same shape")

    child1 = alpha * m + (1 - alpha) * d
    child2 = alpha * d + (1 - alpha) * m

    return child1, child2

def uniform_rows_crossover(m, d):
    if m.shape != d.shape:
        raise ValueError("mom and dad must have the same shape")

    mask = np.random.randint(0, 2, size=m.shape, dtype=bool)
    child1 = np.where(mask, m, d)
    child2 = np.where(mask, d, m)
    return child1, child2

# TODO: maybe also some columns crossover

def subsquare_crossover(parent1, parent2, max_sub_size=16):
    h, w, c = parent1.shape
    # Random sub-square size
    sub_h = random.randint(1, max_sub_size)
    sub_w = random.randint(1, max_sub_size)

    # Random top-left corner
    start_x = random.randint(0, h - sub_h)
    start_y = random.randint(0, w - sub_w)

    # Create copy to avoid overwriting parent data
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)

    # Perform the swap
    child1[start_x:start_x+sub_h, start_y:start_y+sub_w] = parent2[start_x:start_x+sub_h, start_y:start_y+sub_w]
    child2[start_x:start_x+sub_h, start_y:start_y+sub_w] = parent1[start_x:start_x+sub_h, start_y:start_y+sub_w]

    return child1, child2

def pixel_mutation(ind, prob=0.1):
    num_mutations = int(prob * SIZE * SIZE)
    x_coords = np.random.randint(0, SIZE, size=num_mutations)
    y_coords = np.random.randint(0, SIZE, size=num_mutations)
    for x, y in zip(x_coords, y_coords):
        ind[x, y] = np.random.rand(CHANNELS) # .rand creates an array of CHANNELS elems, each between 0..1
                                            
    return ind

def pattern_mutation(ind, pattern_size=3):
    x, y = random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)
    color = ind[x, y]
    shape = random.randint(0, 2)   # choose between circle, square, or triangle

    # FIXME: GPT generated, i was too lazy to code the shapes in, so beware
    if shape == 0:   # circle
        xx, yy = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= pattern_size ** 2
        ind[mask] = color

    elif shape == 1:  # square
        x_min, x_max = max(0, x - pattern_size), min(SIZE, x + pattern_size)
        y_min, y_max = max(0, y - pattern_size), min(SIZE, y + pattern_size)
        ind[x_min:x_max, y_min:y_max] = color

    elif shape == 2:  # triangle
        for i in range(max(0, x - pattern_size), min(SIZE, x + pattern_size)):
            for j in range(max(0, y - pattern_size), min(SIZE, y + pattern_size)):
                if abs(i - x) + abs(j - y) <= pattern_size:
                    ind[i, j] = color

    return ind

toolbox.register("blend_XO", blend_crossover)
toolbox.register("uniform_row_XO", uniform_rows_crossover)
toolbox.register("subsquare_XO", subsquare_crossover)
toolbox.register("pattern_mut", pattern_mutation)
toolbox.register("pixel_mut", pixel_mutation)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selRoulette)


def evolution():
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    random.seed(42)

    POP_SIZE = 500
    NGEN = 300
    CXPB = 0.7
    MUTPB = 0.3

    STAGNATION_LIMIT = 10
    EPSILON = 1e-6
    stagnation_count = 0
    prev_mean = None

    population = toolbox.population(n=POP_SIZE)

    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    for gen in range(NGEN):
        print(f"=== Generation {gen} ===")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                if random.random() < 0.5:
                    child1[:], child2[:] = toolbox.blend_XO(child1, child2)
                else:
                    child1[:], child2[:] = toolbox.uniform_row_XO(child1, child2)

        for mutant in offspring:
            if random.random() < MUTPB:
                if random.random() < 0.8:
                    toolbox.pixel_mut(mutant)
                else:
                    toolbox.pattern_mut(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum_sq = sum(x*x for x in fits)
        std = abs(sum_sq/length - mean**2)**0.5
        print(f"  Min {min(fits):.3f}")
        print(f"  Max {max(fits):.3f}")
        print(f"  Avg {mean:.3f}")
        print(f"  Std {std:.3f}")

        if prev_mean is not None:
            if abs(mean - prev_mean) < EPSILON:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= STAGNATION_LIMIT:
                print("Aladin found its carpet")
                break
        prev_mean = mean

        if gen % 5 == 0:
            curr_best_ind = tools.selBest(population, 1)[0]
            img = build_symmetric_image(curr_best_ind)
            os.makedirs(f"run{timestamp}", exist_ok=True)
            plt.imshow(torch.tensor(img).cpu().numpy())
            plt.savefig(f"run{timestamp}/gen_{gen}.png")

    best_ind = tools.selBest(population, 1)[0]
    print("\nBest individual is:", best_ind)
    print("Best fitness is:", best_ind.fitness.values[0])

    return best_ind

if __name__ == "__main__":
    best_ind = evolution()
    best_ind = build_symmetric_image(best_ind)
    best_image = torch.tensor(best_ind).cpu().numpy()

    plt.imshow(best_image)
    plt.savefig("best_image.png")
    plt.show()
