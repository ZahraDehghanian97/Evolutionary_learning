import numpy as np
import random
import matplotlib.pyplot as plt


def fitness(individual):
    sum_w = 0
    sum_p = 0
    # get weights and profits
    for index, i in enumerate(individual):
        if i == 0:
            continue
        else:
            sum_w += weights[index]
            sum_p += profits[index]

    # if greater than the optimal return -1 or the number otherwise
    if sum_w > C:
        return -1
    else:
        return sum_p


def createSort(genes):
    parent = []
    for k in range(genes):
        k = random.randint(0, 1)
        parent.append(k)
    return parent


def initialPopulation(selection, genes):
    population = []
    for i in range(0, selection):
        population.append(createSort(genes))
    return population


def rankPopulation(population):
    fitnessList = []
    for individual in population:
        fitnessList.append([individual, fitness(individual)])
    return sorted(fitnessList, key=lambda t: t[1], reverse=True)


def crossover(parent1, parent2):

    threshold = random.randint(1, len(parent1) - 1)
    child1 = parent1[:threshold]
    child2 = parent2[:threshold]
    child1.extend(parent2[threshold:])
    child2.extend(parent1[threshold:])
    return child1, child2


def crossoverParent(parents):
    children = []
    parents = random.sample(parents, len(parents))
    for i in range(0, int(len(parents) / 2)):
        child1, child2 = crossover(parents[i], parents[len(parents) - i - 1])
        children.append(child1)
        children.append(child2)

    return children


def mutate(child, mutationRate):
    mutated_child = child
    for i in range(len(child)):
        if random.random() < mutationRate:
            mutated_child[i] = (mutated_child[i]+1)%2
    return mutated_child


def mutateChildren(children, mutationRate):
    mutatedChildren = []
    for child in children:
        mutatedChildren.append(mutate(child, mutationRate))
    return mutatedChildren


def selectParent(currentGen, selection):
    popRank = rankPopulation(currentGen)
    popRank = np.array(popRank)
    popRank = popRank[:, 0]
    popRank = popRank[0:selection]
    return list(popRank)


def nextGeneration(currentGen, selection, mutationRate):
    parents = selectParent(currentGen, selection)
    children = crossoverParent(parents)
    children = mutateChildren(children, mutationRate)
    nextGeneration = parents
    nextGeneration.extend(children)
    return nextGeneration


def plotResult(progress, bestAnswer):
    plt.figure(1)
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()


def computeWeight(bestAnswer):
    c = 0
    for i in range(len(bestAnswer)):
        c += bestAnswer[i] * weights[i]
    return c


def geneticAlgorithm(population, selection, mutationRate, generations):
    print("Initial Fitness: " + str(rankPopulation(population)[0][1]))
    progress = [rankPopulation(population)[0][1]]
    for i in range(0, generations):
        population = nextGeneration(population, selection, mutationRate)
        r = rankPopulation(population)
        progress.append(r[0][1])
        if i % 50 == 0 and i > 0:
            print("generation " + str(i) + " , fitness = " + str(progress[-1]))
    bestAnswer = rankPopulation(population)[0][0]
    print("Final Fitness: " + str(rankPopulation(population)[0][1]))
    print("Final weights backpack: " + str(computeWeight(bestAnswer)))
    print("Maximum Capacity: "+str(C))
    print("best Answer found : " + str(bestAnswer))
    plotResult(progress, bestAnswer)

def generate_items(number_of_item):
    weights=[]
    profits = []
    for i in range(number_of_item):
        weights.append(random.randint(0,10)*100)
        profits.append(random.randint(0, 10) )
    return weights , profits


# properties for this particular problem

all_weights = [1000, 900, 700, 1000, 900, 100, 900, 30, 100, 80, 100, 600, 500, 1000, 300, 800, 40, 90, 900, 800, 200, 300, 900, 500, 100, 400, 500, 400, 400, 600]
all_profits =[1, 1, 9, 3, 3, 2, 10, 1, 1, 8, 2, 4, 6, 9, 4, 9, 10, 9, 0, 5, 8, 3, 5, 2, 5, 7, 9, 0, 4, 3]
number_of_item = 15
# weights , profits = generate_items(number_of_item)
weights = all_weights[:number_of_item]
profits = all_profits[:number_of_item]
C = np.sum(weights)/2
print("List of Objects :")
print("-------------------------")
print("weights : "+str(weights))
print("profits"+str(profits))
print("-------------------------")

# for i in range(0, 30):
#     cityList.append([int(random.random() * 200), int(random.random() * 200)])
population = initialPopulation(selection=10, genes=len(weights))
geneticAlgorithm(population=population, selection=5, mutationRate=0.01, generations=200)
