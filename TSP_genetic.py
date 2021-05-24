import numpy as np
import random
import matplotlib.pyplot as plt


def distance(city1, city2):
    xDis = abs(city1[0] - city2[0])
    yDis = abs(city1[1] - city2[1])
    distance = np.sqrt((xDis ** 2) + (yDis ** 2))
    return distance


def fitness(route):
    pathDistance = 0
    for i in range(0, len(route)):
        fromCity = route[i]
        if i + 1 < len(route):
            toCity = route[i + 1]
        else:
            toCity = route[0]
        pathDistance = pathDistance + distance(toCity, fromCity)
    return 1 / float(pathDistance)


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(selection, genes):
    population = []
    for i in range(0, selection):
        population.append(createRoute(genes))
    return population


def rankRoutes(population):
    fitnessList = []
    for route in population:
        fitnessList.append([route, fitness(route)])
    return sorted(fitnessList, key=lambda t: t[1], reverse=True)


def crossover(parent1, parent2):
    crossoverPoint = int(len(parent1) / 2)
    part1parent1 = parent1[:crossoverPoint]
    part2parent1 = parent1[crossoverPoint:]
    part1parent2 = parent2[:crossoverPoint]
    part2parent2 = parent2[crossoverPoint:]
    child1 = part1parent1
    child2 = part1parent2
    child1.extend([item for item in part2parent2 if item not in part1parent1])
    child1.extend([item for item in part1parent2 if item not in part1parent1])
    child2.extend([item for item in part2parent1 if item not in part1parent2])
    child2.extend([item for item in part1parent1 if item not in part1parent2])
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
    for i in range(len(child)):
        if random.random() < mutationRate:
            j = int(random.random() * len(child))
            city1 = child[i]
            city2 = child[j]
            child[i] = city2
            child[j] = city1
    return child


def mutateChildren(children, mutationRate):
    mutatedChildren = []
    for child in children:
        mutatedChildren.append(mutate(child, mutationRate))
    return mutatedChildren


def selectParent(currentGen, selection):
    popRank = rankRoutes(currentGen)
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


def plotResult(progress, bestRoute):
    plt.figure(1)
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.figure(2)  # row 1, col 2 index 1
    bestRoute.append(bestRoute[0])
    bestRoute = np.array(bestRoute)
    plt.title("best route found")
    plt.scatter(bestRoute[:, 0], bestRoute[:, 1])
    plt.plot(bestRoute[:, 0], bestRoute[:, 1], 'r--')
    plt.show()


def geneticAlgorithm(population, selection, mutationRate, generations):
    print("Initial Fitness: " + str(rankRoutes(population)[0][1]))
    progress = [rankRoutes(population)[0][1]]
    for i in range(0, generations):
        population = nextGeneration(population, selection, mutationRate)
        r = rankRoutes(population)
        progress.append(r[0][1])
        if i % 10 == 0 and i > 0:
            print("generation " + str(i) + " , fitness = " + str(progress[-1]))
    print("Final Fitness: " + str(rankRoutes(population)[0][1]))
    bestRoute = rankRoutes(population)[0][0]
    print("best route found : " + str(bestRoute))
    plotResult(progress, bestRoute)


cityList = [[175, 43], [130, 36], [147, 116], [101, 71], [184, 93], [14, 155], [169, 116], [89, 184], [48, 196],
            [107, 18], [78, 13], [59, 34], [95, 163], [124, 97], [100, 24], [99, 88], [131, 24], [76, 114], [69, 56],
            [124, 92], [137, 78], [94, 54], [174, 156], [136, 53], [139, 148], [137, 114], [155, 166], [127, 6],
            [166, 145], [34, 138]]
# for i in range(0, 30):
#     cityList.append([int(random.random() * 200), int(random.random() * 200)])
population = initialPopulation(selection=100, genes=cityList)
geneticAlgorithm(population=population, selection=50, mutationRate=0.01, generations=1000)
