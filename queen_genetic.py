import numpy as np
import random
import matplotlib.pyplot as plt


def distance(city1, city2):
    xDis = abs(city1[0] - city2[0])
    yDis = abs(city1[1] - city2[1])
    distance = np.sqrt((xDis ** 2) + (yDis ** 2))
    return distance


def fitness(Solution):
    score  = len(Solution)**2 + len(Solution)
    for i in range(len(Solution)):
        for j in range(len(Solution)):
            if Solution[j]== abs(j-i)+Solution[i]  or Solution[j]== Solution[i] - abs(j-i) : score-=1
    return score


def createSolution(genes):
    Solution = random.sample(genes, len(genes))
    return Solution


def initialPopulation(selection, genes):
    population = []
    for i in range(0, selection):
        population.append(createSolution(genes))
    return population


def rankSolutions(population):
    fitnessList = []
    for Solution in population:
        fitnessList.append([Solution, fitness(Solution)])
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
    popRank = rankSolutions(currentGen)
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


def plotResult(progress, bestSolution):
    plt.figure(1)
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.figure(2)  # row 1, col 2 index 1
    size = len(bestSolution)
    lines  = []
    for i in range(size):
        temp = []
        temp.append([i*10 , 0])
        temp.append([i * 10, size*10])
        lines.append(temp)
        temp =[]
        temp.append([0,i * 10])
        temp.append([size * 10,i * 10 ])
        lines.append(temp)
    lines.append([[0,size*10],[size*10,size*10]])
    lines.append([[ size * 10,0], [size * 10, size * 10]])
    solution = []
    for i in range(len(bestSolution)):
        solution.append([(bestSolution[i]*10)+5,i*10 +5])
    solution = np.array(solution)
    plt.title("best Solution found")
    plt.scatter(solution[:, 0], solution[:, 1], s=100,c='red')
    lines = np.array(lines)
    for line in lines :
        plt.plot(line[:, 0], line[:, 1], 'b--')
    plt.show()


def geneticAlgorithm(population, selection, mutationRate, generations):
    print("Initial Fitness: " + str(rankSolutions(population)[0][1]))
    progress = [rankSolutions(population)[0][1]]
    for i in range(0, generations):
        population = nextGeneration(population, selection, mutationRate)
        r = rankSolutions(population)
        progress.append(r[0][1])
        if i % 10 == 0 and i > 0:
            print("generation " + str(i) + " , fitness = " + str(progress[-1]))
    fit = rankSolutions(population)[0][1]
    bestSolution = rankSolutions(population)[0][0]
    print("Final Fitness: " + str(fit))
    print("Number of thread: "+str(int((len(bestSolution)**2 - fit)/2)))
    print("best Solution found : " + str(bestSolution))
    plotResult(progress, bestSolution)

number_of_queen = 8
genes = []
for i in range(number_of_queen): genes.append(i)
population = initialPopulation(selection=5, genes=genes)
geneticAlgorithm(population=population, selection=5, mutationRate=0.2, generations=40)
