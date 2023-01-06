from numpy import asarray, exp
from numpy.random import randn, rand, seed
from matplotlib import pyplot
from tqdm.notebook import tqdm, trange

seed(1)

"""
Simulated Annealing Class
"""
import random
import math

class SimulatedAnnealing:
    def __init__(self, kLow, kHigh, initialSolution, solutionEvaluator, initialTemp, finalTemp, tempReduction, neighborOperator, sizeFactor=16, alpha=0.95, beta=5):
        self.solution = initialSolution
        self.evaluate = solutionEvaluator
        self.initialTemp = initialTemp
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.sizeFactor = sizeFactor
        self.alpha = alpha
        self.beta = beta
        self.neighborOperator = neighborOperator
        self.counter = 0
        self.acceptedSols = 0
        self.kLower = kLow
        self.kUpper = kHigh
        self.globalBest = 10000000000000
        self.bestK = self.kLower

        self.n = self.solution.number_of_nodes()

        if tempReduction == "linear":
            self.decrementRule = self.linearTempReduction
        elif tempReduction == "geometric":
            self.decrementRule = self.geometricTempReduction
        elif tempReduction == "slowDecrease":
            self.decrementRule = self.slowDecreaseTempReduction
        else:
            self.decrementRule = tempReduction

    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

    def isTerminationCriteriaMet(self):
        # can add more termination criteria
        return self.currTemp <= self.finalTemp or self.counter >=5 #or self.neighborOperator(self.solution) == 0

    def mutateSolution(self, v, team, newB, newC_w):
        oldT = self.solution.nodes[v]['team']
        self.counts[oldT - 1] -= 1
        self.solution.nodes[v]['team'] = team
        self.counts[team - 1] += 1
        self.acceptedSols += 1
        self.output[v] = team
        self.b = newB
        self.Cw = newC_w
        self.bPen = math.exp(B_EXP * self.b)
        self.currCost = self.Cw + self.kPen + self.bPen
        
    def run(self):
      #outer loop of trying different k values
        for currK in range(self.kLower, self.kUpper + 1):
            generateInitialSequential(self.solution, currK)
            seq = self.evaluate(self.solution)
            generateInitialChunk(self.solution, currK)
            chunk = self.evaluate(self.solution)
            generateInitialRandom(self.solution, currK)
            ran = self.evaluate(self.solution)
            if seq == min([seq, chunk, ran]):
                generateInitialSequential(self.solution, currK)
            elif chunk == min([seq, chunk, ran]):
                generateInitialChunk(self.solution, currK)
                
            
            self.output = [self.solution.nodes[v]['team'] for v in range(self.solution.number_of_nodes())]
            self.teams, self.counts = np.unique(self.output, return_counts=True)
            self.iterationPerTemp = self.sizeFactor * self.solution.number_of_nodes()
            self.currTemp = self.initialTemp
            self.counter = 0
            
            self.Cw, self.kPen, self.bPen = self.evaluate(self.solution, True)
            self.b = math.log(self.bPen) / B_EXP
            self.currCost = self.Cw + self.kPen + self.bPen
            
            
            while not self.isTerminationCriteriaMet():
                # iterate that number of times
                for i in range(self.iterationPerTemp):
                    self.acceptedSols = 0
                    # get the cost between the two solutions
                    # neighbouroperator should select a random neighbour and return new cost
                    #pick a random vertex to switch partition
                    randVertex1 = random.randint(0, self.n - 1)
                    randV1Team = self.solution.nodes[randVertex1]['team']
                    randVertex2 = random.randint(0, self.n - 1)
                    while randV1Team == self.solution.nodes[randVertex2]['team']:
                        randVertex2 = random.randint(0, self.n - 1)
                    randV2Team = self.solution.nodes[randVertex2]['team']
                    neighbors, v, team, newB, newC_w = self.neighborOperator(self.solution, randV2Team, randVertex1, self.b, self.Cw, currK, self.counts)
                    cost = self.currCost - neighbors
                    # if the new solution is better, accept it
                    if cost >= 0:
                        self.mutateSolution(v, team, newB, newC_w)
                        # check if swap is even better
                        self.currCost = self.Cw + self.kPen + self.bPen
                        # find vertex in randTeam
                        neighbors, v, team, newB, newC_w = self.neighborOperator(self.solution, randV1Team, randVertex2, self.b, self.Cw, currK, self.counts)
                        # self.swap(v)
                        cost = self.currCost - neighbors
                        if cost >= 0:
                            self.mutateSolution(v, team, newB, newC_w)
                                         
                    # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                    else:
                        if random.uniform(0, 1) < math.exp((cost / self.currTemp)):
#                             print("Current temp: ", self.currTemp)
#                             print("Probability of escape:", math.exp( (cost / self.currTemp)))
                            self.mutateSolution(v, team, newB, newC_w)

                #calculate accepted Solutions percentage
                if self.acceptedSols/self.iterationPerTemp <= 0.02:
                    self.counter += 1
                # decrement the temperature
                self.decrementRule()
            
            c = self.evaluate(self.solution)
#             print("candidate score for " + str(currK) + " :", c)
            if c < self.globalBest:
                self.globalBest = c
                self.bestOutput = [self.solution.nodes[v]['team'] for v in range(self.solution.number_of_nodes())]
                self.bestK = currK
            # TERMINATION CONDITION IF SCORE IS TOO BAD
            elif c > 1.5*self.globalBest:
                break
    
        #at the end of the day, set output
        for i in range(self.solution.number_of_nodes()):
            self.solution.nodes[i]['team'] = self.bestOutput[i]
        print("best k", self.bestK)
        return self.bestK, self.bestOutput[:], self.globalBest
            
def generateInitialRandom(G: nx.Graph, newK):
    k = newK
    n = G.number_of_nodes()
    for i in range(n):
        G.nodes[i]['team'] = random.randint(1, k)

def generateInitialSequential(G: nx.Graph, newK):
    k = newK
    n = G.number_of_nodes()
    i = 0
    while i < n:
        G.nodes[i]['team'] = (i % k)+1
        i += 1

def generateInitialChunk(G: nx.Graph, newK):
    k = newK
    n = G.number_of_nodes()
    start = 0
    end = n // k
    incr = end
    team = 1
    while start < n:
        for i in range(start, min(n, end)):
            G.nodes[i]['team'] = team
        team += 1
        start, end = start + incr, end + incr


def findRandomNeighbour(G: nx.Graph, newTeam, v, b, Cw, k, counts):
  # changing team of v to newTeam 
  # partition i is old, j is new
    n = G.number_of_nodes()
    
    oldTeam = G.nodes[v]['team']
    #find new Cp
    bSquared = b**2
    # number of vertices in part i
    pi = counts[oldTeam - 1]
#     print('pi:', pi)
    # deviation from mean
    oldbi = pi/n - 1/k
    # number of vertices in part j
    pj = counts[newTeam - 1]
    # deviation from mean
    oldbj = pj/n - 1/k
    newB = math.sqrt(max(0, bSquared - oldbi**2 - oldbj**2 + (oldbi - 1/n)**2 + (oldbj + 1/n)**2))
    newCp = math.exp(B_EXP * newB)
#     print('newCp:', newCp)
    #find new Cw
    # C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v])
#     print('C_w:', C_w)
    for u in G.neighbors(v):
        if G.nodes[u]['team'] == oldTeam:
            Cw = Cw - G[u][v]['weight']
        elif G.nodes[u]['team'] == newTeam:
            Cw = Cw + G[u][v]["weight"]
#     print('newC_w:', C_w)
#     print('midTerm:', K_COEFFICIENT * math.exp(K_EXP * k))
    #return updated cost
    newCost = Cw + K_COEFFICIENT * math.exp(K_EXP * k) + newCp
    newC_w = Cw
    
    return newCost, v, newTeam, newB, newC_w



def solve(G: nx.Graph):
    # TODO implement this function with your solver
    # Assign a team to v with G.nodes[v]['team'] = team_id
    # Access the team of v with team_id = G.nodes[v]['team']

    scoresArr = []
    outputsArr = []
    for _ in range(40):
        trial = SimulatedAnnealing(2, 15, G, score, 5, 0.01, "geometric", findRandomNeighbour, 2)
        maybeK, y1, x1 = trial.run()
        trial2 = SimulatedAnnealing(max(2, maybeK - 1), maybeK + 1, G, score, 5, 0.01, "geometric", findRandomNeighbour, 4)
        bestK, y2, x2 = trial2.run()
        model = SimulatedAnnealing(bestK, bestK, G, score, 52, 0.01, "geometric", findRandomNeighbour, 90)
        _, bestOutput, bestScore = model.run()
        x = min(bestScore, x1, x2)
        if bestScore == x:
            scoresArr.append(bestScore)
            outputsArr.append(bestOutput)
        elif x2 == x:
            scoresArr.append(x2)
            outputsArr.append(y2)
        else:
            scoresArr.append(x1)
            outputsArr.append(y1)
    
#     print(scoresArr)
    ultimateScore = min(scoresArr)
    ultimateOutput = outputsArr[scoresArr.index(ultimateScore)]
    for i in range(model.solution.number_of_nodes()):
            model.solution.nodes[i]['team'] = ultimateOutput[i]
                        
    print("solution:")
    print(ultimateOutput)