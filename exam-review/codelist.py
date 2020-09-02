'''
Implementation of some algorithms in GA,
a suppliment of KCL BIM lecture notes

@ Author: Ricky Zhu
@ email:  rickyzhu@foxmail.com
'''
import numpy as np

def bin_int(integer):
    '''convert integer to binary string'''
    if integer == 0:
        return '0'
    result = []
    while integer != 0:
        quotient, remainer = divmod(integer, 2)
        result.append(str(remainer))
        integer = quotient
    result.reverse()
    return ''.join(result)

def bin_frac(fractional, precision=4):
    '''
    convert a fractional number to binary string
    Example: bin_frac(0.3215, 4)
    '''
    result = ''
    i = 0
    while fractional != 0 and i < precision:
        result += str(int(fractional * 2))
        fractional = fractional * 2 - int(fractional * 2)
        i += 1
    return result

# print(bin_int(25)+'.'+bin_frac(0.3125))

def encode(lo, hi, num, N_bits):
    '''encode an number to a binary string within a given range'''
    assert num>=lo and num<=hi, "num {0} out of domain [{1}, {2}]".format(num, lo, hi)
    mapping = round(((num-lo) / (hi-lo))* (2**N_bits - 1))
    return bin_int(mapping).zfill(N_bits)

def decode(lo, hi, gene, precision=2):
    '''decode a binary string to a number within a given range'''
    return round(lo + int(gene, 2)*((hi-lo)/(2**len(gene)-1)), precision)

def binary2grey(bstr):
    '''convert an odinary binary string into a gray code form'''
    gstr = bstr[0]
    for i in range(1, len(bstr)):
        gstr += str(int(bstr[i]) ^ int(bstr[i-1]))
    return gstr

def grey2binary(gstr):
    '''convert an gray code string into an odinary binary form'''
    bstr = gstr[0]
    for i in range(1, len(gstr), 1):
        bstr += str(int(bstr[-1]) ^ int(gstr[i]))
    return bstr

# print(binary2grey('1011'))
# print(grey2binary('1110'))

def pmx(par1, par2, cxp1, cxp2):
    '''partially matched crossover'''
    def _pmx(par1, par2, cxp1, cxp2):
        offsp = par1[:cxp1] + par2[cxp1:cxp2] + par1[cxp2:]
        for i in range(len(offsp)):
            if i in range(cxp1, cxp2):
                continue
            if offsp[i] in offsp[cxp1:cxp2]:
                for x2 in par2:    
                    if x2 not in offsp:
                        offsp[i] = x2
                        break
        return offsp
    return _pmx(par1, par2, cxp1, cxp2), _pmx(par2, par1, cxp1, cxp2)

# print(pmx([3,4,6,2,1,5], [4,1,5,3,2,6], 1, 3))

def ox(par1, par2, cxp1, cxp2):
    '''ordered crossover'''
    def _ox(par1, par2, cxp1, cxp2):
        offsp = par1[:cxp1] + par2[cxp1:cxp2] + par1[cxp2:]
        # copy from the second crossover point
        modified_par1 = par1[cxp2:] + par1[:cxp2]   
        for x in par2[cxp1:cxp2]:  
            # Cross out duplicates
            modified_par1.remove(x)
        # copy from the second crossover point
        positions =  list(range(cxp2, len(par1))) + list(range(0, cxp1))  
        for i in range(len(positions)):
            offsp[positions[i]] = modified_par1[i]
        return offsp
    return _ox(par1, par2, cxp1, cxp2), _ox(par2, par1, cxp1, cxp2)

# print(ox([3,4,6,2,1,5], [4,1,5,3,2,6], 2, 4))

def cx(par1, par2):
    '''cycled crossover'''
    def swap(chrom1, chrom2, pos):
        chrom1[pos], chrom2[pos] = chrom2[pos], chrom1[pos]
    def is_duplicated(chrom, pos):
        if chrom.count(chrom[pos]) > 1:
            return True
        return False
    def is_terminate(chrom):
        for i in range(len(chrom)):
            if is_duplicated(chrom, i):
                return False
        return True
    offsp1 = par1.copy()
    offsp2 = par2.copy()
    swap(offsp1, offsp2, 0)
    while not is_terminate(offsp1):
        for i in range(1, len(offsp1)):
            if is_duplicated(offsp1, i):
                swap(offsp1, offsp2, i)
    return offsp1, offsp2

# print(cx([3,4,6,2,1,5], [4,1,5,3,2,6]))

def ref_encode(par):
    '''coding with reference list'''
    coded_par = []
    reflist = list(range(1, len(par)+1))
    for x in par:
        coded_par.append(reflist.index(x)) # code start from 0
        # coded_par.append(reflist.index(x)+1) # code start from 1
        reflist.remove(x)
    return coded_par 

# print(ref_encode([3,4,6,2,1,5]))

def ref_decode(coded_par):
    '''decode a coded parent'''
    par = []
    reflist = list(range(1, len(coded_par)+1))
    for i in coded_par:
        par.append(reflist[i])
        reflist.remove(reflist[i])
    return par

# print(ref_decode(ref_encode([3,4,6,2,1,5])))

def inversion(chrom, start, end):
    substr = chrom[start:end]
    substr.reverse()
    return chrom[:start] + substr + chrom[end:]

# print(inversion([6,1,5,3,2,4], 1, 5))

def insertion(chrom, start, end, pos):
    substr = chrom[start:end]
    displacement = chrom[:start] + chrom[end:]
    return displacement[:pos] + substr + displacement[pos:]

# print(insertion([6,1,5,3,2,4], 2, 5, 1))

def reciprocal(chrom, pos1, pos2):
    replicate = chrom.copy()
    replicate[pos1], replicate[pos2] = replicate[pos2], replicate[pos1]
    return replicate

# print(reciprocal([6,1,5,3,2,4], 1, 4))



# pop = [
#     {'chr': '100011', 'val': [2, 1], 'cost': 5, 'Ncost': None, 'p': None, 'sum_p':None},
#     {'chr': '001110', 'val': [-1, 4], 'cost': 17, 'Ncost': None, 'p': None, 'sum_p':None},
#     {'chr': '100101', 'val': [2, 3], 'cost': 13, 'Ncost': None, 'p': None, 'sum_p':None},
#     {'chr': '011001', 'val': [1,-1], 'cost': 2, 'Ncost': None, 'p': None, 'sum_p':None}
# ]

def mating_pool(population:list, N_keep:int)->list:
    '''pick N_keep individuals into selection'''
    pool = []
    ranked_pop = population.copy()
    ranked_pop.sort(key=lambda x:x['cost'])
    total_Ncost = 0
    for i in range(N_keep):
        ranked_pop[i]['Ncost'] = ranked_pop[i]['cost'] - ranked_pop[N_keep]['cost']
        total_Ncost += ranked_pop[i]['Ncost']
    total_p = 0
    for i in range(N_keep):
        ranked_pop[i]['p'] = ranked_pop[i]['Ncost'] / total_Ncost
        total_p += ranked_pop[i]['p']
        ranked_pop[i]['sum_p'] = total_p
        pool.append(ranked_pop[i])
    return pool

# print(mating_pool(pop, 2))

def bga_weighted_rank_selection(mating_pool:list, r:float)->tuple:
    assert r>=0 or r <= 1, 'r should be within [0, 1]'
    for i in mating_pool:
        if i['sum_p'] >= r:
            return i
    return None

# pool = mating_pool(pop, 2)
# print(bga_weighted_rank_selection(pool, 0))
# print(bga_weighted_rank_selection(pool, 0.3))
# print(bga_weighted_rank_selection(pool, 0.8))
# print(bga_weighted_rank_selection(pool, 1))

def bga_crossover(chr1:str, chr2:str, cxp1:int, cxp2=None)->tuple:
    '''
    single or double point crossover
    cxp means the gaps between numbers, starting from 1
    '''
    if cxp2 == None:
        cxp2 = len(chr1)
    res1 = chr1[:cxp1] + chr2[cxp1:cxp2] + chr1[cxp2:]
    res2 = chr2[:cxp1] + chr1[cxp1:cxp2] + chr2[cxp2:]
    return res1, res2

# print(bga_crossover('11000', '00111', 2))
# print(bga_crossover('11000', '00111', 2, 4))
# print(bga_crossover('11000', '00111', 1, 4))


# ===============================================
#             Evolution Strategy
# ===============================================

def plus_strategy(parents, offspring, func):
    population = np.concatenate([parents, offspring])
    miu = len(parents)
    fitness = [func(x) for x in population]
    idx = np.argsort(fitness)
    res = []
    for i in idx[:miu]:
        res.append(population[i])
    return np.array(res)

def comma_strategy(parents, offspring, func):
    population = offspring
    miu = len(parents)
    fitness = [func(x) for x in population]
    idx = np.argsort(fitness)
    res = []
    for i in idx[:miu]:
        res.append(population[i])
    return np.array(res)

def local_discrete_cx(x1, x2, s1, s2, r):
    '''
    Reconbination with local, discrete crossover

    Parameters
    ----------
    x1, x2 - 1D np array, selected parent individuals
    s1, s2 - 1D np array, respective strategy parameter of parent individuals
    r      - 1D np array, a sequence of random numbers
    '''
    new_x = np.zeros(len(x1))
    new_s = np.zeros(len(s1))
    for i in len(len(r)):
        if r[i] <= 0.5:
            new_x[i] = x1[i]
            new_s[i] = s1[i]
        else:
            new_x[i] = x2[i]
            new_s[i] = s2[i]
    return new_x, new_s

def local_intermediate_cx(x1, x2, s1, s2, r):
    new_x = x1 * r + x2 * (1-r)
    new_s = s1 * r + s2 * (1-r)
    return new_x, new_s
    
def global_discrete_cx(x1, X, s1, S, r, j):
    new_x = np.zeros(len(x1))
    new_s = np.zeros(len(s1))
    for i in len(len(r)):
        if r[i] <= 0.5:
            new_x[i] = x1[i]
            new_s[i] = s1[i]
        else:
            new_x[i] = X[j[i]][i]
            new_s[i] = S[j[i]][i]

    return new_x, new_s

def global_intermediate_cx(X, S):
    new_x = X.mean(axis=0)
    new_s = S.mean(axis=0)
    return new_x, new_s

def offspring_mutation(x, s, noises):
    off_x = x + s * noises
    return off_x

# Examples: Tutorial 6, Q 6
# --------------------------
def f(x):
    return x[0]**2 * np.sin(x[1]) + 2*(x[0]-x[1]) - x[1]**2 * np.cos(x[0])
X = np.array([[3, 8],[10,-10],[5, -5]])
S = np.array([[1, 2],[3, 4],[5, 6],])
r = np.array([0.5, 0.5])
noises = np.array([1,4])
new_x1, new_s1 = local_intermediate_cx(X[2], X[0], S[2], S[0], r)
offsp_x1 = offspring_mutation(new_x1, new_s1, noises)
offsp_s1 = new_s1 * 0.9

new_x2, new_s2 = local_intermediate_cx(X[1], X[2], S[1], S[2], r)
offsp_x2 = offspring_mutation(new_x2, new_s2, noises)
offsp_s2 = new_s2 * 0.9

print(plus_strategy(X, [offsp_x1, offsp_x2], f))



# ===============================================
#          Differential Evolution
# ===============================================

def trial_vector(target_vec, diff_vec1, diff_vec2, beta):
    return target_vec + beta * (diff_vec1 - diff_vec2)

def offspring_vector(x, u, j):
    new_x = np.zeros(len(x))
    for i in range(len(x)):
        if i in j:
            new_x[i] = u[i]
        else:
            new_x[i] = x[i]
    return new_x

# ===============================================
#                 Ant Colony
# ===============================================

class SACOGraph:
    def __init__(self, adjmat, pheroomes, evaperation_rate):
        self.adjmat = adjmat
        self.TAO = pheroomes
        self.p = evaperation_rate

    def transition_probability(self):
        probs = np.zeros(self.adjmat.shape)
        for i in range(self.adjmat.shape[0]):
            for j in range(self.adjmat.shape[1]):
                probs[i, j] = self.TAO[i,j] / self.TAO[i].sum()
        return probs

    def evaperate_pheromone(self):
        self.TAO = self.TAO * (1 - self.p)

    def update_pheromone(self, Q, func, routes):
        '''

        '''
        for i in range(self.adjmat.shape[0]):
            for j in range(self.adjmat.shape[1]):
                delta_tao_ij = []
                for k in range(len(routes)):
                    if (i,j) in routes[k]:
                        delta_tao_ij.append(Q/func(routes))
                self.TAO[i.j] += sum(delta_tao_ij)


# ===============================================
#          Parcle Swarm Optimization
# ===============================================

def update_velocity(V, X, Y, Y_hat, c1,c2, r1,r2):
    '''

    '''
    new_V = V + c1 * r1 * (Y-X) + c2 * r2 * (Y_hat - X)
    return new_V

def update_particlew(X, new_V):
    '''

    '''
    new_X = X + new_V
    return new_X

def updatew_personal_best(Y, new_X, func):
    '''

    '''
    res = []
    for i in range(len(Y)):
        if func(Y[i]) < func(new_X[i]):
            res.append(Y[i])
        else:
            res.append(new_X[i])
        return np.array(res)

def update_global_best(new_Y, func):
    '''

    '''
    fitness = [func(y) for y in new_Y]
    min_idx = np.argmin(fitness)
    new_global_best = new_Y[min_idx]
    return new_global_best
