'''This module contains the following:

AvWrapper
  Callable class for mapping candidate evaluation and averaging.
UHLMMAES
  An ES for problems in dimensions >> 100 under uncertainty.
UHES
  A stripped down version of the UHLMMAES implementation.
  Featuring no CMA or respective approximation. ES is reasonable to use
  in extremely high dimensions.
'''

import numpy as np
from base import BaseOptimizer
from multiprocessing.dummy import Pool


class AvWrapper(object):
    '''Callable class for mapping candidate evaluation and averaging.
    '''

    def __init__(self, averaging, f):
        '''Intializer function object.

        Args:
            averaging (int):
                Number of evaluation of function f for noise handling.
            f (function):
                Function handle for candidate evaluation.
        '''
        self.averaging = averaging
        self.f               = f
    def __call__(self, x):
        '''Call method for candidate evaluation including uncertainty handling.

        Args:
            x (numpy.ndarray):
                Candidate to evaluate.

        Returns:
            float: Indicating candidate average performance.
        '''
        return np.median([self.f(x) for _ in range(self.averaging)])


class UHLMMAES(BaseOptimizer):
    '''UH-LM-MA-ES for black box optimization under uncertainty.

    A limited memory(/time) version of CMA-ES. Useful for non-convex
    and noisy optimization for which gradient information is not available.

    Reference:
        https://arxiv.org/pdf/1705.06693.pdf
    '''

    def __init__(self, y0, sigma, f, function_budget=10000, function_target=None,
                 rng=np.random.RandomState(), threads=1, lmbd = None):
        '''Initialization of the LMMAES

        Args:
            y0 (np.ndarray): 
                Initial candidate solution. A numpy array of dimension n.
                Optimum should not be more distant than 3*step_size.

            sigma (float): 
                Global step size or mutation strength.

            f (function): 
                Fitness function, taking a candidate as input.

            function_budget (int, optional): 
                Maximum number of function evaluations. Defaults to 10000.
                If function_budget and function_target are not specified the
                algorithm    does not terminate automatically.

            function_target (numeric, optional): 
                Target function value f(y*). If function_budget and function_target
                are not specified the algorithm    does not terminate automatically.

            rng (class instance, optional): 
                Random number generator similar to numpy's np.random.RandomState().
                Requires at least methods similar to np.randn and np.randint.

            threads (int, optional):
                The number of threads to use to evalutate candidates.

            lmbd (int, optional):
                Number of evolution paths, the rank of the covariance
                matrix approximation. The value is tied to the number of 
                selected candidates by self.mu = self.lmbd//2, as well as 
                equal to the number of candidates self.m.
                Setting this manually might offset some constants.
        '''
        super().__init__()

        self.function_evals = 0

        # set if required
        self.function_budget = function_budget
        self.function_target = function_target

        # set random number generator
        self.rng = rng

        # initialize pool
        self.pool = Pool(threads)

        # 1: given
        self.n= len(y0)
        if lmbd != None:
            self.lmbd = lmbd
        else:
            self.lmbd = int(4 + np.floor(3 * np.log(self.n)))

        # otherwise tuning constants break - use standard CMA-ES instead :)
        assert self.lmbd < self.n
        self.mu = self.lmbd//2
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.w /= np.sum(self.w)
        self.mu_w = 1 / np.sum(np.square(self.w))

        self.m = self.lmbd

        self.c_sigma = (2*self.lmbd)/self.n

        self.c_d = np.zeros((self.m,))
        self.c_c = np.zeros((self.m,))

        for i in range(self.m):
            self.c_d[i] = 1 / (np.power(1.5, i) * self.n)
            self.c_c[i] = self.lmbd / (np.power(4.0, i) * self.n)

        # 2: initialize
        self.t = 0
        self.y = y0
        self.f = f
        self.sigma = sigma
        self.p_sigma = np.zeros((self.n,))
        self.M = np.zeros((self.m,self.n))

        # useful values
        self.c_sigma_update = np.sqrt( self.mu_w*self.c_sigma*(2-self.c_sigma) ) 
        self.c_c_update = np.sqrt( self.mu_w*self.c_c*(2-self.c_c) )
        self.fd = np.zeros((self.lmbd,))

        # deviation from the paper
        # damping constant
        self.d_sigma = 2

        # ~ expected length of normally distributed vector(not in use)
        self.chi = np.sqrt(self.n) * (1 - (1/(4*self.n)) + (1/(21*self.n*self.n)) )

        # number of generations till next uncertainty handling check
        self.uncertainty_handling = 1

        # number of fitness evaluations to average for uncertainty handling
        self.averaging_f = 1.0
        self.averaging = 1
        self.targetnoise = 0.12   # relative rank change
        self.S         = 0.12

    def step(self):
        '''Optimization step of the LMMAES.

        Returns:
            Tuple of (function_evals, False, False). If terminated a
            Tuple of (function_evals, y, a_flag). a_flag is a letter
            specifying the termination criterion. Either 'B' or 'T'.
        '''
        # sample offspring, vectorized version
        self.z = self.rng.randn(self.lmbd, self.n)
        self.d = np.copy(self.z)
        for j in range(min(self.t, self.m)):
            self.d = ((1 - self.c_d[j]) * self.d) + (self.c_d[j] * np.outer(np.dot(self.d, self.M[j,:]), self.M[j,:]))

        # evaluate offspring and check stopping criteria
        self.x = [(self.y + self.sigma * self.d[i,:]) for i in range(self.lmbd)]

        self.fd = self.pool.map(AvWrapper(self.averaging,self.f), self.x)
        self.function_evals += self.lmbd*self.averaging


        if self.reachedFunctionBudget( self.function_budget, self.function_evals ):
            # if budget is reached return parent
            return self.function_evals, self.y, 'B'

        if self.function_target!=None:            
            if self.reachedFunctionTarget(self.function_target, np.mean(self.fd)):
                # if function target is reach return population expected value
                return self.function_evals, self.y , 'T'


        # sort by fitness
        order = np.argsort(self.fd)

        # uncertainty handling
        self.uncertainty_handling -= 1
        if self.uncertainty_handling <= 0:
            # number of generations to wait, limiting the added cost to at most 5% = 2 / 40
            self.uncertainty_handling = int(np.ceil(40 / self.lmbd))

            # find two random individuals for re-evaluation
            fd2 = np.copy(self.fd)
            i1 = self.rng.randint(self.lmbd)
            i2 = self.rng.randint(self.lmbd - 1)
            if i2 >= i1:
                i2 += 1

            # re-evaluate i1
            z = self.y + self.sigma * self.d[i1,:]

            fd2[i1] = np.median([self.f(z) for _ in range(self.averaging)])
            self.function_evals += self.averaging

            # re-evaluate i2
            z = self.y + self.sigma * self.d[i2,:]

            fd2[i2] = np.median([self.f(z) for _ in range(self.averaging)])
            self.function_evals += self.averaging

            # sort by fitness
            order2 = np.argsort(fd2)

            # rankings
            rank1 = np.argsort(self.fd)
            rank2 = np.argsort(fd2)

            # compute rank difference statistics (inspired by Hansen 2008, but simplified)
            self.S = abs(rank1[i1] - rank2[i1]) + abs(rank1[i2] - rank2[i2])
            self.S /= 2 * (self.lmbd-1)

            # accumulate
            c_uh = max(1.0, 10.0 * self.lmbd / self.n)
            max_averaging = 1000.0   # hard bound...

            self.averaging_f *= np.exp(c_uh * (self.S - self.targetnoise))
            self.averaging_f = max(1.0, min(max_averaging, self.averaging_f))

            # adapt amount of averaging
            self.averaging = int(round(self.averaging_f))

            # incorporate additional fitness evaluations
            self.fd[i1] = 0.5 * (self.fd[i1] + fd2[i1])
            self.fd[i2] = 0.5 * (self.fd[i2] + fd2[i2])
            order = np.argsort(self.fd)

        # update mean
        for i in range(self.mu):
            self.y += self.sigma * self.w[i] * self.d[order[i],:]

        # compute weighted mean
        self.wz = 0
        for i in range(self.mu):
            self.wz += self.w[i] * self.z[order[i],:]

        # update evolution path
        self.p_sigma *= 1 - self.c_sigma
        self.p_sigma += self.c_sigma_update * self.wz

        if self.S<=self.targetnoise:
            # update direction vectors
            for i in range(self.m):
                self.M[i,:] = ((1 - self.c_c[i]) *self.M[i,:]) + (self.c_c_update[i] * self.wz)

        # update step size
        # if self.S<=self.targetnoise:
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.dot(self.p_sigma, self.p_sigma) / self.n - 1))

        # generation counter
        self.t += 1

        # number of evals, approx optimum, termination flag
        return self.function_evals, False, False

class UHES(BaseOptimizer):
    '''UH-ES for black box optimization under uncertainty.
    '''

    def __init__(self, y0, sigma, f, function_budget=10000, function_target=None,
                 rng=np.random.RandomState(), threads=1):
        '''Initialization of UH-ES.

        Args:
            y0 (numpy.ndarray): 
                Initial candidate solution. A numpy array of dimension n.
                Optimum should not be more distant than 3*step_size.

            sigma (float): 
                Global step size or mutation strength.

            f (function): 
                Fitness function, taking a candidate as input.

            function_budget (int, optional): 
                Maximum number of function evaluations. Defaults to 10000.
                If function_budget and function_target are not specified the
                algorithm    does not terminate automatically.

            function_target (numeric, optional): 
                Target function value f(y*). If function_budget and function_target
                are not specified the algorithm    does not terminate automatically.

            rng (class instance, optional): 
                Random number generator similar to numpy's np.random.RandomState().
                Requires at least methods similar to np.randn and np.randint.

            threads (int, optional):
                The number of threads to use to evalutate candidates.
        '''
        super().__init__()

        self.function_evals = 0

        # set if required
        self.function_budget = function_budget
        self.function_target = function_target

        # set random number generator
        self.rng = rng

        # initialize pool
        self.pool = Pool(threads)

        # 1: given
        self.n= len(y0)
        self.lmbd= int(4 + np.floor(3 * np.log(self.n)))

        # otherwise tuning constants break - use standard CMA-ES instead :)
        assert self.lmbd < self.n
        self.mu = self.lmbd//2
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.w /= np.sum(self.w)
        self.mu_w = 1 / np.sum(np.square(self.w))

        self.c_sigma = (2*self.lmbd)/self.n

        # 2: initialize
        self.t = 0
        self.y = y0
        self.f = f
        self.sigma = sigma
        self.p_sigma = np.zeros((self.n,))

        # useful values
        self.c_sigma_update = np.sqrt( self.mu_w*self.c_sigma*(2-self.c_sigma) ) 
        self.fd = np.zeros((self.lmbd,))

        # deviation from the paper
        # damping constant
        self.d_sigma = 2
        # ~ expected lengthof normal distributed vector
        self.chi = np.sqrt(self.n) * (1 - (1/(4*self.n)) - (1/(21*self.n*self.n)) )
        # number of generations till next uncertainty handling check
        self.uncertainty_handling = 1

        # number of fitness evaluations to average for uncertainty handling
        self.averaging_f = 1.0
        self.averaging = 1
        self.targetnoise = 0.12   # relative rank change
        self.S         = 0.12


    def step(self):
        '''Optimization step of the UH-ES.

        Returns:
            Tuple of (function_evals, False, False). If terminated a
            Tuple of (function_evals, y, a_flag). a_flag is a letter
            specifying the termination criterion. Either 'B' or 'T'.
        '''
        # sample offspring, vectorized version
        self.z = self.rng.randn(self.lmbd, self.n)

        # evaluate offspring and check stopping criteria
        self.x = [(self.y + self.sigma * self.z[i,:]) for i in range(self.lmbd)]

        self.fd = self.pool.map(AvWrapper(self.averaging,self.f), self.x)
        self.function_evals += self.lmbd*self.averaging

        if self.reachedFunctionBudget( self.function_budget, self.function_evals ):
            # if budget is reached return parent
            return self.function_evals, self.y, 'B'

        if self.function_target!=None:                
            if self.reachedFunctionTarget(self.function_target, np.mean(self.fd)):
                # if function target is reach return population expected value
                return self.function_evals, self.y , 'T'


        # sort by fitness
        self.order = np.argsort(self.fd)

        # uncertainty handling
        self.uncertainty_handling -= 1
        if self.uncertainty_handling <= 0:
            # number of generations to wait, limiting the added cost to at most 5% = 2 / 40
            self.uncertainty_handling = int(np.ceil(40 / self.lmbd))

            # find two random individuals for re-evaluation
            fd2 = np.copy(self.fd)
            i1 = self.rng.randint(self.lmbd)
            i2 = self.rng.randint(self.lmbd - 1)
            if i2 >= i1:
                i2 += 1

            # re-evaluate i1
            z = self.y + self.sigma * self.z[i1,:]
            fd2[i1] = np.median([self.f(z) for _ in range(self.averaging)])
            self.function_evals += self.averaging

            # re-evaluate i2
            z = self.y + self.sigma * self.z[i2,:]
            fd2[i2] = np.median([self.f(z) for _ in range(self.averaging)])
            self.function_evals += self.averaging

            # sort by fitness
            order2 = np.argsort(fd2)

            # rankings
            rank1 = np.argsort(self.fd)
            rank2 = np.argsort(fd2)

            # compute rank difference statistics (inspired by Hansen 2008, but simplified)
            self.S = abs(rank1[i1] - rank2[i1]) + abs(rank1[i2] - rank2[i2])
            self.S /= 2 * (self.lmbd-1)

            # accumulate
            c_uh = max(1.0, 10.0 * self.lmbd / self.n)
            max_averaging = 1000.0   # hard bound...

            self.averaging_f *= np.exp(c_uh * (self.S - self.targetnoise))
            self.averaging_f = max(1.0, min(max_averaging, self.averaging_f))

            # adapt amount of averaging
            self.averaging = int(round(self.averaging_f))

            # incorporate additional fitness evaluations
            self.fd[i1] = 0.5 * (self.fd[i1] + fd2[i1])
            self.fd[i2] = 0.5 * (self.fd[i2] + fd2[i2])
            order = np.argsort(self.fd)

        # update mean
        for i in range(self.mu):
            self.y += self.sigma * self.w[i] * self.z[self.order[i],:]
        
        # compute weighted mean
        self.wz = 0
        for i in range(self.mu):
            self.wz += self.w[i] * self.z[self.order[i],:]

        # update evolution path
        self.p_sigma *= 1 - self.c_sigma
        self.p_sigma += self.c_sigma_update * self.wz

        # update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) *((np.square(np.linalg.norm(self.p_sigma)) / self.n) - 1))

        # generation counter
        self.t += 1

        return self.function_evals, False, False
