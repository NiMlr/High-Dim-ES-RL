
'''This module contains the following:

LMMAES
  An ES for problems in dimensions >> 100.
MAES
  An ES for problems in dimensions  > 100.
ES
  A stripped down version of the LMMAES implementation.
  Featuring no CMA or approximation. ES is reasonable to use
  in extremely high dimension.
'''

import numpy as np
from multiprocessing.dummy import Pool
from base import BaseOptimizer


class LMMAES(BaseOptimizer):
    '''LM-MA-ES for black box optimization.

    A limited memory(/time) version of CMA-ES. Useful for badly conditioned
    functions with high dimensional real parameter spaces.

    Reference:
            https://arxiv.org/pdf/1705.06693.pdf
    '''

    def __init__(self, y0, sigma, f, function_budget=10000, function_target=None,
                 rng=np.random.RandomState(), threads=1, lmbd=None):
        '''Initialization of the LMMAES

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
                algorithm	does not terminate automatically.

        function_target (numeric, optional): 
                Target function value f(y*). If function_budget and function_target
                are not specified the algorithm	does not terminate automatically.

                rng (class instance, optional): 
                        Random number generator similar to numpy's np.random.RandomState().
                        Requires at least a method similar to np.randn.

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
        self.n = len(y0)
        if lmbd != None:
            self.lmbd = lmbd
        else:
            self.lmbd = int(4 + np.floor(3 * np.log(self.n)))

        # otherwise tuning constants break - use standard CMA-ES instead :)
        assert self.lmbd < self.n
        self.mu = self.lmbd//2
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i + 1)
                           for i in range(self.mu)])
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
        self.M = np.zeros((self.m, self.n))

        # useful values
        self.c_sigma_update = np.sqrt(self.mu_w*self.c_sigma*(2-self.c_sigma))
        self.c_c_update = np.sqrt(self.mu_w*self.c_c*(2-self.c_c))
        self.fd = np.zeros((self.lmbd,))

        # deviation from the paper
        # damping constant
        self.d_sigma = 2
        # ~ expected length of normally distributed vector
        self.chi = np.sqrt(self.n) * (1 - (1/(4*self.n)) -
                                      (1/(21*self.n*self.n)))

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
            self.d = ((1 - self.c_d[j]) * self.d) + (self.c_d[j] *
                                                     np.outer(np.dot(self.d, self.M[j, :]), self.M[j, :]))

        # evaluate offspring and check stopping criteria
        self.x = [(self.y + self.sigma * self.d[i, :])
                  for i in range(self.lmbd)]
        self.fd = self.pool.map(self.f, self.x)
        self.function_evals += self.lmbd

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent
            return self.function_evals, self.y, 'B'

        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, np.mean(self.fd)):
                # if function target is reach return population expected value
                return self.function_evals, self.y, 'T'

        # sort by fitness
        self.order = np.argsort(self.fd)

        # update mean
        for i in range(self.mu):
            self.y += self.sigma * self.w[i] * self.d[self.order[i], :]

        # compute weighted mean
        self.wz = 0
        for i in range(self.mu):
            self.wz += self.w[i] * self.z[self.order[i], :]

        # update evolution path
        self.p_sigma *= 1 - self.c_sigma
        self.p_sigma += self.c_sigma_update * self.wz

        # update direction vectors
        for i in range(self.m):
            self.M[i, :] = ((1 - self.c_c[i]) * self.M[i, :]) + \
                (self.c_c_update[i] * self.wz)

        # update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) *
                             ((np.square(np.linalg.norm(self.p_sigma)) / self.n) - 1))

        # generation counter
        self.t += 1

        return self.function_evals, False, False


class MAES(BaseOptimizer):
    '''MA-ES for black box optimization.

        Reference:
                 https://arxiv.org/pdf/1705.06693.pdf
    '''

    def __init__(self, y0, sigma, f, function_budget=10000, function_target=None,
                 rng=np.random.RandomState(), threads=1, lmbd=None):
        '''Initialization of the LMMAES

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
                algorithm	does not terminate automatically.

        function_target (numeric, optional): 
                Target function value f(y*). If function_budget and function_target
                are not specified the algorithm	does not terminate automatically.

        rng (class instance, optional): 
                Random number generator similar to numpy's np.random.RandomState().
                Requires at least a method similar to np.randn.

        threads (int, optional):
                The number of threads to use to evalutate candidates.

        lmbd (int, optional):
                Number of evolution paths, the rank of the covariance
                matrix approximation. The value is tied to the number of 
                selected candidates by self.mu = self.lmbd//2, as well as 
                equal to the number of candidates self.m.
                Setting this manually might offset some constants.
'''
        # initializes self.save_to and self.buffer_length and
        # if required self.log, self.log_iterator
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
        self.n = len(y0)
        if lmbd != None:
            self.lmbd = lmbd
        else:
            self.lmbd = int(4 + np.floor(3 * np.log(self.n)))

        # otherwise tuning constants break - use standard CMA-ES instead :)
        assert self.lmbd < self.n
        self.mu = self.lmbd//2
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i + 1)
                           for i in range(self.mu)])
        self.w /= np.sum(self.w)
        self.mu_w = 1 / np.sum(np.square(self.w))

        self.c_sigma = (self.mu_w+2)/(self.n+self.mu_w+5)
        self.c_1 = 2/(np.power((self.n + 1.3), 2)+self.mu_w)
        self.c_mu = min(1-self.c_1, 2*(self.mu_w-2+(1/self.mu_w)
                                       )/(np.power(self.n+2, 2)+self.mu_w))

        # 2: initialize
        self.t = 0
        self.y = y0
        self.f = f
        self.sigma = sigma
        self.p_sigma = np.zeros((self.n,))
        self.M = np.identity(self.n)

        # useful values
        self.c_sigma_update = np.sqrt(self.mu_w*self.c_sigma*(2-self.c_sigma))
        self.fd = np.zeros((self.lmbd,))

        # deviation from the paper
        # damping constant
        self.d_sigma = 2
        # ~ expected lengthof normal distributed vector
        self.chi = np.sqrt(self.n) * (1 - (1/(4*self.n)) -
                                      (1/(21*self.n*self.n)))

    def step(self):
        '''Optimization step of the MAES.

        Returns:
                Tuple of (function_evals, False, False). If terminated a
                Tuple of (function_evals, y, a_flag). a_flag is a letter
                specifying the termination criterion. Either 'B' or 'T'.
        '''
        # sample offspring, vectorized version
        self.z = self.rng.randn(self.n, self.lmbd)
        self.d = np.matmul(self.M, self.z)

        # evaluate offspring and check stopping criteria
        self.x = [(self.y + self.sigma * self.d[:, i])
                  for i in range(self.lmbd)]
        self.fd = self.pool.map(self.f, self.x)
        self.function_evals += self.lmbd

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent
            return self.function_evals, self.y, 'B'

        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, np.mean(self.fd)):
                # if function target is reached, return population expected value
                return self.function_evals, self.y, 'T'

        # sort by fitness
        self.order = np.argsort(self.fd)

        # update mean
        for i in range(self.mu):
            self.y += self.sigma * self.w[i] * self.d[:, self.order[i]]

        # pre-compute
        self.d_sigma_M = np.dot(self.M, self.p_sigma)

        # compute weighted mean
        self.wz = 0
        temp2 = np.zeros((self.n, self.n))
        for i in range(self.mu):
            temp = self.w[i] * self.z[:, self.order[i]]
            self.wz += temp
            temp2 += np.outer(self.d[:, self.order[i]], temp)

        # update evolution path
        self.p_sigma *= 1 - self.c_sigma
        self.p_sigma += self.c_sigma_update * self.wz

        # update matrix
        self.M *= (1-(self.c_1/2)-(self.c_mu/2))
        self.M += (np.outer((self.c_1/2)*self.d_sigma_M,
                            self.p_sigma))+((self.c_mu/2)*temp2)

        # update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) *
                             ((np.square(np.linalg.norm(self.p_sigma)) / self.n) - 1))

        # generation counter
        self.t += 1

        return self.function_evals, False, False


class ES(BaseOptimizer):
    '''ES for black box optimization.
    '''

    def __init__(self, y0, sigma, f, function_budget=10000, function_target=None,
                 rng=np.random.RandomState(), threads=1):
        '''Initialization of the ES.

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
                algorithm	does not terminate automatically.

        function_target (numeric, optional): 
                Target function value f(y*). If function_budget and function_target
                are not specified the algorithm	does not terminate automatically.

        rng (class instance, optional): 
                Random number generator similar to numpy's np.random.RandomState().
                Requires at least a method similar to np.randn.

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
        self.n = len(y0)
        self.lmbd = int(4 + np.floor(3 * np.log(self.n)))

        # otherwise tuning constants break - use standard CMA-ES instead :)
        assert self.lmbd < self.n
        self.mu = self.lmbd//2
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i + 1)
                           for i in range(self.mu)])
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
        self.c_sigma_update = np.sqrt(self.mu_w*self.c_sigma*(2-self.c_sigma))
        self.fd = np.zeros((self.lmbd,))

        # deviation from the paper
        # damping constant
        self.d_sigma = 2
        # ~ expected lengthof normal distributed vector
        self.chi = np.sqrt(self.n) * (1 - (1/(4*self.n)) -
                                      (1/(21*self.n*self.n)))

    def step(self):
        '''Optimization step of the ES.

        Returns:
                Tuple of (function_evals, False, False). If terminated a
                Tuple of (function_evals, y, a_flag). a_flag is a letter
                specifying the termination criterion. Either 'B' or 'T'.
        '''
        # sample offspring, vectorized version
        self.z = self.rng.randn(self.lmbd, self.n)

        # evaluate offspring and check stopping criteria
        self.x = [(self.y + self.sigma * self.z[i, :])
                  for i in range(self.lmbd)]
        self.fd = self.pool.map(self.f, self.x)
        self.function_evals += self.lmbd

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent
            return self.function_evals, self.y, 'B'

        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, np.mean(self.fd)):
                # if function target is reached, return population expected value
                return self.function_evals, self.y, 'T'

        # sort by fitness
        self.order = np.argsort(self.fd)

        # update mean
        for i in range(self.mu):
            self.y += self.sigma * self.w[i] * self.z[self.order[i], :]

        # compute weighted mean
        self.wz = 0
        for i in range(self.mu):
            self.wz += self.w[i] * self.z[self.order[i], :]

        # update evolution path
        self.p_sigma *= 1 - self.c_sigma
        self.p_sigma += self.c_sigma_update * self.wz

        # update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) *
                             ((np.square(np.linalg.norm(self.p_sigma)) / self.n) - 1))

        # generation counter
        self.t += 1

        return self.function_evals, False, False
