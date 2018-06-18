'''This module contains the following:

BenignEllipse
  A moderately conditioned function.
BenignEllipseNoisyThres
  A moderately conditioned function with additive noise above a certain
  (function value) threshold.
BenignEllipseAddNoise
  A moderately conditioned function with additive noise of a specified
  strength applied.
BenignEllipseMultNoise
  A moderately conditioned function with multiplicative noise of a specified
  strength applied.
Ellipse
  A badly conditioned function.
EllipseAddNoise
  A badly conditioned function with additive noise of a specified
  strength applied.
EllipseMultNoise
  A badly conditioned function with multiplicative noise of a specified
  strength applied.
sphere
  The standard spherical quadratic function.
SphereAddNoise
  The standard spherical quadratic function with additive noise of a
  specified streght applied.
SphereMultNoise
  The standard spherical quadratic function with multiplicative noise of a
  specified streght applied.
sigmoidal
  A sigmoidal function thresholding large values.
'''
import numpy as np
import sys


class BenignEllipse(object):
    '''Implements a quadratic function of condition number 1e2.'''
    def __init__(self, n):
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e2, i / (n - 1))

    def __call__(self, x):
        return np.sqrt(np.dot(x, self.elli_factors * x))


class BenignEllipseNoisyThres(object):
    '''Implements a quadratic function of condition number 1e2
    with a noise strength proportional to the problem dimension

    The noise is applied only above a threshold of 3.5 (function value).'''
    def __init__(self, n, rng=np.random.RandomState()):
        self.n = n
        self.rng = rng
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e2, i / (n - 1))

    def __call__(self, x):
        f = np.sqrt(np.dot(x, self.elli_factors * x))

        if f > 3.5:
            f += (self.rng.randn()*(200/self.n))

        return f

class BenignEllipseAddNoise(object):
    '''An implementation of a quadratic function of condition number 1e6
    on which additive noise is applied.'''
    def __init__(self, n, noiseamp, rng=np.random.RandomState()):
        self.noiseamp = noiseamp
        self.rng = rng
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e2, i / (n - 1))
    def __call__(self, x):
        f_no_noise = np.sqrt(np.dot(x, self.elli_factors * x))
        return f_no_noise*self.rng.randn()*self.noiseamp

class BenignEllipseMultNoise(object):
    '''An implementation of a quadratic function of condition number 1e6
    on which multiplicative noise is applied.'''
    def __init__(self, n, noiseamp, rng=np.random.RandomState()):
        self.noiseamp = noiseamp
        self.rng = rng
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e2, i / (n - 1))
    def __call__(self, x):
        f_no_noise = np.sqrt(np.dot(x, self.elli_factors * x))
        return f_no_noise*sigmoidal(self.rng.randn()*self.noiseamp)


class Ellipse(object):
    '''Implements a quadratic function of condition number 1e6.'''
    def __init__(self, n):
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e6, i / (n - 1))

    def __call__(self, x):
        return np.sqrt(np.dot(x, self.elli_factors * x))

class EllipseAddNoise(object):
    '''An implementation of a quadratic function of condition number 1e6
    on which additive noise is applied.'''
    def __init__(self, n, noiseamp, rng=np.random.RandomState()):
        self.noiseamp = noiseamp
        self.rng = rng
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e6, i / (n - 1))
    def __call__(self, x):
        f_no_noise = np.sqrt(np.dot(x, self.elli_factors * x))
        return f_no_noise*self.rng.randn()*self.noiseamp

class EllipseMultNoise(object):
    '''An implementation of a quadratic function of condition number 1e6
    on which multiplicative noise is applied.'''
    def __init__(self, n, noiseamp, rng=np.random.RandomState()):
        self.noiseamp = noiseamp
        self.rng = rng
        self.elli_factors = np.zeros(n)
        for i in range(n):
            self.elli_factors[i] = np.power(1e6, i / (n - 1))
    def __call__(self, x):
        f_no_noise = np.sqrt(np.dot(x, self.elli_factors * x))
        return f_no_noise*sigmoidal(self.rng.randn()*self.noiseamp)

def sphere(x):
    '''Implements a quadratic function of condition number 1e0.'''
    return np.sqrt(np.dot(x, x))

class SphereAddNoise(object):
    '''An implementation of a quadratic function of condition number 1
    on which additive noise is applied.'''
    def __init__(self, n, noiseamp, rng=np.random.RandomState()):
        self.noiseamp = noiseamp
        self.rng = rng
    def __call__(self, x):
        f_no_noise = np.sqrt(np.dot(x, x))
        return f_no_noise*self.rng.randn()*self.noiseamp

class SphereMultNoise(object):
    '''An implementation of a quadratic function of condition number 1
    on which multiplicative noise is applied.'''
    def __init__(self, n, noiseamp, rng=np.random.RandomState()):
        self.noiseamp = noiseamp
        self.rng = rng
    def __call__(self, x):
        f_no_noise = np.sqrt(np.dot(x, x))
        return f_no_noise*sigmoidal(rng.randn()*self.noiseamp)

def sigmoidal(x):
    '''A sigmoidal function thresholding large values.'''
    return 2 / (1 + np.exp(-x))
    
