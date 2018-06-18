'''This module contains the following:

BaseOptimizer
  A class containing useful methods that are inherited by any optimizer.
  Generally it is an attempt, to make the code more readable.
  
Todo:
    * Add automatic verbose report.'''
import sys


class BaseOptimizer(object):
    '''Baseclass with useful methods.

    Inherited by all optimizers.'''

    @staticmethod
    def reachedFunctionTarget(function_target, candidate_fitness):
        '''Check if fitness is below function_target.

        ... and maybe save some lines of code.
        Args:
                function_target (numeric):
                        Target function value f(y*).
                candidate_fitness (numeric):
                        Function value of the candidate f(c).

        Returns:
                bool: A boolean in indicating if function_target is reached.
        '''
        if function_target != None:
            return (candidate_fitness <= function_target)
        else:
            return False

    @staticmethod
    def reachedFunctionBudget(function_budget, function_evals):
        '''Check if maximum number of function evaluations is reached.

        ... and maybe save some lines of code.
        Args:
                function_budget (int):
                        Budget of function evaluations.
                function_evals (int):
                        Function evaluations executed.

        Returns:
                bool: A boolean in indicating if function_budget is reached.
        '''
        if function_budget != None:
            return (function_evals >= function_budget)
        else:
            return False

    @staticmethod
    def report(string_to_print):
        '''Report current state.

        Makes a nice user interface.

        **Maybe extend to verbose/non-verbose setting.**

        Args:
                string_to_print (str):
                        String to be printed.
        '''
        sys.stdout.write(string_to_print)
        sys.stdout.flush()
