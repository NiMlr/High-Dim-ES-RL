'''This module contains the following:

Controller
  A class for Keras (Tensorflow backend) based OpenAI gym controllers.
Models
  A class implementing and supplying Keras models to the Controller
  class.
ActionTransformations
  A container class for methods that transform the controller (Keras model)
  output (action) to a representation suitable for the OpenAI gym
  environment.
action_transformations
  A dictionary that links the action transformation to the specific
  environment name. The Controller.fitness method accesses this
  dictionary.
EarlyStop
  A class containing a method and dictionary that enable
  the controller.fitness evaluation to be prematurely terminated if a
  candidate controllers performance is poor. This reduces computational cost.
'''
import numpy as np
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.utils import plot_model


class Controller(object):
    '''Class for Keras (Tensorflow backend) based OpenAI gym controllers.'''

    def __init__(self, modelFunctionHandle, env,
                 episode_length, device='/cpu:0', render=False,
                 force_action_space=None):
        '''Initialize a controller.

        Args:
            modelFunctionHandle (function):
                Function that returns a keras model and tensorflow default
                graph (threading). The function takes the input and output
                dimensions of the keras model as an argument.
            env (str):
                A OpenAI gym envrionment name.
            episode_length (int):
                Number of frames to process.
            device (str, optional):
                String that specifies the tensorflow device to use.
            render (bool, optional):
                Boolean to indicate whether the environment is rendered.
            force_action_space (int, optional):
                Whenever the gym environment is not correctly implemented
                ( `type(env.action_space_low)!=np.ndarray` ) use this input to
                manually specify action_space dimension.
         '''
        # get obs/act space dims. was not always correctly implemented in gym
        # hence the uglyness
        self.env = gym.make(env)
        self.observation_space_low = self.env.observation_space.low
        self.action_space_low = self.env.action_space.sample()

        # get the model
        if type(self.action_space_low) == np.ndarray:
            self.model, self.graph = modelFunctionHandle(
                self.observation_space_low.shape[0], len(self.action_space_low))
        # whenever gym would not work properly, set using additional parameter
        else:
            self.model, self.graph = modelFunctionHandle(
                self.observation_space_low.shape[0], force_action_space)
        self.stacked_weights = self.model.get_weights()

        # save some useful things
        self.env_name = env
        self.episode_length = episode_length
        self.device = device
        self.frame_count = 0
        self.render = render
        # save weight sizes for output as column vector
        self.weight_sizes = [(x.shape, x.size) for x in self.stacked_weights]
        # save the dimension by simply adding all sizes
        self.n = sum([x.size for x in self.stacked_weights])

    def fitness(self, flat_weights):
        '''Sample the cumulative return of one episode acting according to
        current weights.

        Args:
            flat_wights (numpy.ndarray): Vector of length self.n specifying the
                weights of the controller to sample.

        Returns:
            float: Cumulative reward after an episode
               of length self.episode_length.
        '''

        # convert weight vector to keras structure and set
        self.set_weights(flat_weights)
        # reset environment
        observation = self.env.reset()

        fitness = 0
        # loop over steps
        for step in range(self.episode_length):
            # check rendering
            if self.render:
                self.env.render()

            # be sure to use preferred device
            with tf.device(self.device):
                # resolves error in multithreading
                with self.graph.as_default():
                    # get controller output
                    action = self.model.predict(np.array([observation]))

            # convert action to gym format
            action = action_transformations[self.env_name](action)
            # act
            observation, reward, done, info = self.env.step(action)

            fitness += reward
            self.frame_count += 1
            # check for early stopping
            if done or EarlyStop.check(step, fitness, self.env_name):
                # inverse fitness for minimizing algorithms
                return -fitness
        # inverse fitness for minimizing algorithms
        return -fitness

    def set_weights(self, flat_weights):
        '''Convert the weight vector from optimizer friendly format
        to a layerwise representation. Use this to set model weights to.

        Args:
            flat_weights (numpy.ndarray): A vector of shape (self.n,) holding
                the weights the controller should be set to.
        '''
        # resolves threading error
        with self.graph.as_default():
            i = 0
            j = 0
            # get layer representation
            for weight_size in self.weight_sizes:
                self.stacked_weights[j] = np.reshape(
                    flat_weights[i:i+weight_size[1]], weight_size[0])
                j += 1
                i += weight_size[1]
            # set keras model weights
            self.model.set_weights(self.stacked_weights)

    def get_weights(self):
        '''Just a wrapper for the standard methods that returns the
        stacked (layerwise) weights of the Keras model.

        Returns:
            Stacked model weights.
        '''
        return self.model.get_weights()


class Models(object):
    '''Container for methods that return a Keras model and the
    tensorflow default graph.

    The method must take the dimensionality of the state space as well
    as the dimensionality of the action space as arguments.
    '''
    @staticmethod
    def smallModel(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(10, input_dim=input_dim))
        model.add(Activation('elu'))
        model.add(Dense(output_dim))
        model.add(Activation('sigmoid'))

        # resolves error in multithreading
        graph = tf.get_default_graph()
        return model, graph
 
    @staticmethod
    def bipedalModel(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(30, input_dim=input_dim))
        model.add(Activation('elu'))
        model.add(Dense(30))
        model.add(Activation('elu'))
        model.add(Dense(15))
        model.add(Activation('elu'))
        model.add(Dense(10))
        model.add(Activation('elu'))
        model.add(Dense(output_dim))
        model.add(Activation('sigmoid'))

        # resolves error in multithreading
        graph = tf.get_default_graph()
        return model, graph

    @staticmethod
    def robopongModel(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(30, input_dim=input_dim))
        model.add(Activation('elu'))
        model.add(Dense(30))
        model.add(Activation('elu'))
        model.add(Dense(15))
        model.add(Activation('elu'))
        model.add(Dense(10))
        model.add(Activation('elu'))
        model.add(Dense(output_dim))
        model.add(Activation('sigmoid'))

        graph=tf.get_default_graph()
        return model, graph

    @staticmethod
    def acrobotModel(input_dim, output_dim):
        input_dim=input_dim[0]
        model=Sequential()
        model.add(Dense(30, input_dim=input_dim))
        model.add(Activation('elu'))
        model.add(Dense(30))
        model.add(Activation('elu'))
        model.add(Dense(10))
        model.add(Activation('elu'))
        model.add(Dense(output_dim))
        model.add(Activation('sigmoid'))

        graph=tf.get_default_graph()
        return model, graph


class ActionTransformations(object):
    '''Container for methods that transform the controller (Keras model)
    output (action) to a representation suitable for the OpenAI gym
    environment.

    Typically the method is implemented to suit a specific
    controller-environment configuration.
    '''
    @staticmethod
    def cartPoleV0(action):
        return int(action[0, 0])

    @staticmethod
    def carRacingV0(action):
        return action[0]

    @staticmethod
    def bipedalWalkerV2(action):
        return (action[0]-[0.5, 0.5, 0.5, 0.5])*2

    @staticmethod
    def breakoutRamV0(action):
        return np.argmax(action[0])

    @staticmethod
    def roboschoolPongV1(action):
        return (action[0]-[0.5, 0.5])*2

    @staticmethod
    def acrobotV1(action):
        #print(action,np.argmax(action[0]))
        return np.argmax(action[0])


action_transformations={'CartPole-v0': ActionTransformations.cartPoleV0,
                          'CarRacing-v0': ActionTransformations.carRacingV0,
                          'BipedalWalker-v2': ActionTransformations.bipedalWalkerV2,
                          'Breakout-ram-v0': ActionTransformations.breakoutRamV0,
                          'RoboschoolPong-v1': ActionTransformations.roboschoolPongV1,
                          'Acrobot-v1': ActionTransformations.acrobotV1}
'''dict: Links the action transformation to the specific environment name.
   The fitness method accesses this dictionary.'''


class EarlyStop(object):
    '''Contains a method and dictionary that enable
    the controller.fitness evaluation to be prematurely terminated if a
    candidate controllers performance is poor. This reduces computational cost.

    If a given controller falls short of reaching a the specified cumulative
    reward within the corresponding number of timesteps, the evaluation
    in controller.fitness is prematurely terminated in order to reduce
    the runtime. The interface to the controller.fitness is given by the
    EarlyStop.check method.
    '''
    step_fitness_dict = {'CartPole-v0': [],
                         'CarRacing-v0': [],
                         'Breakout-ram-v0': [],
                         'BipedalWalker-v2': [(190, 15), (300, 30), (400, 40), (600, 50),
                                              (700, 65), (800, 80)],
                         'RoboschoolPong-v1': [],
                         'Acrobot-v1': []}
    '''dict: A dictionary specifying corresponding fitness and
    time-step thresholds for the envrionments.'''

    @classmethod
    def check(cls, step, fitness, env_name):
        '''The interface to the controller.fitness.

        Here the check is performed.

        Args:
            step (int): The current time-step.
            fitness (float): The current cumulative reward.
            env_name (str): The environments name.

        Returns:
            bool: Indicating whether evaluation should be prematurely
                 terminated.
        '''
        for i in range( len(cls.step_fitness_dict[env_name]) ):
            if ( step > cls.step_fitness_dict[env_name][i][0] ) and\
                    ( fitness < cls.step_fitness_dict[env_name][i][1] ):
                return True
        return False
