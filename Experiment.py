import numpy as np
import torch
import itertools
import match_net_torch as mn
import util


class Experiment:
    def __init__(self, args, internal_S, n_hos, n_types, model):
        """
        Initialize Experiment

        :param args: model argument
        :param internal_S: Structure matrix for internal matches
        :param n_hos: number of hospitals
        :param n_types: number of types
        :param model: MatchNet model
        """
        # Create internal and central structure matrix
        self.int_S = internal_S
        self.central_S = util.convert_internal_S(self.int_S, n_hos)

        # Setting some parameters
        self.int_structs = self.int_S.shape[1]
        self.num_structs = self.central_S.shape[1]
        self.n_hos = n_hos
        self.n_types = n_types

        self.model_args = args

        self.model = model

    def run_experiment(self, train_batches):
        batch_size = train_batches.shape[1]
        train_tuple = mn.train_loop(train_batches, self.model, batch_size, self.internal_S,
                                    self.n_hos, self.n_types)

