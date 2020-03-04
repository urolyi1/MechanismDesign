import numpy as np
import torch
import itertools
import match_net_torch as mn
import util



class Experiment:
    def __init__(self, args, internal_S, n_hos, n_types, model, dir='results/'):
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
        self.central_S = torch.tensor(util.convert_internal_S(self.int_S.numpy(), n_hos),
                                      requires_grad=False, dtype=torch.float32)

        # Setting some parameters
        self.int_structs = self.int_S.shape[1]
        self.num_structs = self.central_S.shape[1]
        self.n_hos = n_hos
        self.n_types = n_types

        self.model_args = args

        self.model = model

        self.dir = dir

    def run_experiment(self, train_batches, test_batches, save=False):
        batch_size = train_batches.shape[1]

        # Train model on training sample
        self.train_tuple = mn.train_loop(train_batches, self.model, batch_size, self.central_S, self.n_hos,
                                         self.n_types, net_lr=self.model_args.main_lr, main_iter=self.model_args.main_iter,
                                         misreport_iter=self.model_args.misreport_iter,
                                         misreport_lr=self.model_args.misreport_lr)

        # Evaluate model on test_batches
        self.test_regrets, self.test_misreports = mn.test_model_performance(test_batches, self.model, batch_size,
                                                                            self.int_S, self.n_hos, self.n_types) # misreport_iter, misreport_lr

        if save:
            self.save_experiment(self.dir, train_batches, test_batches)

    def save_experiment(self, dir, train_batches, test_batches):
        batch_size = train_batches.shape[1]

        final_p, rgt_loss_lst, tot_loss_lst, util_loss_lst = self.train_tuple
        np.save(dir + 'util_loss.npy', util_loss_lst)
        np.save(dir + 'rgt_loss.npy', rgt_loss_lst)
        np.save(dir + 'tot_loss.npy', tot_loss_lst)

        # Actually look at the allocations to see if they make sense
        # print((model.forward(final_p[0], batch_size) @ central_s.transpose(0, 1)).view(batch_size, 2, 2))
        # print(final_p[0])

        # Save model and results on train/test batches
        self.model.save(filename_prefix=dir)
        np.save(dir + 'train_batches.npy', train_batches.numpy())

        np.save(dir + 'test_batches.npy', test_batches.numpy())

        final_train_regrets, _ = mn.test_model_performance(train_batches, self.model, batch_size, self.model.S,
                                                           self.model.n_hos, self.model.n_types,
                                                           misreport_iter=self.model_args.misreport_iter,
                                                           misreport_lr=1.0)

        test_regrets, test_misreports = mn.test_model_performance(test_batches, self.model, batch_size, self.model.S,
                                                                  self.model.n_hos, self.model.n_types,
                                                                  misreport_iter=1000, misreport_lr=1.0)

        torch.save(test_regrets, dir + 'test_batch_regrets.pytorch')
        torch.save(final_train_regrets, dir + 'train_batch_regrets.pytorch')

