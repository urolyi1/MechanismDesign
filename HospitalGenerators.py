import numpy as np

class SingleHospital:
    def __init__(self, n_types, dist_lst):
        ''' Takes in number of pair types along with a list of functions that
        generate the number of people in that hospital with pair type.
        '''
        self.n_types = n_types
        self.dists = dist_lst
    def generate(self, batch_size):
        '''generate a report from this hospital'''
        X = np.zeros((batch_size, self.n_types))
        for i, dist in enumerate(self.dists):
            X[:, i] = dist(size=batch_size)
        return X


class ReportGenerator:
    def __init__(self, hos_lst, single_report_shape):
        self.hospitals = hos_lst
        self.single_shape = single_report_shape

    def generate_report(self, batch_size):
        X = np.zeros((batch_size,) + self.single_shape)
        for i, hos in enumerate(self.hospitals):
            X[:, i, :] = hos.generate(batch_size)
        yield X


def randint(low, high):
    return lambda size: np.random.randint(low, high, size)


def create_simple_generator(low_lst_lst, high_lst_lst, n_hos, n_types):
    ''' Creates a generator object to create batches'''
    hos_lst = []
    for h in range(n_hos):
        tmp_dist_lst = []
        for t in range(n_types):
            tmp_dist_lst.append(randint(low_lst_lst[h][t], high_lst_lst[h][t]))
        hos_lst.append(SingleHospital(n_types, tmp_dist_lst))
    gen = ReportGenerator(hos_lst, (n_hos, n_types))
    return gen