import numpy as np


class RealisticHospital:
    def __init__(self, k, with_tissue=False):
        # [O, A, B, AB] Probabilities
        self.bloodtypes = ['', 'A', 'B', 'AB']
        self.bloodtype_probs = [0.08, 0.34, 0.14, 0.44]

        # Tissue incompatibilitiy probabilities
        self.tissue_probs = [.7, .2, .1]

        # Tissue incompat values
        self.tissue_vals = [0.05, 0.45, 0.9]
        self.patients = k
        self.tissue = with_tissue

    def generate(self, batch_size):
        num_types = len(self.bloodtype_probs)
        num_tissue = len(self.tissue_probs)

        if self.tissue:
            # patient type, donor_type, patient tissue_type, donor tissue type
            bids = np.zeros((batch_size, num_types, num_types, num_tissue, num_tissue))
            for i in range(batch_size):
                for j in range(self.patients):
                    bids[(i,) + self.generate_pair()] += 1.0
        else:
            # patient type, donor_type,
            bids = np.zeros((batch_size, num_types, num_types))
            for i in range(batch_size):
                for j in range(self.patients):
                    bids[(i,) + self.generate_pair()[:2]] += 1.0

        return bids.reshape((batch_size, -1))

    def generate_pair(self):
        incompat = False
        ret_tuple = None
        while (not incompat):
            tissue_idx = np.random.choice(len(self.tissue_probs), p=self.tissue_probs, size=2)
            p_tissue_idx = tissue_idx[0]
            d_tissue_idx = tissue_idx[1]

            blood_idx = np.random.choice(len(self.bloodtype_probs), p=self.bloodtype_probs, size=2)
            p_blood_idx = blood_idx[0]
            d_blood_idx = blood_idx[1]

            if (self.tissue_vals[d_tissue_idx] < self.tissue_vals[p_tissue_idx]
                    or not self.is_blood_compat(p_blood_idx, d_blood_idx)):
                incompat = True
                ret_tuple = (p_blood_idx, d_blood_idx, p_tissue_idx, d_tissue_idx)
        return ret_tuple

    def is_blood_compat(self, p_idx, d_idx):
        p_type = self.bloodtypes[p_idx]
        d_type = self.bloodtypes[d_idx]
        for ch in d_type:
            if ch not in p_type:
                return False
        return True


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