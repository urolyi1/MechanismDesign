import numpy as np

def softmax(vec):
    expon = np.exp(vec) 
    return expon / np.sum(expon)

def check_tot_pair(vec, num_pairs, num_hospitals, tot_pairs, show):
    valid = True    
    for p_type, total in enumerate(tot_pairs):
        if show:
            print("Pair type: {}".format(p_type))
            print("Allocated: {} Available: {}".format(vec[:, :, p_type].sum(), total))
        if vec[:, :, p_type].sum() > total:
            print ("OVER ALLOCATED FOR PAIR TYPE: {}".format(p_type))
            valid = False
    return valid

def check_hos_pair_combo(vec, num_pairs, num_hospitals, inputs, show):
    valid = True
    for h in range(num_hospitals):
        for p in range(num_pairs):
            allocated = vec[h, p, :].sum()
            need = inputs[h, p]
            if show:
                print("Hospital: {}, Pair type: {}".format(h, p))
                print("Allocated: {} Needed: {}".format(allocated, need))
            if allocated > need:
                print ("OVER ALLOCATED FOR HOSPITAL, PAIR: {},{}".format(h, p))
                valid = False
    return valid

def allocate(s, s_prime, num_hospitals, num_pairs, inputs, tot_pairs, show=False):
    ''' Run allocation scheme on output of s, s_prime to generate a final allocation'''
    # Normalizing s and s_prime using softmaxes
    norm_s = np.zeros((num_hospitals, num_pairs, num_pairs))
    norm_prime = np.zeros((num_hospitals, num_pairs, num_pairs))

    # softmax for each hospital pair combo to not over allocate than hospital needs
    for h in range(num_hospitals):
        for p in range(num_pairs):
            norm_s[h, p, :] = softmax(s[h, p, :]) * inputs[h, p] 

    # softmax over total pairs to not over allocate from total pool
    for p in range(num_pairs):
        norm_prime[:, :, p] = softmax(s_prime[:, :, p]) * tot_pairs[p]


    # Final alloc is the min between norm_s and norm_prime
    final_alloc = np.floor(np.minimum(norm_s, norm_prime))
    if show:
        print("---Final Allocation---")
        print(final_alloc)
    return final_alloc


def run_check(num_hospitals, num_pairs, show=False):
    # Generate Random Inputs
    inputs = np.random.randint(1, 100, size=(num_hospitals, num_pairs))
    # Calculate Total pool of each pair type
    tot_pairs = inputs.sum(axis=0)

    # Generate random output between 0 and 1 (sigmoid)
    s = np.random.rand(num_hospitals, num_pairs, num_pairs)
    s_prime = np.random.rand(num_hospitals, num_pairs, num_pairs)
    final_alloc = allocate(s, s_prime, num_hospitals, num_pairs, inputs, tot_pairs, show)
    if show:
        print("----Pair Allocation Check----")
    # Run over allocation checks
    valid = True
    if not check_tot_pair(final_alloc, num_pairs, num_hospitals, tot_pairs, show):
        valid = False
    if not check_hos_pair_combo(final_alloc, num_pairs, num_hospitals, inputs, show):
        valid = False
    if not valid:
        print(final_alloc)
        print(inputs)
    return valid
