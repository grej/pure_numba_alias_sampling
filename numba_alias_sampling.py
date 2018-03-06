import numpy as np
import numba

K_IDX = 0
q_IDX = 1
J_IDX = 2


@numba.njit
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i]*lj))
        if r2[i] < q[kk]: res[i] = kk
        else: res[i] = J[kk]
    return res


@numba.njit
def prep_var_sample(probs):
    K = probs.size
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32) 
    smaller,larger  = [],[]
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0: smaller.append(kk)
        else: larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small,large = smaller.pop(),larger.pop()
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0: smaller.append(large)
        else: larger.append(large)    
    return(K, q, J)


@numba.njit
def draw_one(var_sample_prep):
    K,q,J = var_sample_prep[K_IDX], var_sample_prep[q_IDX], var_sample_prep[J_IDX]
    kk = int(np.floor(np.random.rand()*len(J)))
    if np.random.rand() < q[kk]: return kk
    else: return J[kk]
    

@numba.njit
def draw_n(n, var_sample_prep):
    K,q,J = var_sample_prep[K_IDX], var_sample_prep[q_IDX], var_sample_prep[J_IDX]    
    r1,r2 = np.random.rand(n),np.random.rand(n)
    return sample(n, q, J, r1, r2)
