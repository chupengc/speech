from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import numpy as np
import os, fnmatch
import random

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        pre1 = (self._d / 2) * np.log(2 * np.pi)
        pre2 = 0.5 * np.log(self.Sigma[m]).sum()

        return -pre1 - pre2

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    mu = myTheta.mu[m]  # (d, )
    sigma = myTheta.Sigma[m]  # (d, )
    d = mu.shape[1]
    const = myTheta.precomputedForM(m)
    if x.shape == (d,):
        term = (x - mu) ** 2 / 2 * sigma
        term = term.sum()  # (d, )
    else:
        mu = np.tile(mu, (x.shape[0], 1))  # (T, d)
        term = (x - mu) ** 2 / 2 * sigma
        term = term.sum(axis=1)  # (T, )

    return -term + const


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    m, t = log_Bs.shape
    omega = myTheta.omega  # (M, 1)
    log_omega = np.log(np.tile(omega, (1, t)))  # (M, T)
    term1 = log_omega + log_Bs
    log_sum = logsumexp(np.multiply(log_omega, log_Bs), axis=0)  # (T, )
    term2 = np.tile(log_sum, (m, 1))  # (M, T)

    return term1 - term2


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    m, t = log_Bs.shape
    omega = myTheta.omega
    log_omega = np.log(np.tile(omega, (1, t)))  # (M, T)
    log_sum = logsumexp(np.multiply(log_omega, log_Bs), axis=0)  # (T, )
    log_lik = log_sum.sum()

    return log_lik


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu,
    sigma) """
    myTheta = theta(speaker, M, X.shape[1])
    # initialization
    t, d = X.shape
    init_omega = np.array(1 / M for m in range(M))  # (M, )
    init_mu = np.array([X[np.random.randint(t)] for m in range(M)])  # (M, d)
    init_sigma = np.ones((M, d))  # (M, d)
    myTheta.reset_omega(init_omega)
    myTheta.reset_mu(init_mu)
    myTheta.reset_Sigma(init_sigma)

    i = 0
    prev_l = -np.inf
    improvement = np.inf
    while i < maxIter and improvement >= epsilon:
        log_Bs = np.array([log_b_m_x(m, X, myTheta) for m in range(M)])  # (M, t)
        log_Ps = log_p_m_x(log_Bs, myTheta)  # (M, t)
        l = logLik(log_Bs, myTheta)
        myTheta = update(myTheta, log_Ps, X, M)
        improvement = l - prev_l
        prev_l = l
        i += 1

    return myTheta


def update(myTheta, log_Ps, X, M):
    T, D = X.shape
    p = np.exp(log_Ps)  # (M, t)
    p_sum = p.sum(axis=1)  # (M, )

    # update omega
    updated_omega = p_sum / T  # (M, )
    myTheta.reset_omega(updated_omega)
    # update mu
    p_tile_mt = np.tile(p_sum, T).reshape(M, T)  # (M, T)
    p_tile_md = np.tile(p_sum, D).reshape(M, D)  # (M, D)
    updated_mu = np.dot(p_tile_mt, X) / p_tile_md  # (M, D)
    myTheta.reset_mu(updated_mu)
    #update sigma
    updated_sigma = np.dot(p_tile_mt, X ** 2) / p_tile_md - (updated_mu ** 2)
    myTheta.reset_Sigma(updated_sigma)

    return myTheta

def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    print("TODO")

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)),
                                   "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
