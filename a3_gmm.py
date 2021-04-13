from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import sys

dataDir = "/u/cs401/A3/data/"
# dataDir = "/scratch/ssd001/home/cchoquet/csc401/a3/code/data/data/"


def sumexp(a, b, axis=0):
    return np.exp(a-b).sum(axis=axis, keepdims=True)


def compute_logs(x, M, theta, just_bs=False):
    log_bs = np.array([log_b_m_x(m, x, theta) for m in range(M)])
    if just_bs:
        return log_bs
    log_ps = np.array(log_p_m_x(log_bs, theta))
    return log_bs, log_ps


def update_theta(t, x, log_ps):

    ps = np.exp(log_ps)
    maxp = log_ps.max(1, keepdims=True)
    t.reset_mu((ps @ x) / (maxp + sumexp(log_ps, maxp, 1)))
    t.reset_omega(np.mean(ps, 1))

    sigma = (ps @ np.power(x, 2)) / (maxp + sumexp(log_ps, maxp, 1))
    sigma += 1e-9 - np.power(t.mu, 2)
    t.reset_Sigma(sigma)
    # print(f"omega: {t.omega}, sigma: {t.Sigma}, mu: {t.mu}")
    return t


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
        self.precompute = None

    def reset_precompute(self):
        precomp = (np.power(self.mu, 2) / (2 * self.Sigma)).sum(axis=1)
        precomp += (self._d / 2) * np.log(2 * np.pi)
        precomp += np.log(self.Sigma).sum(axis=1) / 2
        self.precompute = precomp

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        return self.precompute[m]

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
    mu = myTheta.mu[m]
    sigma = np.expand_dims(myTheta.Sigma[m], 0)
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    log_bmx = - np.einsum('ij,ji->i', x / sigma, x.T) / 2
    log_bmx += (x / sigma) @ mu
    log_bmx -= myTheta.precomputedForM(m)
    # print(log_bmx)
    return log_bmx


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)
    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)
    For further information, See equation 2 of handout
    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]
    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    alllog = log_Bs + np.log(myTheta.omega)
    logmax = alllog.max(axis=0, keepdims=True)
    log_pmx = alllog - logmax - np.log(sumexp(alllog, logmax))
    # print(log_pmx)
    return log_pmx


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x
        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).
        We don't actually pass X directly to the function because we instead pass:
        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.
        See equation 3 of the handout
    """
    alllog = log_Bs + np.log(myTheta.omega)
    logmax = alllog.max(axis=0, keepdims=True)
    # print(f"alllog: {alllog}, logmax: {logmax}")
    return (logmax + np.log(sumexp(alllog, logmax))).sum()


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # set omega
    omega = np.exp(np.random.rand(M, 1))
    myTheta.reset_omega(omega / omega.sum())
    # set mu
    myTheta.reset_mu(X[np.random.randint(0, X.shape[0], M)])  # pick random points
    # set sigma
    sig_shape = (M, X.shape[1])
    sig = np.reciprocal(np.arange(1, M+1).astype(np.float))
    myTheta.reset_Sigma(np.broadcast_to(np.expand_dims(sig, 1), sig_shape))
    # print(myTheta.omega)
    # print(myTheta.mu)
    # print(myTheta.Sigma)
    i = 0
    prev_l = -np.inf
    delta = np.inf
    while i < maxIter and delta >= epsilon:
        myTheta.reset_precompute()
        log_bs, log_ps = compute_logs(X, M, myTheta)
        l = logLik(log_bs, myTheta)
        # print(f"l: {l}, prev_l: {prev_l}")
        myTheta = update_theta(myTheta, X, log_ps)
        delta = l - prev_l
        prev_l = l
        i += 1
        # print(f"iteration {i} done with l: {round(l, 3)} and delta: {round(delta, 3)}")
    myTheta.reset_precompute()  # 1 last one now that we're done.
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
    loglikes = []
    M = len(models[0].omega)
    for model in models:
        log_bs = compute_logs(mfcc, M, model, just_bs=True)
        l = logLik(log_bs, model)
        loglikes.append(l)
    # print(loglikes)
    best_indices = np.argsort(-np.array(loglikes))  # -'ves to reverse order
    bestModel = best_indices[0]
    if k > 0:
        print(models[correctID].name)
        top_k = best_indices[:k]
        kmodels = [models[i] for i in top_k]
        klogs = [loglikes[i] for i in top_k]
        for model, loglike in zip(kmodels, klogs):
            print(f"{model.name} {loglike}")
        print("")
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    np.random.seed(2)
    random.seed(2)
    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 6
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
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
    stdout = sys.stdout  # steal stdout so that we can redirect to file.
    print(f"accuracy: {accuracy}")
