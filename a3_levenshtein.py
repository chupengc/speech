import os
import numpy as np
import re

dataDir = '/u/cs401/A3/data/'


def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    # allocate matrix
    R = np.zeros((len(r) + 2, len(h) + 2))

    # add <s> and </s> to the start and end of sequence
    r.insert(0, "<s>")
    r.insert(len(r), "</s>")
    h.insert(0, "<s>")
    h.insert(len(h), "<s>")

    # initialization
    R[:, 0] = np.arange(len(R[:, 0]))
    R[0, :] = np.arange(len(R[0, :]))

    for i in range(len(R[:, 0]) - 1):
        for j in range(len(R[0, :]) - 1):
            delete = R[i, j + 1] + 1
            insert = R[i + 1, j] + 1
            if r[i + 1] == h[j + 1]:
                sub = R[i, j]
            else:
                sub = R[i, j] + 1

            R[i + 1, j + 1] = min(delete, insert, sub)

    R[-1, -1] -= 1

    # print(R)

    sub = 0
    insert = 0
    delete = 0
    # count the number of each error using backtracking
    i, j = R.shape[0] - 1, R.shape[1] - 1

    while i != 0 or j != 0:
        # check substitution
        if i > 0 and j > 0:
            i -= 1
            j -= 1
            if R[i, j] == R[i - 1, j - 1]:
                sub += 1
            elif R[i, j] == R[i - 1, j - 1] + 1:
                pass

        # check insertion
        elif j > 0 and R[i, j] == R[i, j - 1] + 1:
            insert += 1
            j -= 1

        # check deletion
        elif i > 0 and R[i, j] == R[i - 1, j] + 1:
            delete += 1
            i -= 1

    if len(r) - 2 == 0:
        wer = float("inf")
    else:
        wer = R[-1, -1] / (len(r) - 2)

    return round(wer, 3), sub, insert, delete


def preprocess(s):
    new_s = re.sub("[^a-zA-Z \[\]]", "", s)
    return new_s.lower()


if __name__ == "__main__":

    transcripts = ["transcripts.txt", "transcripts.Kaldi.txt",
                   "transcripts.Google.txt"]
    wer_kaldi = []
    wer_google = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            paths = []
            for transcript in transcripts:
                paths.append(os.path.join(subdir, speaker, transcript))

            lines = []
            for path in paths:
                lines.append(open(path, "r").readlines())
            lines = zip(*lines)

            for i, (stand, kaldi, google) in enumerate(lines):
                ref = preprocess(stand).split()[1:]
                h_k = preprocess(kaldi).split()[1:]
                h_g = preprocess(google).split()[1:]
                wer_k, sub_k, insert_k, delete_k = Levenshtein(ref, h_k)
                wer_g, sub_g, insert_g, delete_g = Levenshtein(ref, h_g)
                wer_kaldi.append(wer_k)
                wer_google.append(wer_g)
                print("{} Kaldi {} {} S:{}, I:{}, D:{}".format(speaker, i, wer_k, sub_k, insert_k, delete_k))
                print("{} Google {} {} S:{}, I:{}, D:{}".format(speaker, i, wer_g, sub_g, insert_g, delete_g))

    print("Kaldi")
    print("average: {},  standard deviation:{}".format(np.average(wer_kaldi), np.std(wer_kaldi)))
    print("Google")
    print("average: {},  standard deviation:{}".format(np.average(wer_google), np.std(wer_google)))
