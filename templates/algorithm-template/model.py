"""Model definition.

Code -> paper convention: annotate the correspondence in docstrings, e.g.
"Implements Eq. (3) of <paper title>" or "Section 3.2: ...", so tracing
code back to the paper needs no extra tooling.
"""


class Model:
    """TODO: implement the algorithm.

    Paper mapping: <e.g. Algorithm 1 / Eq. (1)-(4) / Section 3.2>.
    """

    def fit(self, x, y=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
