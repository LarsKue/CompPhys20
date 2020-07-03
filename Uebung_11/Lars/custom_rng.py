
from scipy.stats import rv_continuous


class CustomRNG:
    def __init__(self, pdf, a, b):
        class __CustomPDF(rv_continuous):
            def _pdf(self, x, **kwargs):
                # must be normalized over its range
                return pdf(x)

        self.generator = __CustomPDF(a=a, b=b)

    def sample(self, *args, **kwargs):
        return self.generator.rvs(*args, **kwargs)
