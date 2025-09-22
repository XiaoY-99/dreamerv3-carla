import numpy as np

class RunningNorm:
    def __init__(self, eps=1e-5, max_count=1e6):
        self.mean = None
        self.var = None
        self.count = eps
        self.max_count = max_count

    def update(self, x):
        # Ensure 2D (batch, dim)
        x = np.atleast_2d(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Lazy init
        if self.mean is None:
            self.mean = np.zeros_like(batch_mean)
            self.var = np.ones_like(batch_var)

        old_count = self.count
        tot_count = old_count + batch_count

        # Update mean/var using parallel algorithm
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * old_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * old_count * batch_count / tot_count
        new_var = M2 / tot_count
        new_var = np.maximum(new_var, 1e-6)  # avoid div by 0

        # Commit updates
        self.mean = new_mean
        self.var = new_var
        self.count = min(tot_count, self.max_count)

    def normalize(self, x):
        if self.mean is None or self.count < 20:  # wait until stats stabilize
            return x
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
