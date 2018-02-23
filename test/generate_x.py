import sys
import numpy as np

assert len(sys.argv) == 6
x = np.random.randn(*map(int, sys.argv[1:-1]))
np.save(sys.argv[-1], x)
