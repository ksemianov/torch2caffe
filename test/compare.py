import sys
import numpy as np

first = np.load(sys.argv[1])
second = np.load(sys.argv[2])

if np.allclose(first, second, atol=1e-5):
    print('Ok')
    sys.exit(0)
else:
    print('Failed')
    sys.exit(1)
