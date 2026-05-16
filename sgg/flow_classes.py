import numpy as np

# 6 classes for radius 
flow_edges = np.array([0.53, 1.42, 3.93, 10.68, 33.97], dtype=np.float32)

    # Class 0: r < 0.53  
    # Class 1: 0.53 <= r < 1.42
    # Class 2: 1.42 <= r < 3.93
    # Class 3: 3.93 <= r < 10.68
    # Class 4: 10.68 <= r < 33.97
    # Class 5: r >= 33.97

