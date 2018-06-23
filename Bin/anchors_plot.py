import numpy as np
from matplotlib import pyplot as plt

hot_hitok_val = {1000: 0.194, 3000: 0.400, 6000: 0.598, 10000: 0.765, 15000: 0.801, 20000: 0.867,
                 30000: 0.946, 40000: 0.983, 50000: 0.992, 80000: 0.945, 100000: 0.962}

uniform_hitok_val = {1000: 0.077, 3000: 0.214, 6000: 0.391, 10000: 0.575, 15000: 0.750, 20000: 0.829,
                     30000: 0.923, 40000: 0.968, 50000: 0.983, 80000: 0.986, 100000: 0.9998}

edges_hitok_val = {1000: 0.193, 3000: 0.402, 6000: 0.569, 10000: 0.680, 15000: 0.804, 20000: 0.783,
                   30000: 0.877, 40000: 0.895, 50000: 0.933, 80000: 0.954, 100000: 0.956}

x = np.array(list(hot_hitok_val.keys()))
y1 = np.array(list(hot_hitok_val.values()))
y2 = np.array(list(uniform_hitok_val.values()))
y3 = np.array(list(edges_hitok_val.values()))
y = np.stack((y1, y2, y3), axis=0)

fig = plt.figure()
plt.plot(x, y1, label='hotpoint', color='red')
plt.plot(x, y2, label='uniform', color='green')
plt.plot(x, y3, label='edges', color='blue')
plt.xlabel('Sample Points Nums')
plt.ylabel('Hit Ok Rate')
plt.legend()
plt.grid()


ax = plt.gca()

plt.show()


