import numpy as np

r = np.random.uniform(0.001, 10, size=(10000000))
u = np.random.uniform(0.0001, 1000, size=(10000000))
y = r/u

z = np.zeros((10000000,3)).transpose()
z[0] = r
z[1] = u
z[2] = y

np.savetxt('data_1e~3_10_1e~4_1000_expanded_10000000.csv', z.transpose(), delimiter=',')