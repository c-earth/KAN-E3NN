import matplotlib.pyplot as plt
import numpy as np

vvn_100_data = np.loadtxt('./models/240814-101808.txt')

epochs = vvn_100_data[:, :1].astype(np.int32)+1
vvn_100_data = vvn_100_data[:, 1:]
vvn_020_data = np.loadtxt('./models/240814-153019.txt')[:, 1:]
kan_vvn_020_data = np.loadtxt('./models/240814-152709.txt')[:, 1:]

plt.figure()
plt.plot(epochs, vvn_100_data[:, 0], 'b-', label='train vvn 100')
plt.plot(epochs, vvn_100_data[:, 1], 'b--', label='valid vvn 100')
plt.plot(epochs, vvn_020_data[:, 0], 'r-', label='train vvn 020')
plt.plot(epochs, vvn_020_data[:, 1], 'r--', label='valid vvn 020')
plt.plot(epochs, kan_vvn_020_data[:, 0], 'g-', label='train kan vvn 020')
plt.plot(epochs, kan_vvn_020_data[:, 1], 'g--', label='valid kan vvn 020')
plt.legend()
plt.savefig('./KAN_compare.png')