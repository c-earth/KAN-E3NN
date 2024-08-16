import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_load import load_band_structure_data
from utils.utils_data import generate_gamma_data_dict
from utils.utils_model import GraphNetwork_VVN
from torch_geometric.loader import DataLoader
import scipy

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

run_name = '240814-152709'
model_dir = './models'
data_dir = './data'
raw_dir = './data/phonon'
data_file = 'DFPT_band_structure.pkl'

lmax = 2
mul = 16
nlayers = 2
r_max = 4
number_of_basis = 10
radial_layers = 1
radial_neurons = 20
node_dim = 118
node_embed_dim = 16
input_dim = 118
input_embed_dim = 16
vn_an = 26
irreps_out = '1x0e'
option='vvn'

data = load_band_structure_data(data_dir, raw_dir, data_file)
with open(f'./data/idx_{run_name}_te.txt') as f:
    test_rows = np.loadtxt(f).astype(np.int32)

data_dict = generate_gamma_data_dict(data_dir, run_name, data.iloc[test_rows], r_max, vn_an)
te_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
te_loader = DataLoader(te_set, batch_size = 1)

model = GraphNetwork_VVN(mul,
                        irreps_out,
                        lmax,
                        nlayers,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,
                        node_dim,
                        node_embed_dim,
                        input_dim,
                        input_embed_dim)

model.load_state_dict(torch.load(f'./models/{run_name}.torch')['state'])
model.to(device)

reals = []
preds = []
for i, d in enumerate(te_loader):
    d.to(device)
    real = d.y
    pred = model(d)
    reals.append(real)
    preds.append(pred)

reals = torch.concat(reals, dim = 1).flatten().cpu().detach().numpy()
preds = torch.concat(preds, dim = 1).flatten().cpu().detach().numpy()
boundary = (-1, 4)

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

plt.figure()
plt.plot(reals, preds, 'k.', label = f'rsquare:{rsquared(reals, preds)}')
plt.plot(boundary, boundary, 'r-')
plt.xlim(boundary)
plt.ylim(boundary)
plt.gca().set_aspect('equal')
plt.legend()
plt.savefig(f'{run_name}_correlation.png')