# we need check the grad_hash_grid;
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
import numpy as np
from hashgrid import _hash_encode
import random
import os
# import torch.random as random
device=torch.device(0)
input_dim=3
num_levels=4
level_dim=4
per_level_scale=1.5
base_resolution=4
log2_hashmap_size=5
# inputs , embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False

output_dim = num_levels * level_dim

if level_dim % 2 != 0:
    print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

# allocate parameters
offsets = []
offset = 0
max_params = 2 ** log2_hashmap_size
for i in range(num_levels):
    resolution = int(np.ceil(base_resolution * per_level_scale ** i))
    params_in_level = min(max_params, (resolution + 1) ** input_dim) # limit max number
    params_in_level = int(params_in_level / 8) * 8 # make divisible
    offsets.append(offset)
    offset += params_in_level
offsets.append(offset)

def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

random_idx = 41
seed_torch(random_idx)
# print(offset)
# parameters
inputs = torch.rand(1, input_dim, dtype= torch.double, requires_grad=True).to(device)*0.5
# inputs = torch.rand(2, 1, dtype= torch.double, requires_grad=True).to(device).expand(-1, input_dim)*0.1
offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)).to(inputs.device)
seed_torch(random_idx)
# embeddings = torch.ones(offset, level_dim, dtype=torch.double, requires_grad=True).to(device)*0.9
embeddings = torch.rand(offset, level_dim, dtype=torch.double, requires_grad=True).to(device)*0.5
print(inputs)
# print(embeddings)
# inputs = torch.ones(1, input_dim, dtype= torch.double, requires_grad=True).to(device)*0.5
# embeddings = torch.ones(offset, level_dim, dtype=torch.double, requires_grad=True).to(device)

# outputs = (inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
Inputs = (inputs,embeddings, offsets, per_level_scale, base_resolution, True)
# check_results1 = torch.autograd.gradcheck(_hash_encode.apply,(inputs,embeddings), eps=1e-2, atol=1e-3)
check_results1 = torch.autograd.gradcheck(_hash_encode.apply,Inputs, eps=1e-4, atol=1e-3)
print("check_results1", check_results1)
