# we need check the grad_hash_grid;
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
import numpy as np
from hashgrid import _hash_encode, _hash_encode_dbackwardFunction
import random
import os
# import torch.random as random
device=torch.device(0)
input_dim=3
num_levels=8
level_dim=1

# control the uncontiguous;
per_level_scale=1.2
base_resolution=8
log2_hashmap_size=11
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
    print(max_params, params_in_level, resolution)

    params_in_level = int(params_in_level / 8) * 8 # make divisible
    offsets.append(offset)
    offset += params_in_level
offsets.append(offset)
print(offsets)
def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

random_idx = 41
seed_torch(random_idx)
# print(offset)
# parameters
inputs = torch.rand(2, input_dim, dtype= torch.double, requires_grad=True).to(device)*0.5
# inputs = torch.rand(2, 1, dtype= torch.double, requires_grad=True).to(device).expand(-1, input_dim)*0.1
offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)).to(inputs.device)
seed_torch(random_idx)
# embeddings = torch.ones(offset, level_dim, dtype=torch.double, requires_grad=True).to(device)*0.9
embeddings = torch.rand(offset, level_dim, dtype=torch.double, requires_grad=True).to(device)*0.5
print(inputs)
print(embeddings.shape)
# inputs = torch.ones(1, input_dim, dtype= torch.double, requires_grad=True).to(device)*0.5
# embeddings = torch.ones(offset, level_dim, dtype=torch.double, requires_grad=True).to(device)
seed_torch(random_idx)
# outputs = (inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
Inputs = (inputs,embeddings, offsets, per_level_scale, base_resolution, True)

grad_output = torch.randn(2, num_levels, level_dim, dtype= torch.double, requires_grad=True).to(device)*0.1
# grad_output = torch.ones(2, num_levels, level_dim, dtype= torch.double, requires_grad=False).to(device)*0.1

# check_results1 = torch.autograd.gradcheck(_hash_encode.apply,Inputs, eps=1e-4, atol=1e-3)

# print(check_results1)
outputs, dy_dx = _hash_encode.apply(inputs,embeddings, offsets, per_level_scale, base_resolution, True)

dy_dx = dy_dx.detach().double()
# print(dy_dx)

Inputs = (inputs,embeddings, offsets, per_level_scale, base_resolution, True)
# check_results1 = torch.autograd.gradcheck(_hash_encode.apply,(inputs,embeddings), eps=1e-2, atol=1e-3)
# check_results1 = torch.autograd.gradcheck(_hash_encode.apply,Inputs, eps=1e-4, atol=1e-3)

# print("check_results1", check_results1)
# print(grad_output.shape)
Inputs2 = ( inputs, embeddings, grad_output, offsets, per_level_scale, base_resolution, dy_dx, True)


dbackward2 = _hash_encode_dbackwardFunction.apply

# grad_inputs, grad_grid = dbackward2(inputs, embeddings, grad_output, offsets, per_level_scale, base_resolution, dy_dx)
# print(grad_inputs)

# inputs1 = torch.rand(2, input_dim, dtype= torch.double, requires_grad=True).to(device)*0.1 + torch.ones_like(inputs)*1

# grad_inputs1, grad_grid = dbackward2(inputs1, embeddings, grad_output, offsets, per_level_scale, base_resolution, dy_dx)
# print(inputs1)
# print(grad_inputs1)
check_results2 = torch.autograd.gradcheck(_hash_encode_dbackwardFunction.apply, Inputs2, eps=1e-4, atol=1e-4)
print("check_results2", check_results2)