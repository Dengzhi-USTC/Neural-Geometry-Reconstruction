// we use the hashgrid of the original code to represent the cu;
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")
#define TCNN_HOST_DEVICE __host__ __device__

template <typename scalar_t, uint32_t N_ELEMS>
struct vector_t {
	TCNN_HOST_DEVICE scalar_t& operator[](uint32_t idx) {
		return data[idx];
	}

	TCNN_HOST_DEVICE scalar_t operator [](uint32_t idx) const {
		return data[idx];
	}

	scalar_t data[N_ELEMS];
	static constexpr uint32_t N = N_ELEMS;
};

template <typename scalar_t, uint32_t N_FLOATS>
using vector_fullp_t = vector_t<scalar_t, N_FLOATS>;

template <uint32_t N_DIMS>
__device__ uint32_t fast_hash(const uint32_t pos_grid[N_DIMS]) {
	static_assert(N_DIMS <= 7, "fast_hash can only hash up to 7 dimensions.");

	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
	constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
	// constexpr uint32_t primes[7] = { 7, 7, 7, 3674653429, 2097192037, 1434869437, 2165219737 };
	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i = 0; i < N_DIMS; ++i) {
		result ^= pos_grid[i] * primes[i];
	}
	return result;
}

static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
	return atomicAdd(reinterpret_cast<__half*>(address), val);
  }

  // pos: absolute coordinates, 3
// pos_grid: grid_coordinates, 3
// w_neighs: 8 * 3
// pos_neighs: 8 *3

static inline __host__ __device__ void generate_dy_dx(float *pos, uint32_t * pos_grid, float * w_neighs, uint32_t *pos_neighs)
{
	float x = pos[0];
	float y = pos[1];
	float z = pos[2];
	    // for (1-x)* (1-y)* (1-z)
	w_neighs[0] = -1*(1-y)*(1-z);
	w_neighs[1] = -1*(1-x)*(1-z);
	w_neighs[2] = -1*(1-x)*(1-y);
	pos_neighs[0] = pos_grid[0];
	pos_neighs[1] = pos_grid[1];
	pos_neighs[2] = pos_grid[2];
		// for (x)* (1-y)* (1-z)
	w_neighs[3] = (1-y)*(1-z);
	w_neighs[4] = -1*(x)*(1-z);
	w_neighs[5] = -1*(x)*(1-y);
	pos_neighs[3] = pos_grid[0]+1;
	pos_neighs[4] = pos_grid[1];
	pos_neighs[5] = pos_grid[2];

	    // for (1-x)* y* (1-z)
	w_neighs[6] = -1*(y)*(1-z);
	w_neighs[7] = (1-x)*(1-z);
	w_neighs[8] = -1*(1-x)*(y);
	pos_neighs[6] = pos_grid[0];
	pos_neighs[7] = pos_grid[1]+1;
	pos_neighs[8] = pos_grid[2];

	    // for (x)* y* (1-z)
	w_neighs[9] = (y)*(1-z);
	w_neighs[10] = (x)*(1-z);
	w_neighs[11] = -1*(x)*(y);
	pos_neighs[9] = pos_grid[0]+1;
	pos_neighs[10] = pos_grid[1]+1;
	pos_neighs[11] = pos_grid[2];

	    // for (1-x)* (1-y)* z
	w_neighs[12] = -1*(1-y)*z;
	w_neighs[13] = -1*(1-x)*(z);
	w_neighs[14] = (1-x)*(1-y);
	pos_neighs[12] = pos_grid[0];
	pos_neighs[13] = pos_grid[1];
	pos_neighs[14] = pos_grid[2]+1;

	    // for (x)* (1-y)* z
	w_neighs[15] = (1-y)*(z);
	w_neighs[16] = -1*(x)*(z);
	w_neighs[17] = (x)*(1-y);
	pos_neighs[15] = pos_grid[0]+1;
	pos_neighs[16] = pos_grid[1];
	pos_neighs[17] = pos_grid[2]+1;

	    // for (1-x)* (y)* z
	w_neighs[18] = -1*(y)*(z);
	w_neighs[19] = (1-x)*(z);
	w_neighs[20] = (1-x)*(y);
	pos_neighs[18] = pos_grid[0];
	pos_neighs[19] = pos_grid[1]+1;
	pos_neighs[20] = pos_grid[2]+1;

	    // for (x)* (y)* z
	w_neighs[21] = (y)*(z);
	w_neighs[22] = (x)*(z);
	w_neighs[23] = (x)*(y);
	pos_neighs[21] = pos_grid[0]+1;
	pos_neighs[22] = pos_grid[1]+1;
	pos_neighs[23] = pos_grid[2]+1;
}

static inline __host__ __device__ void generate_dy_dx(float *pos, uint32_t * pos_grid, float * w_neighs)
{
	float x = pos[0];
	float y = pos[1];
	float z = pos[2];
	    // for (1-x)* (1-y)* (1-z)
	w_neighs[0] = -1*(1-y)*(1-z);
	w_neighs[1] = -1*(1-x)*(1-z);
	w_neighs[2] = -1*(1-x)*(1-y);
		// for (x)* (1-y)* (1-z)
	w_neighs[3] = (1-y)*(1-z);
	w_neighs[4] = -1*(x)*(1-z);
	w_neighs[5] = -1*(x)*(1-y);

	    // for (1-x)* y* (1-z)
	w_neighs[6] = -1*(y)*(1-z);
	w_neighs[7] = (1-x)*(1-z);
	w_neighs[8] = -1*(1-x)*(y);

	    // for (x)* y* (1-z)
	w_neighs[9] = (y)*(1-z);
	w_neighs[10] = (x)*(1-z);
	w_neighs[11] = -1*(x)*(y);

	    // for (1-x)* (1-y)* z
	w_neighs[12] = -1*(1-y)*z;
	w_neighs[13] = -1*(1-x)*(z);
	w_neighs[14] = (1-x)*(1-y);

	    // for (x)* (1-y)* z
	w_neighs[15] = (1-y)*(z);
	w_neighs[16] = -1*(x)*(z);
	w_neighs[17] = (x)*(1-y);

	    // for (1-x)* (y)* z
	w_neighs[18] = -1*(y)*(z);
	w_neighs[19] = (1-x)*(z);
	w_neighs[20] = (1-x)*(y);

	    // for (x)* (y)* z
	w_neighs[21] = (y)*(z);
	w_neighs[22] = (x)*(z);
	w_neighs[23] = (x)*(y);
}

static inline __host__ __device__ void generate_weights_pos(float *pos, uint32_t * pos_grid, float * w_neighs, uint32_t *pos_neighs)
{
	float x = pos[0];
	float y = pos[1];
	float z = pos[2];
	    // for (1-x)* (1-y)* (1-z)
	w_neighs[0] = (1-x)*(1-y)*(1-z);
	pos_neighs[0] = pos_grid[0];
	pos_neighs[1] = pos_grid[1];
	pos_neighs[2] = pos_grid[2];
		// for (x)* (1-y)* (1-z)
	w_neighs[1] = x*(1-y)*(1-z);
	pos_neighs[3] = pos_grid[0]+1;
	pos_neighs[4] = pos_grid[1];
	pos_neighs[5] = pos_grid[2];

	    // for (1-x)* y* (1-z)
	w_neighs[2] = (1-x)*(y)*(1-z);
	pos_neighs[6] = pos_grid[0];
	pos_neighs[7] = pos_grid[1]+1;
	pos_neighs[8] = pos_grid[2];

	    // for (x)* y* (1-z)
	w_neighs[3] = x*(y)*(1-z);
	pos_neighs[9] = pos_grid[0]+1;
	pos_neighs[10] = pos_grid[1]+1;
	pos_neighs[11] = pos_grid[2];

	    // for (1-x)* (1-y)* z
	w_neighs[4] = (1-x)*(1-y)*z;
	pos_neighs[12] = pos_grid[0];
	pos_neighs[13] = pos_grid[1];
	pos_neighs[14] = pos_grid[2]+1;

	    // for (x)* (1-y)* z
	w_neighs[5] = x*(1-y)*(z);
	pos_neighs[15] = pos_grid[0]+1;
	pos_neighs[16] = pos_grid[1];
	pos_neighs[17] = pos_grid[2]+1;

	    // for (1-x)* (y)* z
	w_neighs[6] = (1-x)*(y)*(z);
	pos_neighs[18] = pos_grid[0];
	pos_neighs[19] = pos_grid[1]+1;
	pos_neighs[20] = pos_grid[2]+1;

	    // for (x)* (y)* z
	w_neighs[7] = (y)*(z)*x;
	pos_neighs[21] = pos_grid[0]+1;
	pos_neighs[22] = pos_grid[1]+1;
	pos_neighs[23] = pos_grid[2]+1;
}

// recover the neighbour grid information from the pos and pos_grid.
// pos : D
// pos_grid : D
// w_neights : 2<<D * D * D
// pos_grid_neighs : 2<<D * D
template <uint32_t D>
static inline __host__ __device__ void generate_weights_locations_second(float *pos, uint32_t *pos_grid, float * w_neighs, uint32_t *pos_grid_neighs)
{
    float x = pos[0];
    float y = pos[1];
    float z = pos[2];
    // for (1-x)* (1-y)* (1-z)
    // will be abundant, but it is easy to visit;
    w_neighs[0] = 0;
    w_neighs[1] = (1 - z);
    w_neighs[2] = (1 - y);
    w_neighs[3] = (1 - z);
    w_neighs[4] = 0;
    w_neighs[5] = (1 - x);
    w_neighs[6] = (1 - y);
    w_neighs[7] = (1 - x);
    w_neighs[8] = 0;

    pos_grid_neighs[0] = pos_grid[0];
    pos_grid_neighs[1] = pos_grid[1];
    pos_grid_neighs[2] = pos_grid[2];
    
    // for x* (1-y)* (1-z)

    w_neighs[9] = 0;
    w_neighs[10] = -1*(1 - z);
    w_neighs[11] = -1*(1 - y);
    w_neighs[12] = -1*(1 - z);
    w_neighs[13] = 0;
    w_neighs[14] = x;
    w_neighs[15] = -1*(1 - y);
    w_neighs[16] = x;
    w_neighs[17] = 0;

    pos_grid_neighs[3] = pos_grid[0] +1;
    pos_grid_neighs[4] = pos_grid[1];
    pos_grid_neighs[5] = pos_grid[2];

    // for (1-x)* y* (1-z)

    w_neighs[18] = 0;
    w_neighs[19] = -1*(1 - z);
    w_neighs[20] = y;
    w_neighs[21] = -1*(1 - z);
    w_neighs[22] = 0;
    w_neighs[23] = -1*(1 - x);
    w_neighs[24] = y;
    w_neighs[25] = -1*(1 - x);
    w_neighs[26] = 0;

    pos_grid_neighs[6] = pos_grid[0];
    pos_grid_neighs[7] = pos_grid[1]+1;
    pos_grid_neighs[8] = pos_grid[2];

    // for (x)* y* (1-z)

    w_neighs[27] = 0;
    w_neighs[28] = (1 - z);
    w_neighs[29] = -1*y;
    w_neighs[30] = (1 - z);
    w_neighs[31] = 0;
    w_neighs[32] = -1*x;
    w_neighs[33] = -1*y;
    w_neighs[34] = -1*x;
    w_neighs[35] = 0;

    pos_grid_neighs[9] = pos_grid[0]+1;
    pos_grid_neighs[10] = pos_grid[1]+1;
    pos_grid_neighs[11] = pos_grid[2];

    // for (1-x)* (1-y)* z

    w_neighs[36] = 0;
    w_neighs[37] = z;
    w_neighs[38] = -1*(1 - y);
    w_neighs[39] = z;
    w_neighs[40] = 0;
    w_neighs[41] = -1*(1 - x);
    w_neighs[42] = -1*(1 - y);
    w_neighs[43] = -1*(1 - x);
    w_neighs[44] = 0;

    pos_grid_neighs[12] = pos_grid[0];
    pos_grid_neighs[13] = pos_grid[1];
    pos_grid_neighs[14] = pos_grid[2]+1;

    // for (x)* (1-y)* z

    w_neighs[45] = 0;
    w_neighs[46] = -1*z;
    w_neighs[47] = (1 - y);
    w_neighs[48] = -1*z;
    w_neighs[49] = 0;
    w_neighs[50] = -1*x;
    w_neighs[51] = (1 - y);
    w_neighs[52] = -1*x;
    w_neighs[53] = 0;

    pos_grid_neighs[15] = pos_grid[0]+1;
    pos_grid_neighs[16] = pos_grid[1];
    pos_grid_neighs[17] = pos_grid[2]+1;

    // for (1-x)* (y)* z

    w_neighs[54] = 0;
    w_neighs[55] = -1*z;
    w_neighs[56] = -1*y;
    w_neighs[57] = -1*z;
    w_neighs[58] = 0;
    w_neighs[59] = (1- x);
    w_neighs[60] = -1*y;
    w_neighs[61] = (1 - x);
    w_neighs[62] = 0;

    pos_grid_neighs[18] = pos_grid[0];
    pos_grid_neighs[19] = pos_grid[1]+1;
    pos_grid_neighs[20] = pos_grid[2]+1;

    // for (x)* (y)* z

    w_neighs[63] = 0;
    w_neighs[64] = z;
    w_neighs[65] = y;
    w_neighs[66] = z;
    w_neighs[67] = 0;
    w_neighs[68] = x;
    w_neighs[69] = y;
    w_neighs[70] = x;
    w_neighs[71] = 0;
    pos_grid_neighs[21] = pos_grid[0]+1;
    pos_grid_neighs[22] = pos_grid[1]+1;
    pos_grid_neighs[23] = pos_grid[2]+1;
}

template <typename F, typename FPRIME, typename FPRIME2>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, float* pos_derivative2,uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative, FPRIME2 interpolation_fun_derivative2) {
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
    *pos_derivative2 = interpolation_fun_derivative2(*pos);
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F, typename FPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative) {
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F>
__device__ inline void pos_fract(const float input, float* pos, uint32_t* pos_grid, float scale, F interpolation_fun) {
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos = interpolation_fun(*pos);
}

__device__ inline float identity_fun(float val) {
	return val;
}

__device__ inline float identity_derivative(float val) {
	return 1;
}

// may be we need construct the 1st-smooth;
__device__ inline float smoothstep(float val) {
	return val*val*(3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6*val*(1.0f - val);
}

__device__ inline float smoothstep_derivative_second(float val) {
	return -12*val + 6;
}
// may be we need construct the 2nd-smooth;
__device__ inline float smoothstep_2(float val) {
	// return val*val*(3.0f - 2.0f * val);
	return val*val*val*(6.0f*val*val - 15.0f * val + 10);
}

__device__ inline float smoothstep_derivative_2(float val) {
	return 30*val*val*(1.0f - val)*(1.0f -val);
}

__device__ inline float smoothstep_derivative_second_2(float val) {
	// return -12*val + 6;
    return 60*val*(2*val*val -3*val + 1);
}

template <typename scalar_t>
static inline __host__ __device__ scalar_t div_round_up(scalar_t val, scalar_t divisor) {
	return (val + divisor - 1) / divisor;
}

template <uint32_t N_DIMS, uint32_t C>
__device__ uint32_t grid_index(const uint32_t feature, const uint32_t hashmap_size, const uint32_t grid_resolution, const uint32_t pos_grid[N_DIMS]) {
	uint32_t stride = 1;
	uint32_t index = 0;

	// The second part of the loop condition is needed to avoid integer overflows in finer levels.
	#pragma unroll
	for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
		index += pos_grid[dim] * stride;
		stride *= grid_resolution;
	}

	if (hashmap_size < stride) {
		index = fast_hash<N_DIMS>(pos_grid);
	}

	return (index % hashmap_size) * C + feature;
}

// for the grid
// we need pay attention to the size?
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(
	const uint32_t B, // batch_size
	const int* offsets, // search the grid from the 3d space
	const uint32_t H, // 
	const float S,
	// const InterpolationType interpolation_type,
	const scalar_t* __restrict__ grid, // L, C
	const scalar_t* __restrict__ positions_in, //input D, B;
	scalar_t* __restrict__ encoded_positions  // output, L, C, B
	// T* __restrict__ dy_dx  // L, C, B, D
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= B) return;
	const uint32_t level = blockIdx.y; // <- the level is the same for all threads
	// printf("%d", level);
	grid += offsets[level] * C;
	const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
	const float scale = exp2f(level * S) * H - 1.0f;
	const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);
	// printf("%f\n", scale);
	float pos[D];
	// float pos_derivative[D];
	uint32_t pos_grid[D];
    #pragma unroll
    for (uint32_t dim = 0; dim < D; ++dim) {
        pos_fract(positions_in[i + dim * B], &pos[dim], &pos_grid[dim], scale, smoothstep_2);
    }	
	auto grid_val = [&](const uint32_t local_pos[D]) {
		uint32_t index = grid_index<D, C>(0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<scalar_t, C>*)&grid[index];
	};
	uint32_t pos_neighs[24] = {0};
	float weights_neighs[8] = {0};
	// N-linear interpolation
	vector_t<scalar_t, C> result = {};
	vector_fullp_t<scalar_t, D> grads[C] = {0};
	generate_weights_pos(pos, pos_grid, weights_neighs, pos_neighs);
	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << D); ++idx) {
		float weight = 1;
		uint32_t pos_grid_local[D];
		pos_grid_local[0] = pos_neighs[idx*3];
		pos_grid_local[1] = pos_neighs[idx*3 +1];
		pos_grid_local[2] = pos_neighs[idx*3 +2];
		weight *= weights_neighs[idx];
		auto val = grid_val(pos_grid_local);
		#pragma unroll
		for (uint32_t feature = 0; feature < C; ++feature) {
			float data = (float)((scalar_t*)&val)[feature];
			((scalar_t*)&result)[feature] += (scalar_t)(weight * data);
			}
	}
    #pragma unroll
    for (uint32_t f = 0; f < C; ++f) {
        encoded_positions[i + (level * C + f) * B] = result[f];
    }
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_grad_inputs_pre(
	const uint32_t B, // batch_size
	const int* offsets, // search the grid from the 3d space
	const uint32_t H, // 
	const float S,
	// const InterpolationType interpolation_type,
	const scalar_t* __restrict__ grid, // L, C
	// const float* __restrict__ positions_in, //input D, B;
	const scalar_t* __restrict__ positions_in, //input D, B;
	scalar_t* __restrict__ dy_dx  // L, C, B, D
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= B) return;

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads
	// printf("%d", level);
	grid += offsets[level] * C;
	const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
	const float scale = exp2f(level * S) * H - 1.0f;
	const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);
	// printf("%f\n", scale);
	float pos[D];
	float pos_derivative[D];
	uint32_t pos_grid[D];
    #pragma unroll
    for (uint32_t dim = 0; dim < D; ++dim) {
        pos_fract(positions_in[i + dim * B], &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, smoothstep_2, smoothstep_derivative_2);
    }	
	auto grid_val = [&](const uint32_t local_pos[D]) {
		uint32_t index = grid_index<D, C>(0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<scalar_t, C>*)&grid[index];
	};
    uint32_t pos_neighs_dy_dx[24] = {0};
	float weights_neighs_dy_dx[24] = {0};
	// N-linear interpolation
	vector_t<scalar_t, C> result = {};
	vector_fullp_t<scalar_t, D> grads[C] = {0};
    generate_dy_dx(pos, pos_grid, weights_neighs_dy_dx, pos_neighs_dy_dx);
	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << D); ++idx) {
		float weight = 1;
		uint32_t pos_grid_local[D];
		pos_grid_local[0] = pos_neighs_dy_dx[idx*3];
		pos_grid_local[1] = pos_neighs_dy_dx[idx*3 +1];
		pos_grid_local[2] = pos_neighs_dy_dx[idx*3 +2];
		auto val = grid_val(pos_grid_local);
		#pragma unroll
		for (uint32_t feature = 0; feature < C; ++feature) {
			float data = (float)((scalar_t*)&val)[feature];
			((scalar_t*)&result)[feature] += (scalar_t)(weight * data);
				#pragma unroll
				for (uint32_t grad_dim = 0; grad_dim < D; ++grad_dim)
				{
						grads[feature][grad_dim] += scalar_t(scale*float(weights_neighs_dy_dx[idx*3+grad_dim]) * ((float)val[feature]))*pos_derivative[grad_dim];
			}
			}
	}
		#pragma unroll
		for (uint32_t f = 0; f < C; ++f) {
			((vector_fullp_t<scalar_t, D>*)dy_dx)[i + (level * C + f) * B] = grads[f];
		}
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_grid_backward(
	const uint32_t B,
	const int* offsets,
	const uint32_t H,
	const float S,
	scalar_t* __restrict__ grid_gradient, // L, C
	const scalar_t* __restrict__ positions_in,
	const scalar_t* __restrict__ dL_dy  // the gradients of the node, L, C, B
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / C;
	if (i >= B) return;

	const uint32_t level = blockIdx.y ; // <- the level is the same for all threads.
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * C;
	grid_gradient += offsets[level] * C;
	const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

	const float scale = exp2f(level * S) * H - 1.0f;
	const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);

	auto add_grid_gradient = [&](const uint32_t local_pos[D], const vector_t<scalar_t, N_FEATURES_PER_THREAD>& grad, const float weight) {
		uint32_t index = grid_index<D, C>(feature, hashmap_size, grid_resolution, local_pos);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEATURES_PER_THREAD > 1 && std::is_same<scalar_t, __half>::value) {
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; f += 2) {
				__half2 v = {(__half)((float)grad[f] * weight), (__half)((float)grad[f+1] * weight)};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (std::is_same<scalar_t, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
					// printf("%f\n",((float)grad[f] * weight));
					atomicAdd(&grid_gradient[index + f], (weight * float(grad[f])));
				}
			}
		}
	};

	float pos[D];
	uint32_t pos_grid[D];
		#pragma unroll
		for (uint32_t dim = 0; dim < D; ++dim) {
			pos_fract(positions_in[i + dim * B], &pos[dim], &pos_grid[dim], scale, smoothstep_2);
		}
    vector_t<scalar_t, N_FEATURES_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = dL_dy[i + (level * C + feature + f) * B];
	}
	uint32_t pos_neighs[24] = {0};
	float weights_neighs[8] = {0};
	// N-linear interpolation
	generate_weights_pos(pos, pos_grid, weights_neighs, pos_neighs);
	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << D); ++idx) {
		float weight = weights_neighs[idx];
		uint32_t pos_grid_local[D];
		pos_grid_local[0] = pos_neighs[3*idx];
		pos_grid_local[1] = pos_neighs[3*idx+1];
		pos_grid_local[2] = pos_neighs[3*idx+2];
		add_grid_gradient(pos_grid_local, grad, weight);
	}
}

// we need compute the gradients of the inputs;
template <typename scalar_t, uint32_t D>
__global__ void kernel_grid_backward_input(
	const uint32_t B, // batch_size;
	const uint32_t num_grid_features,
	const scalar_t* dL_dy_rm, // L, C, B, for the gradients of the Computing Node;
	const scalar_t* __restrict__ dy_dx, // L, C, B, D
	scalar_t* __restrict__ dL_dx // B, D, for the output;
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= B) return;
	vector_fullp_t<scalar_t, D> result = {0};
	for (int k = 0; k < num_grid_features; ++k) {
		float dL_dy_local = (float)dL_dy_rm[i + k * B];
		// printf("%f\n", dL_dy_local);
		auto dy_dx_local = ((vector_fullp_t<scalar_t, D>*)dy_dx)[i + k * B];
		#pragma unroll
		for (uint32_t dim = 0; dim < D; ++dim) {
			result[dim] += dL_dy_local * dy_dx_local[dim];
		}
	}
	// generate the dL_dx;
	for (int dim =0; dim < D; ++dim)
	{
		dL_dx[i*D+dim] = result[dim];
	}
}
// grad of the inputs; we need search the \theta correspondence with grad_grad_out;
// \Theta ---> grad_out;
// CHECK 1
template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward_grad_out_backward(
    const scalar_t * __restrict__ positions_in, // B, D
    const scalar_t * __restrict__ grad_grad_grid, // L, C
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grad_out_grid, // B, L, C, we only considerate a computing graph node; since we concanate the L resolutions;
    const uint32_t B, const uint32_t L, const float S, const uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
	if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;
    // locate
    grad_grad_grid += offsets[level] * C;
    positions_in += b * D;
    grad_grad_out_grid += b * L * C + level * C + ch;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];
    #pragma unroll
    for (uint32_t dim = 0; dim < D; ++dim) {
        pos_fract(positions_in[dim], &pos[dim], &pos_grid[dim], scale, smoothstep_2);
    }	
	uint32_t pos_neighs[24] = {0};
	float weights_neighs[8] = {0};
	generate_weights_pos(pos, pos_grid, weights_neighs, pos_neighs);
    // interpolate, we generate the 2<<D^3 locations;
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = weights_neighs[idx];
        uint32_t pos_grid_local[D];
		pos_grid_local[0] = pos_neighs[3*idx];
		pos_grid_local[1] = pos_neighs[3*idx+1];
		pos_grid_local[2] = pos_neighs[3*idx+2];
        uint32_t index = grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);
        // we need collect the grad_grad_out_grid, 
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(grad_grad_grid[index + c] * w), (__half)(grad_grad_grid[index + c] * w)};
                atomicAdd((__half2*)&grad_grad_out_grid[c], v);
            }
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grad_out_grid[c], w *grad_grad_grid[index + c]);
            }
        }
    }    
}

// X---> \theta, \theta ---> \theta is zero, since \theta is linear representations;
// CHECK 1
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward_grid_backward(
    const scalar_t * __restrict__ grad_out, // B, L, C
    const scalar_t * __restrict__ positions_in, // B, D
    const scalar_t * __restrict__ grad_grad_inputs, // B, D
    const int * __restrict__ offsets,
    scalar_t * __restrict__ grad_input_grad_grid, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x);
	if (b >= B) return;
    const uint32_t level = blockIdx.y;
    // locate
    grad_input_grad_grid += offsets[level] * C;
    positions_in += b * D;
    grad_out += b * L * C + level * C;
    grad_grad_inputs += b*D;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];
    float pos_derivative[D];
    #pragma unroll
    for (uint32_t dim = 0; dim < D; ++dim) {
        pos_fract(positions_in[dim], &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, smoothstep_2, smoothstep_derivative_2);
    }
    // interpolate
    // need considerate d (X) ---> d (\theta);
	float w_neighs[24];
	uint32_t pos_neighs[24];
	generate_dy_dx(pos, pos_grid, w_neighs, pos_neighs);
	uint32_t pos_grid_local[D];
	for (uint32_t idx = 0; idx < (1<<D); ++idx)
	{
		pos_grid_local[0] = pos_neighs[idx*3];
		pos_grid_local[1] = pos_neighs[idx*3 +1];
		pos_grid_local[2] = pos_neighs[idx*3 +2];
		uint32_t index = grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
		float w = 0;
		// the x, y, z; is same constributions to the special index;
		for(uint32_t gd =0; gd < D; ++gd)
		{
			w += scale*grad_grad_inputs[gd]*w_neighs[3*idx+gd]*pos_derivative[gd];
		}
		if (std::is_same<scalar_t, at::Half>::value && C % 2 == 0) {
			#pragma unroll
			for (uint32_t c = 0; c < C; c += 2) {
				// process two __half at once (by interpreting as a __half2)
				__half2 v = {(__half)(float(grad_out[c]) * w), (__half)(float(grad_out[c + 1]) * w)};
				atomicAdd((__half2*)&grad_input_grad_grid[index + c], v);
			}
		// float, or __half when N_C % 2 != 0
		} else {
			#pragma unroll
			for (uint32_t c = 0; c < C; c++) {
				atomicAdd(&grad_input_grad_grid[index + c], w * float(grad_out[c]));
			}
		}
	}
}
// for this part, we ignore the N_C;
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_backward_part1(
    const scalar_t * __restrict__ positions_in, // B, D
    const scalar_t * __restrict__ grid,
    const scalar_t * __restrict__ grad_grad_grid, // L, C
    const scalar_t * __restrict__ grad_out, // B, L, C
    const scalar_t * __restrict__ grad_grad_inputs,
    const int * __restrict__ offsets, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    scalar_t * __restrict__ grad_grid_grad_input,
    scalar_t * __restrict__ grad_inputs_grad_grad_out
    // scalar_t * __restrict__ dy2_dx
)
{
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const uint32_t level = blockIdx.y;
    // locate
    grid += (uint32_t)offsets[level] * C;
    positions_in += b * D;
    grad_grad_inputs += b*D;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];
    float pos_derivative[D];
    float pos_derivative2[D];
    #pragma unroll
    for (uint32_t dim = 0; dim < D; ++dim) {
        pos_fract(positions_in[dim], &pos[dim], &pos_derivative[dim], &pos_derivative2[dim], &pos_grid[dim], scale, smoothstep_2, smoothstep_derivative_2, smoothstep_derivative_second_2);
    }
    // for input_backward_grad_out_backward
    // dy2_dx += b * D*D * L * C + level * D*D * C; // B L D*D C
    grad_inputs_grad_grad_out += b* L * C + level * C;
    grad_grad_grid += (uint32_t)offsets[level] *C;
    grad_out += b*L*C + level*C;
    grad_grid_grad_input += b*D;
	uint32_t pos_neighs_dy_dx[24] = {0};
	float weights_neighs_dy_dx[24] = {0};
	generate_dy_dx(pos, pos_grid, weights_neighs_dy_dx, pos_neighs_dy_dx);
    // float w_neighs[(1<<D) * D *D] = {0};
    // uint32_t pos_grid_neighs[(1<<D) * D] = {0};
    // generate_weights_locations_second<D>(pos, pos_grid, w_neighs, pos_grid_neighs);
    float results_grad[C] = {0}; // temp
	uint32_t pos_grid_local[D];
	#pragma unroll
	for(uint32_t idx = 0; idx < (1<<D); ++idx)
	{
        pos_grid_local[0] = pos_neighs_dy_dx[idx*3];
		pos_grid_local[1] = pos_neighs_dy_dx[idx*3 +1];
		pos_grid_local[2] = pos_neighs_dy_dx[idx*3 +2];
        uint32_t index = grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
        #pragma unroll
        for(uint32_t gd = 0; gd < D; ++gd)
        {
		float w = scale*weights_neighs_dy_dx[idx*D+gd]*pos_derivative[gd];
        // fused the derivatives of (x, y, z) respectively;
		#pragma unroll
		for (uint32_t ch = 0; ch < C; ch++) {
			results_grad[ch] += w * (grid[index + ch])*grad_grad_inputs[gd];
            atomicAdd(&grad_grid_grad_input[gd], (w *float(grad_out[ch]*grad_grad_grid[index + ch])));
		}

        // for input ---> input, connected to the derivatives second; d(gd) ---> d(gd2)
        // #pragma unroll
        // for (uint32_t gd2 = 0; gd2 < D; ++gd2)
        // {
        //     w = scale*scale*(w_neighs[idx*D*D + gd*D + gd2]*pos_derivative[gd]*pos_derivative[gd2]);
        //     if (gd == gd2)
        //     {
        //         w +=scale*scale*weights_neighs_dy_dx[idx*D +gd]*pos_derivative2[gd2];
        //     }
        //     #pragma unroll
        //     for (int ch = 0; ch <C; ch++)
        //     {
        //         // exists ?
        //         dy2_dx[(gd*D +gd2)* C + ch] +=w*(grid[index+ch]);
        //     }
        // }
	}
	}
	#pragma unroll
	for (uint32_t ch = 0; ch < C; ++ch) {
		grad_inputs_grad_grad_out[ch] = results_grad[ch];
	}  
}

// with the input (x) and grid, we deliver the grid;
// CHECK 1;
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward_input_backward_pre(
    const scalar_t * __restrict__ positions_in, // inputs (for position of the input, (x, y, z))
    const scalar_t * __restrict__ grid,
    const int * offsets,
    scalar_t * __restrict__ dy2_dx, // this is just for cal the second derivative; aid to the gradients of d(x) // B L D*D C
    const uint32_t B, const uint32_t L, const float S, const uint32_t H
){
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    // locate
    grid += (uint32_t)offsets[level] * C;
    positions_in += b * D;
    // outputs += level * B * C + b * C;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];
    float pos_derivative[D];
    float pos_derivative2[D];
    #pragma unroll
    for (uint32_t dim = 0; dim < D; ++dim) {
        pos_fract(positions_in[dim], &pos[dim], &pos_derivative[dim], &pos_derivative2[dim], &pos_grid[dim], scale, smoothstep_2, smoothstep_derivative_2, smoothstep_derivative_second_2);
    }
        dy2_dx += b * D*D * L * C + level * D*D * C; // B L D*D C

        // float results_grad[C*D*D] = {0};  // for each dimension, we store 

        // construct D<<2 locations.
        uint32_t pos_grid_local[D];
        float w = scale*scale;
        float w_neighs[(1<<D) * D *D] = {0};
        uint32_t pos_grid_neighs[(1<<D) * D] = {0};
        generate_weights_locations_second<D>(pos, pos_grid, w_neighs, pos_grid_neighs);
        // uint32_t pos_neighs_dy_dx[24] = {0};
        float weights_neighs_dy_dx[24] = {0};
        generate_dy_dx(pos, pos_grid, weights_neighs_dy_dx);
        // multiply correspondence features;
        // calculate whole representations first, then search;
        #pragma unroll
        for (uint32_t idx = 0; idx < 1 << D; ++idx)
        {
            pos_grid_local[0] = pos_grid_neighs[idx*3];
            pos_grid_local[1] = pos_grid_neighs[idx*3 +1];
            pos_grid_local[2] = pos_grid_neighs[idx*3 +2];
            uint32_t  index = grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
            float w;
                    // for input ---> input, connected to the derivatives second; d(gd) ---> d(gd2)

            #pragma unroll
            for (uint32_t gd1 = 0; gd1 < D; gd1++)
            {
                #pragma unroll
                for (uint32_t gd2 = 0; gd2 < D; gd2++)
                {
                    w = scale*scale*(w_neighs[idx*D*D + gd1*D + gd2]*pos_derivative[gd1]*pos_derivative[gd2]);
                    if (gd1 == gd2)
                    {
                        w +=scale*scale*weights_neighs_dy_dx[idx*D +gd1]*pos_derivative2[gd2];
                    }
                    #pragma unroll
                    for (int ch = 0; ch <C; ch++)
                    {
                        dy2_dx[(gd1*D +gd2)* C + ch] +=w*(grid[index+ch]);
                    }
                }
            }
        }
}

// calculate (x, y, z)---> (x, y, z): computing graph; but else \theta ---> (x, y, z);
// CHECK 1
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward_input_backward(
    const scalar_t * __restrict__ grad_out, // B, L, C
    const scalar_t * __restrict__ dy2_dx,
    const scalar_t* __restrict__ grad_inputs_output, // 1st derivative;
    scalar_t * __restrict__ grad_input_grad_input, // B, L, D*D, C
    uint32_t B, uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= B * D) return;
    const uint32_t b = t / D;
    // considerate (x, y, z)
    const uint32_t d = t - b * D;
    grad_out += b * L * C;
    dy2_dx += b * L * D*D * C;
    grad_inputs_output += b*D;
    grad_input_grad_input += b*D;
    # pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            grad_input_grad_input[d] += grad_out[l * C + ch] * (grad_inputs_output[0] *dy2_dx[l * D*D * C + (d) * C + ch] + grad_inputs_output[1]*dy2_dx[l * D*D * C + (D+d)*C +ch] + grad_inputs_output[2]*dy2_dx[l * D*D * C + (2*D+d)*C +ch]); // there exist three computing graph; x-->d, y-->d, z-->d;
        }
    }
}
// calculate the gradients of input, we represent the parameters of the feature is \theta, the parameters of the inputs is \x
template <typename scalar_t, uint32_t D>
void kernel_grid_backward_backward_wrapper(const scalar_t * grad_out, const scalar_t * inputs,const scalar_t * grad_grad_inputs, const scalar_t * grid, const scalar_t * grad_grad_grid, const int * offsets, scalar_t * grad2_grid, scalar_t * grad2_input, scalar_t * dy2_dx, scalar_t * grad_grad_out, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H)
{
    // N_C
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C); // n_features_per_thread
    const dim3 blocks_hashgrid = {div_round_up(B * C / N_C, N_THREAD), L, 1 };
    // no_N_C
    static constexpr uint32_t N_THREAD1 = 512;
	const dim3 blocks_hashgrid1 = {div_round_up(B, N_THREAD1), L, 1 };
    switch (C) {
    // case 1: 
    //     // grid
    //     // N_C
    //     kernel_input_backward_grid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad_out, inputs, grad_grad_inputs, offsets, grad2_grid, B, L, S, H);
    //     // input 
    //     // grid --> input;
    //     // no N_C
    //     // kernel_grid_backward_input_backward<scalar_t, D, 1><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_grid, grad_out, offsets, B, L, S, H, grad2_input);
    
    //     // // input --->input
    //     // // no N_C
    //     // kernel_input_backward_input_backward_pre<scalar_t, D, 1><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, offsets, dy2_dx, B, L, S, H);
    //     // // grad_grad_out
    //     // // no N_C
    //     // kernel_input_backward_grad_out_backward<scalar_t, D, 1><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_inputs, grid, offsets, B, L, S, H, grad_grad_out);

    //     kernel_backward_part1<scalar_t, D, 1><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, grad_grad_grid, grad_out, grad_grad_inputs, offsets, dy2_dx, B, L, S, H, grad2_input, grad_grad_out);
    //     // 
    //     kernel_input_backward_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad_out, dy2_dx, grad_grad_inputs, grad2_input, B, L);

    //     // N_C
    //     kernel_grid_backward_grad_out_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, grad_grad_grid, offsets, grad_grad_out, B, L, S, H);

    //     break;
    case 2: 

        // input 
        // grid --> input;
        // no N_C
        // kernel_grid_backward_input_backward<scalar_t, D, 2><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_grid, grad_out, offsets, B, L, S, H, grad2_input);

        // // input --->input
        // // no N_C

                // grid
        // N_C input ---> grid
        kernel_input_backward_grid_backward<scalar_t, D, 2><<<blocks_hashgrid1, N_THREAD1>>>(grad_out, inputs, grad_grad_inputs, offsets, grad2_grid, B, L, S, H);
        // // grad_grad_out
        // // no N_C
        // kernel_input_backward_grad_out_backward<scalar_t, D, 2><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_inputs, grid, offsets, B, L, S, H, grad_grad_out);

        // // input ---> grad_grad_out & grid ---> input
        kernel_backward_part1<scalar_t, D, 2><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, grad_grad_grid, grad_out, grad_grad_inputs, offsets, B, L, S, H, grad2_input, grad_grad_out);
        kernel_input_backward_input_backward_pre<scalar_t, D, 2><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, offsets, dy2_dx, B, L, S, H);
        // // 
        kernel_input_backward_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad_out, dy2_dx, grad_grad_inputs, grad2_input, B, L);
        // // N_C grid---> grad_grad_out
        kernel_grid_backward_grad_out_backward<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, grad_grad_grid, offsets, grad_grad_out, B, L, S, H);
        break;
    // case 4: 
    //     // grid
    //     // N_C
    //     kernel_input_backward_grid_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad_out, inputs, grad_grad_inputs, offsets, grad2_grid, B, L, S, H);
    //     // input 
    //     // grid --> input;
    //     // no N_C
    //     // kernel_grid_backward_input_backward<scalar_t, D, 4><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_grid, grad_out, offsets, B, L, S, H, grad2_input);
        
    //     // // input --->input
    //     // // no N_C
    //     // kernel_input_backward_input_backward_pre<scalar_t, D, 4><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, offsets, dy2_dx, B, L, S, H);
    //     // // grad_grad_out
    //     // // no N_C
    //     // kernel_input_backward_grad_out_backward<scalar_t, D, 4><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_inputs, grid, offsets, B, L, S, H, grad_grad_out);
    //     kernel_backward_part1<scalar_t, D, 4><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, grad_grad_grid, grad_out, grad_grad_inputs, offsets, dy2_dx, B, L, S, H, grad2_input, grad_grad_out);
    //     // 
    //     kernel_input_backward_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad_out, dy2_dx, grad_grad_inputs, grad2_input, B, L);


    //     // N_C
    //     kernel_grid_backward_grad_out_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, grad_grad_grid, offsets, grad_grad_out, B, L, S, H);
    //     break;
    // case 8: 
    //     // grid
    //     // N_C
    //     kernel_input_backward_grid_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad_out, inputs, grad_grad_inputs, offsets, grad2_grid, B, L, S, H);
    //     // input 
    //     // grid --> input;
    //     // no N_C
    //     // kernel_grid_backward_input_backward<scalar_t, D, 8><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_grid, grad_out, offsets, B, L, S, H, grad2_input);
        
    //     // // input --->input
    //     // // no N_C
    //     // kernel_input_backward_input_backward_pre<scalar_t, D, 8><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, offsets, dy2_dx, B, L, S, H);

    //     // // grad_grad_out
    //     // // no N_C
    //     // kernel_input_backward_grad_out_backward<scalar_t, D, 8><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grad_grad_inputs, grid, offsets, B, L, S, H, grad_grad_out);
    //     // 
    //     kernel_backward_part1<scalar_t, D, 8><<<blocks_hashgrid1, N_THREAD1>>>(inputs, grid, grad_grad_grid, grad_out, grad_grad_inputs, offsets, dy2_dx, B, L, S, H, grad2_input, grad_grad_out);
        
    //     kernel_input_backward_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad_out, dy2_dx, grad_grad_inputs, grad2_input, B, L);

    //     // N_C
    //     kernel_grid_backward_grad_out_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, grad_grad_grid, offsets, grad_grad_out, B, L, S, H);
    //     break;
    default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
}
}


// first, we only considerate the D = 3, and C = 2, must pay attention to the 

// const uint32_t B, // batch_size
// const uint32_t* offsets, // search the grid from the 3d space
// const uint32_t H, // 
// const float S,
// // const InterpolationType interpolation_type,
// const T* __restrict__ grid, // L, C
// const float* __restrict__ positions_in, //input D, B;
// T* __restrict__ encoded_positions,  // output, L, C, B
// float* __restrict__ dy_dx  // L, C, B, D

template <typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx) {
    static constexpr uint32_t N_THREAD = 512;
    // static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        // case 1: kernel_grid<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        case 2: kernel_grid<scalar_t, 3, 2><<<blocks_hashgrid, N_THREAD>>>(B, offsets, H, S, embeddings, inputs, outputs); 
        // if (calc_grad_inputs) kernel_grid_dy_dx<scalar_t, 3, 2><<<blocks_hashgrid, N_THREAD>>>(B, offsets, H, S, embeddings, inputs, outputs, dy_dx); 
		break;
        // case 2: kernel_grid<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        // case 4: kernel_grid<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        // case 8: kernel_grid<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [L, B, C], float (L first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
// dy_dx: [B, L * D * C]
template <typename scalar_t>
void hash_encode_forward_cuda(const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx) {
    switch (D) {
        // case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, S, H, calc_grad_inputs, dy_dx); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, S, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
    
}


// template <typename T, uint32_t D, uint32_t C, uint32_t N_FEATURES_PER_THREAD>
// __global__ void kernel_grid_backward(
// 	const uint32_t B,
// 	const uint32_t* offsets,
// 	const uint32_t H,
// 	const float S,
// 	T* __restrict__ grid_gradient, // L, C
// 	const float* __restrict__ positions_in,
// 	const T* __restrict__ dL_dy  // the gradients of the node, L, C, B

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs) {
    static constexpr uint32_t N_THREAD = 256;
	const uint32_t N_C = std::min(2u, C); // n_features_per_thread
	const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };

    static constexpr uint32_t N_THREAD1 = 512;
    // static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks_hashgrid1 = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        // case 1: 
        //     kernel_grid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H); 
        //     if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
        //     break;
        case 2: 
            kernel_grid_backward<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(B, offsets, H, S, grad_embeddings, inputs, grad);
            if (calc_grad_inputs) 
                {
                    kernel_grid_grad_inputs_pre<scalar_t, 3, 2><<<blocks_hashgrid1, N_THREAD1>>>(B, offsets, H, S, embeddings, inputs, dy_dx);
                    kernel_grid_backward_input<scalar_t, D><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(B, L*C, grad, dy_dx, grad_inputs);
                }
            break;
        // case 4: 
        //     kernel_grid_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H);
        //     if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
        //     break;
        // case 8: 
        //     kernel_grid_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H);
        //     if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
        //     break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void hash_encode_backward_backward_cuda(const scalar_t * grad_out, const scalar_t * inputs,const scalar_t * grad_grad_inputs, const scalar_t * grid, const scalar_t * grad_grad_grid, const int * offsets, scalar_t * grad2_grid, scalar_t * grad2_input, scalar_t * dy2_dx, scalar_t * grad_grad_out, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H)
{
    kernel_grid_backward_backward_wrapper<scalar_t, 3>(grad_out, inputs, grad_grad_inputs, grid, grad_grad_grid, offsets, grad2_grid, grad2_input, dy2_dx, grad_grad_out, B, C, L, S, H);
}
// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution
template <typename scalar_t>
void hash_encode_backward_cuda(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs) {
    switch (D) {
        // case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "hash_encode_forward", ([&] {
        hash_encode_forward_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, S, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>());
    }));
}

void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "hash_encode_backward", ([&] {
        hash_encode_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, S, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>());
    }));
    
}

void hash_encode_backward_backward(const at::Tensor grad_out, const at::Tensor inputs, const at::Tensor grad_grad_inputs, const at::Tensor grid, const at::Tensor grad_grad_grid,const at::Tensor offsets, at::Tensor grad2_grid, at::Tensor grad2_input, at::Tensor dy2_dx, at::Tensor grad_grad_out, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H)
{
    CHECK_CUDA(grad_out);
    CHECK_CUDA(inputs);
    CHECK_CUDA(grad_grad_inputs);
    CHECK_CUDA(grid);
    CHECK_CUDA(grad_grad_grid);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad2_grid);
    CHECK_CUDA(grad2_input);
    CHECK_CUDA(dy2_dx);
    CHECK_CUDA(grad_grad_out);
    
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(grad_grad_inputs);
    CHECK_CONTIGUOUS(grid);
    CHECK_CONTIGUOUS(grad_grad_grid);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad2_grid);
    CHECK_CONTIGUOUS(grad2_input);
    CHECK_CONTIGUOUS(dy2_dx);
    CHECK_CONTIGUOUS(grad_grad_out);

    CHECK_IS_FLOATING(grad_out);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(grad_grad_inputs);
    CHECK_IS_FLOATING(grid);
    CHECK_IS_FLOATING(grad_grad_grid);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad2_grid);
    CHECK_IS_FLOATING(grad2_input);
    CHECK_IS_FLOATING(dy2_dx);
    CHECK_IS_FLOATING(grad_grad_out);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_out.type(), "hash_encode_backward_backward", ([&] {
            hash_encode_backward_backward_cuda<scalar_t>(grad_out.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), grad_grad_inputs.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), grad_grad_grid.data_ptr<scalar_t>(),offsets.data_ptr<int>(), grad2_grid.data_ptr<scalar_t>(),grad2_input.data_ptr<scalar_t>(), dy2_dx.data_ptr<scalar_t>(), grad_grad_out.data_ptr<scalar_t>(), B, C, L, S, H);
        }));

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    //     grad_out.type(), "hash_encode_backward1", ([&] {
    //         hash_encode_backward_cuda<scalar_t>(grad_out.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad2_grid.data_ptr<scalar_t>(), B, D, C, L, S, H, true, dy2_dx.data_ptr<scalar_t>(), grad2_input.data_ptr<scalar_t>());
    //     }));
}