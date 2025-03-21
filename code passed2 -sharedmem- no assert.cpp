

#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);

        
    }

    // if (tid < 10) {
    //     printf("[Move] Particle %d new pos: (%.5e, %.5e), vel: (%.5e, %.5e)\n",
    //         tid, p->x, p->y, p->vx, p->vy);
    //     }
}




// Bin Counting 
__global__ void bin_count_kernel(particle_t* particles, int num_parts, int* bin_counts, int bins_per_row, double bin_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int bin_x = particles[tid].x / bin_size;
    int bin_y = particles[tid].y / bin_size;
    int bin_id = bin_x + bin_y * bins_per_row;

    atomicAdd(&bin_counts[bin_id], 1);

    // if (tid < 1000) {  // Limit output to first 10 particles
    //     printf("[Bin Count] Particle %d -> Bin (%d, %d), Bin ID = %d\n", tid, bin_x, bin_y, bin_id);
    // }
}


// Particle Sorting into bins 
__global__ void bin_sort_kernel(particle_t* particles, particle_t* sorted_particles, 
    int* particle_ids, int* sorted_ids,
    int num_parts, int* bin_prefix, int* bin_offsets, 
    int bins_per_row, double bin_size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid >= num_parts) return;

particle_t p = particles[tid];
int id = particle_ids[tid];

int bin_x = p.x / bin_size;
int bin_y = p.y / bin_size;
int bin_id = bin_x + bin_y * bins_per_row;

int offset = atomicAdd(&bin_offsets[bin_id], 1);
int target_idx = bin_prefix[bin_id] + offset;

sorted_particles[target_idx] = p;
sorted_ids[target_idx] = id;  // preserve original ID



    // if (tid < 1000) {
    //     printf("[Sort] Particle %d placed in Bin ID %d at sorted index %d\n", tid, bin_id, target_idx);
    // }
}
__global__ void compute_forces_gpu(
    particle_t* sorted_particles, int num_parts, int* bin_prefix, int bins_per_row, double bin_size
) {
    int bin_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_bins = bins_per_row * bins_per_row;
    if (bin_id >= num_bins) return;

    int bin_start = bin_prefix[bin_id];
    int bin_end = bin_prefix[bin_id + 1];

    if (bin_start == bin_end) return;  // Empty bin, nothing to do.

    int bin_x = bin_id % bins_per_row;
    int bin_y = bin_id / bins_per_row;

    for (int i = bin_start; i < bin_end; i++) {
        particle_t p = sorted_particles[i];  // Pointer to particle directly in global memory
        p.ax = 0.0;
        p.ay = 0.0;
        
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = bin_x + dx;
                int ny = bin_y + dy;
        
                if (nx >= 0 && ny >= 0 && nx < bins_per_row && ny < bins_per_row) {
                    int neighbor_bin_id = nx + ny * bins_per_row;
        
                    int neigh_start = bin_prefix[neighbor_bin_id];
                    int neigh_end = bin_prefix[neighbor_bin_id + 1];
        
                    bool same_bin = (neighbor_bin_id == bin_id);
        
                    for (int j = neigh_start; j < neigh_end; j++) {
                        if (!same_bin || i != j) {
                            apply_force_gpu(p, sorted_particles[j]);

                            // Correct debugging statement here:
                            // if (bin_id == 0 && (i - bin_start) == 0 && neighbor_bin_id < 3) {
                            //     particle_t neighbor_p = sorted_particles[j];
                            //     printf("[DEBUG NEIGHBOR] Bin 0 Particle %d interacts with Neighbor Bin %d Particle %d (%.5f, %.5f)\n", 
                            //             i, neighbor_bin_id, j, neighbor_p.x, neighbor_p.y);
                            // }
                        }
                    }
                }
            }
        }

        sorted_particles[i].ax = p.ax;
        sorted_particles[i].ay = p.ay;
        

        // if (bin_id == 0 && (i - bin_start) < 3) {
        //     printf("[DEBUG FORCES FIXED] Bin %d, Particle %d ax=%.5e ay=%.5e\n",
        //             bin_id, i, p.ax, p.ay);
        // }
    }
}




void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

void simulate_one_step(particle_t* parts_gpu, int num_parts, double size) {

    // Create particle IDs on host:
std::vector<int> particle_ids(num_parts);
for (int i = 0; i < num_parts; i++)
    particle_ids[i] = i;

// Allocate GPU array for particle IDs:
int* particle_ids_gpu;
cudaMalloc(&particle_ids_gpu, num_parts * sizeof(int));
cudaMemcpy(particle_ids_gpu, particle_ids.data(), num_parts * sizeof(int), cudaMemcpyHostToDevice);

// Allocate GPU memory for sorted IDs:
int* sorted_ids_gpu;
cudaMalloc(&sorted_ids_gpu, num_parts * sizeof(int));

    int bins_per_row = ceil(size / cutoff);
    int num_bins = bins_per_row * bins_per_row;
    double bin_size = size / bins_per_row;

    thrust::device_vector<int> bin_counts(num_bins, 0);
    thrust::device_vector<int> bin_prefix(num_bins + 1);
    thrust::device_vector<int> bin_offsets(num_bins, 0);
    particle_t* sorted_particles_gpu;
    cudaMalloc(&sorted_particles_gpu, num_parts * sizeof(particle_t));

    // Count particles per bin
    bin_count_kernel<<<blks, NUM_THREADS>>>(parts_gpu, num_parts, thrust::raw_pointer_cast(bin_counts.data()), bins_per_row, bin_size);

    // Compute prefix sums
    thrust::exclusive_scan(bin_counts.begin(), bin_counts.end(), bin_prefix.begin());
    bin_prefix[num_bins] = num_parts;

// Sort particles by bin (updated kernel call)
bin_sort_kernel<<<blks, NUM_THREADS>>>(
    parts_gpu, sorted_particles_gpu, particle_ids_gpu, sorted_ids_gpu,
    num_parts, thrust::raw_pointer_cast(bin_prefix.data()),
    thrust::raw_pointer_cast(bin_offsets.data()), bins_per_row, bin_size);

    // Compute forces 
    int bin_blocks = (num_bins + NUM_THREADS - 1) / NUM_THREADS;
    compute_forces_gpu<<<bin_blocks, NUM_THREADS>>>(sorted_particles_gpu, num_parts, thrust::raw_pointer_cast(bin_prefix.data()), bins_per_row, bin_size);

    // Move particles 
    move_gpu<<<blks, NUM_THREADS>>>(sorted_particles_gpu, num_parts, size);



// After GPU kernels finish 
// Copy sorted particles and sorted IDs back to CPU

// Reordering step 
std::vector<particle_t> sorted_particles_host(num_parts);
std::vector<int> sorted_ids_host(num_parts);

cudaMemcpy(sorted_particles_host.data(), sorted_particles_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
cudaMemcpy(sorted_ids_host.data(), sorted_ids_gpu, num_parts * sizeof(int), cudaMemcpyDeviceToHost);

std::vector<particle_t> reordered_particles(num_parts);
for (int i = 0; i < num_parts; i++)
    reordered_particles[sorted_ids_host[i]] = sorted_particles_host[i];

cudaMemcpy(parts_gpu, reordered_particles.data(), num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);


// Free GPU memory
cudaFree(sorted_particles_gpu);
cudaFree(sorted_ids_gpu);
cudaFree(particle_ids_gpu);




}
