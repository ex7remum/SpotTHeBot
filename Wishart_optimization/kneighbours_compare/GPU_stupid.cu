%%cu
#include <stdio.h>
#include <cuda.h>
#include <cublas.h>

#define BLOCK_DIM 16

//source https://github.com/vincentfpgarcia/kNN-CUDA

/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
__global__ void compute_distances(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initialization of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height-1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }
}

/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i=1; i<height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}


/**
 * Computes the square root of the first k lines of the distance matrix.
 *
 * @param dist   distance matrix
 * @param width  width of the distance matrix
 * @param pitch  pitch of the distance matrix given in number of columns
 * @param k      number of values to consider
 */
__global__ void compute_sqrt(float * dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}

bool knn_cuda_global(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,
                     int *         knn_index) {

    // Constants
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Return variables
    cudaError_t err0, err1, err2, err3;

    // Check that we have at least one CUDA device 
    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }

    // Allocate global memory
    float * ref_dev   = NULL;
    float * query_dev = NULL;
    float * dist_dev  = NULL;
    int   * index_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    err0 = cudaMallocPitch((void**)&ref_dev,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    err3 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int,   k);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("ERROR: Memory allocation error\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }

    // Deduce pitch values
    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }

    // Copy reference and query data from the host to the device
    err0 = cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,   ref,   ref_nb * size_of_float,   ref_nb * size_of_float,   dim, cudaMemcpyHostToDevice);
    err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }

    // Compute the squared Euclidean distances
    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
    if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
    if (ref_nb   % BLOCK_DIM != 0) grid0.y += 1;
    compute_distances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }

    // Sort the distances with their respective indexes
    dim3 block1(256, 1, 1);
    dim3 grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0) grid1.x += 1;
    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }

    // Compute the square root of the k smallest distances
    dim3 block2(16, 16, 1);
    dim3 grid2(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0) grid2.x += 1;
    if (k % 16 != 0)        grid2.y += 1;
    compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);	
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    err0 = cudaMemcpy2D(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch_in_bytes,  query_nb * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_nb * size_of_int,   index_dev, index_pitch_in_bytes, query_nb * size_of_int,   k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }

    // Memory clean-up
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev); 

    return true;
}

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Get reference points
     srand(time(NULL));
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Get query points
    for (int i=0; i<query_nb*dim; ++i) {
       query[i] = ref[i];
    }
}


/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    for (int i = 0; i < query_nb * k; ++i) {
        printf("%d ", test_knn_index[i]);
    }
    for (int i = 0; i < query_nb * k; ++i) {
        printf("%f ", test_knn_dist[i]);
    }

    printf("\nPASSED in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}

int main(void) {

    // Parameters
    const int ref_nb   = 1000;
    const int query_nb = 1000;
    const int dim      = 4;
    const int k        = 11;

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);


    // Test all k-NN functions
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_global,  "knn_cuda_global",  1); 
    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
