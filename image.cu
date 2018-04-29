#include <stdio.h>
#include <ctype.h>
#include "getopt.h"
#include <stdint.h>
#include <errno.h>
#include <iostream>

using namespace std;

// CURAND library
#include <curand.h>
#include <curand_kernel.h>

// NPP Library
#include <ImagesNPP.h>
#include <ImagesCPU.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppi_arithmetic_and_logical_operations.h>

// Local includes
#include "image.hpp"   // Library
#include "main.hpp"    // Standard functionality interfaces
#include "cxxopts.hpp" // Command Line Argument Parsing (TODO: Move into library)

__constant__ uint32_t chromakey = 0x11111111;
/* NOTE: while above variable is defined in host namespace, it's value
 *  is not accessible. Trying to access it from the CPU directly will
 *  not yield any errors or warnings, but will always read with a 0-value.
 * The value can be read using cudaMemcpyFromSymbol(), but in most cases it's
 *   simpler to store a second copy in host memory for convenience.
 */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}



// Pipe two data set values together (bitwise-or)
__global__
void data_merge(unsigned int * data, unsigned int * data2)
{
  // blockNum * thradsPerBlock + threadNum
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  data[thread_idx] = data[thread_idx] | data2[thread_idx];
	
}

// Adjust data set by bitwise oring all values against a constant value (chromakey)
__global__
void key_merge(uint32_t *data, uint32_t const opt)
{
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (opt == 0) {
    data[thread_idx] = (data[thread_idx] | chromakey);
  } else {
    data[thread_idx] = (data[thread_idx] & chromakey);
  }
}

/** Flip the image horizontally
 *   This basic version assumes that the number of threads is equal to the
 *      widtdh of the image, and blocks to the height.
 */ 
__global__
void flip_image_row(uint32_t *data)
{
  extern __shared__ int row[];
  const int x = threadIdx.x; // col
  const int nx = blockDim.x; // num_threads = number of columns
  const int ybase = blockIdx.x * nx; // start of row
  
  row[x] = data[ybase+x];
  __syncthreads();
  data[ybase + nx-1 - x] = row[x];
}

/** Flip the image vertically
 *  This basic version assumes that the number of threads is equal to the
 *    height of the image, and blocks to the width.
 */ 
__global__
void flip_image_col(uint32_t *data)
{
  extern __shared__ int col[];
  const int y = threadIdx.x;
  const int dim = blockDim.x;
  const int x = blockIdx.x;

  const unsigned int idx = (y * dim) + x;
  col[y] = data[idx];
  __syncthreads();
  data[idx] = col[dim - 1 - y];
    
}

/**
 *   add random noise to the image with per-channel noise bound
 *    by the defined max noise levels (in chromakey).
 */
__global__
void add_noise(uint32_t *data, unsigned int seed)
{
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int noise;
  char ch_noise;

  curandState_t state;
  curand_init(thread_idx*seed,0,0,&state);
  
  noise = curand(&state);

  ch_noise = GET_R(noise) % GET_R(chromakey);
  noise = SET_R(noise,ch_noise);
  ch_noise = GET_G(noise) % GET_G(chromakey);
  noise = SET_G(noise,ch_noise);
  ch_noise = GET_B(noise) % GET_B(chromakey);
  noise = SET_B(noise,ch_noise);  

  data[thread_idx] += noise;
}

/**
 * This kernel should be executed once for every pixel
 */
__global__
void gpu_steg_image_en(uint32_t *data, char *msg, uint32_t msg_len)
{
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int tmp;
  unsigned char halfByte;
  
  if (thread_idx < msg_len*2)
  {
    tmp = data[thread_idx];

    // Load the half-byte from source message
    if (threadIdx.x & 0x1 == 1) {
      // Odd threads take the upper half-byte
      halfByte = (msg[(thread_idx-1)/2]) >> 4;
    } else {
      // Even threads take the lower half-byte
      halfByte = msg[thread_idx/2];
    }

    // Add cipher (Note: We don't need to mask overflow bytes, since we select bits below)
    halfByte += chromakey;

    // Bit-1 of char to Bit-1 of R
    tmp = tmp ^ (halfByte & 0x1);

    // Bit-2 of char to Bit-1 of G
    tmp = tmp ^ ((halfByte & 0x2) << 7);

    // Bit-3+4 of char to Bit-1+2 of B
    tmp = tmp ^ ((halfByte & 0xC) << 14);

      
    data[thread_idx] = tmp;
  } else {
    // Nothing to be done
  }
  
}

/**
 *  This kernel should be executed once per pixel
 */
__global__
void gpu_steg_image_de(uint32_t *data, uint32_t *data2, char *msg_out)
{
  extern __shared__ char msg[];
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint32_t tmp, tmp2;

  // Difference the two images using XOR
  tmp = data2[thread_idx] ^ data[thread_idx];

  // Calculate the half-word
  tmp2 = GET_R(tmp) & 0x1; // Bit 1
  tmp2 |= (GET_G(tmp) & 0x1) << 1; // Bit 2
  tmp2 |= (GET_B(tmp) & 0x3) << 2; // Bits 3+4
  msg[threadIdx.x] = tmp2;

  // Sync threads
  __syncthreads();

  /* Note: This next part could be somewhat optimized in theory if we
     could keep our shared memory but switch to a kernel with half the
     block size.
   */
  if (threadIdx.x & 0x1 == 1) {
    // Only odd threads will proceed

    // Merge the half-words and apply the cipher (to each half)
    tmp =   ((int)(msg[threadIdx.x-1]) - chromakey) & 0xF;
    tmp |= (((int)(msg[threadIdx.x]) - chromakey) & 0xF) << 4;

    // Output decrypted character
    msg_out[ (thread_idx-1)/2 ] = tmp;
    
  } else {
    // Even threads are now idle/stalled
  }
  
}


/** Overlay sprite onto image starting at given x offset
 *   This version will generate num_frames images, waiting on
 *   an event in between executions.
 *    Note: For proper operation, num_threads=width and numBlocks=height of
 *      src image.
 *   Pixels corresponding to the chromakey will not be copied over.
 */
__global__
void gpu_img_sprite(unsigned int* src, unsigned int* sprite,
		    unsigned int sprite_width, unsigned int sprite_height,
		    unsigned int* gpu_out,
		    unsigned int const sprite_offset
		    )
{
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int sprite_x, sprite_y;
  unsigned int sprite_idx;
  
  sprite_x = threadIdx.x - sprite_offset;
  sprite_y = blockIdx.x - sprite_offset;
  sprite_idx = (sprite_y * sprite_height) + sprite_x;

  if (sprite_x < 0 || sprite_y < 0 || sprite_y > sprite_width || sprite_x > sprite_height || sprite[sprite_idx] == chromakey) {
    gpu_out[thread_idx] = src[thread_idx];
  } else {
    gpu_out[thread_idx] = sprite[sprite_idx];
  }
}

/** Simple (Naive) Convolution Example 
 * Based on https://github.com/bgaster/opencl-book-samples/blob/master/src/Chapter_3/OpenCLConvolution/Convolution.cl
 *
 * This simple implementation is not optimized (and therefore more readable).
 *
 * This simple implementation requires image and mask to each be square. Image must be small enough to run
 *   one image row per GPU block.
 * TODO: If requirement remains in place, enforce it 
**/
__global__ void gpu_convolve(
                         unsigned int * const input,
                         int * const mask,
                         unsigned int * const output,
                         const int inputWidth,
                         const int maskWidth)
{
  const int x = threadIdx.x; //get_global_id(0);
  const int y = blockIdx.x; //get_global_id(1);

  uint sum = 0;
  for (int r = 0; r < maskWidth; r++)
    {
      const int idxIntmp = (y + r) * inputWidth + x;

      for (int c = 0; c < maskWidth; c++)
        {

          sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
        }
    }

  output[y * inputWidth + x] = sum;
}


/*** OpenCL Application ***/
class GIMD_cuda : virtual public GIMD_main
{
  cudaDeviceProp deviceProp;
  uint32_t num_blocks, num_threads;
  bool use_npp = false;
  
public:
  /** Constructor verifies device requirements and initializes hardware references */
  GIMD_cuda()
  {
    // Get and output basic device information
    if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
      throw std::runtime_error("ERROR: Unable to get device properties.\n");
    } else {
      printf("INFO: GPU supports a warpSize of %d, and a maximum of %d threads per block\n",
             deviceProp.warpSize,
             deviceProp.maxThreadsPerBlock
             );
    }
  }
  
  int run(int argc, char* argv[])
  {
    string msg = NULL;

    // Parse Arguments
    cxxopts::Options options = init_options("OpenCL Test App");
    options.add_options()
      ("b,blocks", "Number of blocks", cxxopts::value<uint32_t>(num_blocks))
      ("t,threads", "Number of threads", cxxopts::value<uint32_t>(num_threads))
      ("p,pinned", "Use PINNED memory where applicable")
      ("npp", "Use NPP Library for supported functions", cxxopts::value<bool>(use_npp))
      ;
    cxxopts::ParseResult result = parse_options(options, argc, argv);
    cout << "Arguments parsed" << endl;

    // Set chromakey constant for CUDA
    cudaMemcpyToSymbol(chromakey, &host_chromakey, sizeof(int));

    if (result.count("pinned"))
    {
      memMode = MEM_HOST_PINNED;
    }

    // Execute action
    do_action();  
	
    return 0;
  }
  
  int mod_image(uint32_t* data, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    if (use_npp) {
      return npp_mod_image(data, height, width);
    } else {
      return cuda_mod_image(data, height, width);
    }
  }

  int cuda_mod_image(uint32_t* data, const uint32_t height, const uint32_t width )
  {
    cuda_img_precheck(height, width);
    
    // Data/Array size defined in this example to match thread/block configuration
    uint32_t data_size = num_threads*num_blocks*sizeof(uint32_t);
    
    /* Declare pointers for GPU based params */
    unsigned int *gpu_data;
    
    // Define performance metrics
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc((void **)&gpu_data, data_size);
    
    cudaMemcpy( gpu_data, data, data_size, cudaMemcpyHostToDevice );
    
    /* Execute our kernel */
    cudaEventRecord(start);
    switch(mode) {
    case MODE_ADD_RAND_NOISE:
      add_noise<<<num_blocks, num_threads>>>(gpu_data, time(NULL));
      break;
    case MODE_OR_MASK:
    case MODE_AND_MASK:
      key_merge<<<num_blocks, num_threads>>>(gpu_data, mode);
      break;
    case MODE_FLIP_HOR:
      // Third parameter dynamically allocates shared memory
      flip_image_row<<<num_blocks, num_threads, num_threads*sizeof(int)>>>(gpu_data);
      break;
    case MODE_FLIP_VER:
      // Third parameter dynamically allocates shared memory
      flip_image_col<<<num_blocks, num_threads, num_blocks*sizeof(int)>>>(gpu_data);
      break;
    default:
      printf("ERROR: Invalid mode (%d) passed to mod_image function\n", mode);
    }
    
    // Wait for the GPU launched work to complete
    //   (failure to do so can have unpredictable results)
    cudaThreadSynchronize();	
    
    cudaEventRecord(stop);
    
    /* Free the arrays on the GPU as now we're done with them */
    cudaMemcpy( data, gpu_data, data_size, cudaMemcpyDeviceToHost );
    cudaFree(gpu_data);
    
    /* Iterate through the arrays and output */
    if (verbose) {
      for(unsigned int i = 0; i < num_blocks*num_threads; i++)
        {
          printf("Data: %08x - %03u %03u %03u\n",
                 data[i], GET_R(data[i]), GET_G(data[i]), GET_B(data[i]));
        }
    }
    
    // Output timing metrics
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("mod_image CUDA operation took %f ms\n", milliseconds);
    
    // Report if any errors occurred during CUDA operations
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(err));

    return 1;
  }


  // Simple steganographic message decryption
  int steg_image_de(uint32_t *data, uint32_t **output,
                    const uint32_t image_size, const uint32_t data_height, const uint32_t data_width )
  {
    cuda_img_precheck(data_height, data_width);
    if (num_blocks*num_threads > image_size) {
      throw std::runtime_error("Illegal block/thread size for this image");
    }

    int status;
    uint32_t data2_length=0, height, width;
    uint32_t *data2 = NULL; // Start: Encoded image.  End: Decoded message
    char *msg;
    int msgLen;

    /* Declare pointers for GPU based params */
    unsigned int *gpu_data;
    unsigned int *gpu_data2;
    char *gpu_msg;

    // Define performance metrics
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
 
    // Load Encoded Image
    status = load_image(output_file.c_str(), &data2, &data2_length, &height, &width, memMode);
    if (status < 0) {
      printf("ERROR: Unable to load encoded image\n");
      return -1;
    }

    // Validate that lengths match
    if (data2_length != image_size) {
      printf("ERROR: Encoded and source images must be of the same size!\n");
      return -1;
    }

    // Max message length is 1/2 the number of pixels
    msgLen = width*height/2;
    msg = (char*)malloc(msgLen);

    // Load GPU Data
    cudaMalloc((void **)&gpu_data, image_size);
    cudaMemcpy( gpu_data, data, image_size, cudaMemcpyHostToDevice );
    cudaMalloc((void **)&gpu_data2, image_size);
    cudaMemcpy( gpu_data2, data2, image_size, cudaMemcpyHostToDevice );
    cudaMalloc((void **)&gpu_msg, msgLen);

    /* Execute our kernel */
    cudaEventRecord(start);
    gpu_steg_image_de<<<num_blocks, num_threads, num_threads>>>(gpu_data, gpu_data2, gpu_msg);

    // Wait for the GPU launched work to complete
    //   (failure to do so can have unpredictable results)
    cudaThreadSynchronize();
    cudaEventRecord(stop);
  
  
    /* Cleanup */
    cudaMemcpy( msg, gpu_msg, msgLen, cudaMemcpyDeviceToHost ); 
    cudaFree(gpu_data);
    cudaFree(gpu_data2);
    cudaFree(gpu_msg);
    free(msg);
    switch(memMode) {
    case MEM_HOST_PAGEABLE:
      free(data2);
      break;
    case MEM_HOST_PINNED:
      cudaFreeHost(data2);
      break;
    }

    printf("Decoded message reads: %s \n\n", msg);
    printf("DEBUG: msg[0]=%x\n", msg[0]);

    // Output timing metrics
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("decryption CUDA operation took %f ms\n", milliseconds);

    // Report if any errors occurred during CUDA operations
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(err));

    *output = data2;
    return 1;
  }

  // Simple steganographic message decryption
  int steg_image_en(uint32_t *data, uint32_t **output, uint32_t image_size)
  {
    if (num_threads == 0) {
      num_threads = 32;
    }
    if (num_blocks == 0) {
      num_blocks = (extra.length() / num_threads)+1;
    }

  
    int msgLen = extra.length();

    /* Declare pointers for GPU based params */
    unsigned int *gpu_data;
    char *gpu_data2;

    // Define performance metrics
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
 
    // Load GPU Data
    cudaMalloc((void **)&gpu_data, image_size);
    cudaMemcpy( gpu_data, data, image_size, cudaMemcpyHostToDevice );
    cudaMalloc((void **)&gpu_data2, msgLen);
    cudaMemcpy( gpu_data2, extra.c_str(), msgLen, cudaMemcpyHostToDevice );

    /* Execute our kernel */
    cudaEventRecord(start);
    gpu_steg_image_en<<<num_blocks, num_threads, num_threads>>>(gpu_data, gpu_data2, msgLen);

    // Wait for the GPU launched work to complete
    //   (failure to do so can have unpredictable results)
    cudaThreadSynchronize();
    cudaEventRecord(stop);
  
    /* Cleanup */
    cudaMemcpy( data, gpu_data, image_size, cudaMemcpyDeviceToHost ); 
    cudaFree(gpu_data);
    cudaFree(gpu_data2);

    // Output timing metrics
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("encryption CUDA operation took %f ms\n", milliseconds);

    // Report if any errors occurred during CUDA operations
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(err));

  
    return 1;
}

  int convolve(uint32_t* data, uint32_t **output, const uint32_t height, const uint32_t width )
{
  // Sanity check
  if (data == NULL || !(height == width == num_threads == num_blocks) )
  {
    // TODO: Evaluate this limitation and remove if practical
    throw std::runtime_error("ERROR: This function currently requires height=width=num_threads=num_blocks\n");
  }
  
  // Data/Array size defined in this example to match thread/block configuration
  uint32_t data_size = num_threads*num_blocks*sizeof(uint32_t);
  uint32_t mask_size = sizeof(int32_t)*mask_width*mask_width;
  
  /* Declare pointers for GPU based params */
  unsigned int *gpu_data, *gpu_out;
  int *gpu_mask;

  // Define performance metrics
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaMalloc((void **)&gpu_data, data_size);
  cudaMemcpy( gpu_data, data, data_size, cudaMemcpyHostToDevice );
  
  cudaMalloc((void **)&gpu_out, data_size);
  
  cudaMalloc((void **)&gpu_mask, mask_size);
  cudaMemcpy( gpu_mask, mask, mask_size, cudaMemcpyHostToDevice);

  /* Execute our kernel */
  cudaEventRecord(start);
  gpu_convolve<<<num_blocks, num_threads>>>(gpu_data, gpu_mask, gpu_out,
                                            num_threads, // Provided to match original, but not necessary in current config
                                            mask_width);

  // Wait for the GPU launched work to complete
  //   (failure to do so can have unpredictable results)
  cudaThreadSynchronize();	
  
  cudaEventRecord(stop);
  
  /* Free the arrays on the GPU as now we're done with them */
  cudaMemcpy( data, gpu_out, data_size, cudaMemcpyDeviceToHost );
  cudaFree(gpu_data);
  cudaFree(gpu_out);
  cudaFree(gpu_mask);

  /* Iterate through the arrays and output */
  if (verbose) {
    for(unsigned int i = 0; i < num_blocks*num_threads; i++)
      {
	printf("Data: %08x - %03u %03u %03u\n",
	       data[i], GET_R(data[i]), GET_G(data[i]), GET_B(data[i]));
      }
  }

  // Output timing metrics
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("mod_image CUDA operation took %f ms\n", milliseconds);

  // Report if any errors occurred during CUDA operations
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  return 1;
}



/* Simple sprite animation.  The given sprite image will be overlaid
 *  on the base image. The first image will start at position 0,0, with
 *  the starting column incremented for each pass.  One image will be generated for each width-1 pixels.
 * The resulting images can be converted into an animated gif using the convert tool:
 *  convert -delay 20 -loop 0 out_fn* out_fn.gif
 *  WARNING: This conversion process can be slow. Ideally, this could be sped up
 *   by utilizing CUDA to directly convert files into GIF format and letting the
 *   CPU assemble the results into an animated GIF ... but one step at a time.
 */
int img_sprite_anim(uint32_t *data, uint32_t **output,
                    uint32_t height, uint32_t width
                    )
{
  // this.extra is sprite filename
  cuda_img_precheck(height, width);
  if (memMode != MEM_HOST_PINNED) {
    throw std::runtime_error("ERROR: Pinned memory required for this operation");
  }
  if (extra.empty()) {
    throw std::runtime_error("Sprite file must be defined for this operation");
  }
    
  uint32_t image_size = width*height;
  char out_fn[64];
  uint32_t num_images = 0;
  uint32_t *sprite_data = NULL;
  uint32_t sprite_length=0, sprite_height, sprite_width;
  uint32_t *gpu_src, *gpu_sprite, *gpu_out1, *gpu_out2;
  uint32_t *cpu_out1, *cpu_out2;
  cudaEvent_t start1, stop1, start2, stop2;
  cudaStream_t stream1, stream2;
  int status;
  
  // Load sprite image
  status = load_image(extra.c_str(),
		      &sprite_data, &sprite_length, &sprite_height, &sprite_width,
		      MEM_HOST_PINNED);
  if (status < 0) {
    printf("ERROR: Unable to load sprite\n");
    return -1;
  }
  
  // Calculate number of images to generate (must be even to simplify logic)
  num_images = width - (width&1);
  
  // Initialize remaining CUDA resources
  cudaMallocHost((void **)&cpu_out1, image_size);
  cudaMallocHost((void **)&cpu_out2, image_size);

  cudaMalloc((void **)&gpu_sprite, image_size);
  cudaMemcpy( gpu_sprite, sprite_data, sprite_length, cudaMemcpyHostToDevice );
  cudaMalloc((void **)&gpu_src, image_size);
  cudaMemcpy( gpu_src, data, image_size, cudaMemcpyHostToDevice );
  cudaMalloc((void **)&gpu_out1, image_size);
  cudaMalloc((void **)&gpu_out2, image_size);
  
  // Create events
  cudaEventCreate(&start1);
  cudaEventCreate(&start2);
  cudaEventCreate(&stop1);
  cudaEventCreate(&stop2);

  // Create streams
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Start Initial Kernels
  cudaEventRecord(start1, stream1);
  gpu_img_sprite<<<num_blocks,num_threads,0,stream1>>>(gpu_src,gpu_sprite,
						       sprite_width,sprite_height,
						       gpu_out1,
						       0);
  cudaMemcpyAsync( cpu_out1, gpu_out1, image_size, cudaMemcpyDeviceToHost, stream1 );
  cudaEventRecord(stop1, stream1);
  cudaEventRecord(start2, stream2);
  gpu_img_sprite<<<num_blocks,num_threads,0,stream2>>>(gpu_src,gpu_sprite,
						       sprite_width,sprite_height,
						       gpu_out2,
						       1);
  cudaMemcpyAsync( cpu_out2, gpu_out2, image_size, cudaMemcpyDeviceToHost, stream2 );
  cudaEventRecord(stop2, stream2);
  
  // Generate Frames
  for(int i = 0; i < num_images/2; i++)
  {
    cudaStreamSynchronize(stream1);

    // Write buffer 0 to disk
    sprintf(out_fn, "%s[%04d].ppm", output_file, 2*i);
    write_image(out_fn, cpu_out1, width, height);

    // Restart stream1 for next iteration (if this isn't the last iteration)
    if (i+1 != num_images/2) {
      cudaEventRecord(start1, stream1);
      gpu_img_sprite<<<num_blocks,num_threads,0,stream1>>>(gpu_src,gpu_sprite,
							   sprite_width,sprite_height,
							   gpu_out1,
							   2*(i+1));

      // Copy data to CPU from buffer 0
      cudaMemcpyAsync( cpu_out1, gpu_out1, image_size, cudaMemcpyDeviceToHost, stream1 );
      cudaEventRecord(stop1, stream1);
    }

    cudaStreamSynchronize(stream2);
    
    // Write buffer 2 to disk
    sprintf(out_fn, "%s[%04d].ppm", output_file, 2*i+1);
    write_image(out_fn, cpu_out2, width, height);

    // Start next kernel
    if (i+1 != num_images/2) {
      cudaEventRecord(start2, stream2);
      gpu_img_sprite<<<num_blocks,num_threads,0,stream2>>>(gpu_src,gpu_sprite,
							   sprite_width,sprite_height,
							   gpu_out2,
							   1+(2*(i+1)));

      // Copy data to CPU from buffer 0
      cudaMemcpyAsync( cpu_out2, gpu_out2, image_size, cudaMemcpyDeviceToHost, stream2 );
      cudaEventRecord(stop2, stream2);
    }
    
    
  }

  // Cleanup
  // free CPU+GPU output buffers
  cudaFree(gpu_out1);
  cudaFree(gpu_out2);
  cudaFreeHost(cpu_out1);
  cudaFreeHost(cpu_out2);
  
  // free Sprite buffers
  cudaFree(gpu_sprite);
  cudaFreeHost(sprite_data);
  
  // Note: main fn will free main image cpu buffer
  cudaFree(gpu_src);

  return 0;
}

/** Perform Image Processing functions using the NPP Library NOTE:
 *   While we are loading 3-channel 8-bit RGB images, we will parse as
 *   4-channel images since that data format for NPP matches the
 *   format already used in this file.  The fourth channel (nominally
 *   alpha) is simply ignored with our current import/export
 *   functions.
 */
  int npp_mod_image(uint32_t *data, uint32_t height, uint32_t width)
{
  NppiSize size = {width, height};
  NppStatus status;

  npp::ImageCPU_8u_C4 oHost(width, height);
  memcpy(oHost.data(), data, (width*height*sizeof(uint32_t)) );
  npp::ImageNPP_8u_C4 oDevice(oHost);
  printf("chromakey = %x\n", host_chromakey);
  
  switch(mode) {
  case MODE_AND_MASK:
    status = nppiAndC_8u_C4IR((const Npp8u *)&host_chromakey, oDevice.data(), oDevice.pitch(), size);
    break;
  case MODE_OR_MASK:
    status = nppiOrC_8u_C4IR((const Npp8u *)&host_chromakey, oDevice.data(), oDevice.pitch(), size);
  }

  if (status < 0) {
    printf("ERROR: NPP Operation failed with %i\n");
  }

  oDevice.copyTo(oHost.data(), oHost.pitch());
  memcpy(data, oHost.data(), (width*height*sizeof(uint32_t)) );
  //oDevice.copyTo((Npp8u*)data, oDevice.pitch());
  nppiFree(oDevice.data());
}
  void cuda_img_precheck(uint32_t height, uint32_t width) {
    // Sanity Checks
    if (num_threads == 0) {
      num_threads = width;
    }
    if (num_blocks == 0) {
      //num_blocks = height;
      num_blocks = (width*height)/num_threads;
    }

    if (num_blocks*num_threads > height*width) {
      printf("ERROR: num_blocks&num_threads must be <= image size (%d <= %d)\n", num_blocks*num_threads, height*width);
      throw std::runtime_error("ERROR: Invalid parameters");
    }
    printf("Processing image %s fn of size (%d x %d) with %d threads and %d blocks\n",
           input_file, height, width, num_threads, num_blocks);
}
  
}; // End class GIMP_cuda
  
int main(int argc, char* argv[])
{
#if 1
  // Declare app and initialize
  GIMD_cuda app;

  // Parse Arguments, Execute and run
  return app.run(argc, argv);

#else
  
      case MODE_STEG_DE:
	steg_image_de(num_threads, num_blocks, image, image_size, output_file.c_str(), memMode);
	skip_img_write = 1;
	break;
      case MODE_STEG_EN:
	if (msg.length() == 0) {
	  printf("ERROR: Encryption needs a message to encrypt!\n");
	  break;
	}
	steg_image_en(num_threads, num_blocks, image, image_size, msg.c_str(), memMode);
	break;

#endif
}
