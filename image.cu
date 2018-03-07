#include <stdio.h>
#include <ctype.h>
#include "getopt.h"
#include <stdint.h>
#include <errno.h>

__constant__ uint32_t chromakey = 0x11111111;

typedef enum MemMode_t {
  MEM_HOST_PAGEABLE,
  MEM_HOST_PINNED
} MemMode_t;

typedef enum ImageActions_t {
  MODE_PIPE_MASK = 0,
  MODE_MASK = 1,
  MODE_FLIP_HOR,
  MODE_FLIP_VER,
  MODE_UNSPECIFIED
} ImageActions_t;


/* Image Generation Configuration
 *  We'll use a simple RGB color scheme
 *  This can be extended to other schemes (ie: sRGB, IAB) if needed later
 */
// 10-bit color space -- good in theory, but harder to display in a useful format
// Note: We still reserve 10-bits per channel, but only use 8 when outputting
#define RGB_COMPONENT_COLOR 255
#define MAX_COLOR RGB_COMPONENT_COLOR
#define R_SHIFT 0
#define G_SHIFT 10
#define B_SHIFT 20
#define R_MASK 0x000003FF
#define G_MASK 0x000FFC00
#define B_MASK 0x3FF00000
#define S_MASK 0xC0000000 // reserved (ie: alpha channel)

#define GET_R(data) (data & R_MASK)
#define GET_G(data) ((data & G_MASK) >> G_SHIFT)
#define GET_B(data) ((data & B_MASK) >> B_SHIFT)

#define GET_Rxy(buf,x,y) (GET_R(buf[y*width+x]))
#define GET_Gxy(buf,x,y) (GET_G(buf[y*width+x]))
#define GET_Bxy(buf,x,y) (GET_B(buf[y*width+x]))

// Write cpu_data as a PPM-formatted image (http://netpbm.sourceforge.net/doc/ppm.html)
void write_image(char *fn, uint32_t *buf, unsigned int width, unsigned int height)
{
  FILE *f;
  uint8_t c[3];

  f = fopen(fn, "wb");
  if (f == NULL) {
    perror("Unable to write output file\n");
    return;
  }
  fprintf(f, "P6\n%i %i %i\n", width, height, MAX_COLOR);
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      c[0] = GET_Rxy(buf, x,y);
      c[1] = GET_Gxy(buf, x,y);
      c[2] = GET_Bxy(buf, x,y);
      fwrite(c, 1, 3, f);
      //printf("%d,%d = %d %d %d\n", x,y,c[0],c[1],c[2]);
    }
  }
  fclose(f);
}

/**
 * @param[in] fn Filename to load
 * @param[inout] buff Buffer to load into.  If NULL, buffer will be malloced
 * @param[inout] buf_size Pointer to a variable containing the length of the provided buffer (in int32 elements), if applicable. 
 *               Otherwise, will be set to the length of the allocated buffer.
 * @param[out] length of loaded image (in pixels)
 * @param[out] width of loaded image.
 * @return 1 on success, -1 on failure.
 */
int load_image(char* fn, uint32_t **buff, uint32_t *buf_length, uint32_t *height, uint32_t *width, MemMode_t mem_mode)
{
  FILE *fp;
  int c, rgb_comp_color, r, g, b;
  uint32_t size;
  char tmp[16];
  cudaError_t cuda_status;

  printf("Loading file %s\n", fn);
  fflush(stdout);
  fp = fopen(fn, "rb");
  if (!fp) {
    printf("Unable to open file %s\n", fn);
    return -1;
  }

  //read image format
  if (!fgets(tmp, sizeof(tmp), fp)) {
    perror(fn);
    return -1;
  }

  //check the image format
  if (tmp[0] != 'P' || tmp[1] != '6') {
    perror("Invalid image format (must be 'P6')\n");
    return -1;
  }

  //check for comments
  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n') ;
    c = getc(fp);
  }
  ungetc(c, fp); // put the non-comment character back

  //read image size information
  if (fscanf(fp, "%d %d", height, width) != 2) {
    fprintf(stderr,"Invalid image size (error loading '%s')\n", fn);
    exit(1);
  }
  size = *height * *width * sizeof(int32_t);

  //read rgb component
  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", fn);
    exit(1);
  }

  //check rgb component depth
  if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", fn);
    exit(1);
  }

  // Verify or allocate memory
  if (buff == NULL || buf_length==NULL || height==NULL || width==NULL) {
    perror("Invalid parameters\n");
    return -1;
  } else if (*buff == NULL) {
    // Buffer not defined, malloc one
    switch(mem_mode) {
    case MEM_HOST_PAGEABLE:
      *buff = (unsigned int*)malloc(size);//1048576); // ERROR: Under Windows/cygwin, this fails unless value is hard-coded...
      if (*buff == NULL) {
	printf("ERROR: Malloc(%d) failed with %s\n", size, strerror(errno));
	return -1;
      }
      break;
    case MEM_HOST_PINNED:
      cuda_status = cudaMallocHost((void **)buff, size );
      if (cuda_status != cudaSuccess) {
	printf("ERROR: CudaMallocHost failed with %x\n", cuda_status);
	return -1;
      }
      break;
  default:
      fprintf(stderr, "Invalid memory mode selected\n");
      exit(1);
    }
    *buf_length = size;
    printf("Allocated buffer for image of length %d\n", size);
  } else if (*buf_length != NULL && *buf_length < size ) {
    printf("ERROR: buf_length (%d) undefined or insufficent for image (%d x %d = %d)\n", buf_length, *height, *width, size);
    return -1;
  } else {
    printf("Using pre-allocated buffer (0x%x)\n", buff);
  }

  while (fgetc(fp) != '\n') ; // Flush remaining ASCII.
  
  //read pixel data from file
  // NOTE: We could optimize the load process if we allocated a 3-bit*length buffer instead.
  for(int i = 0; i < size/sizeof(int32_t); i++) {
    r = getc(fp); g = getc(fp); b = getc(fp);
    if (r == EOF || g == EOF || b == EOF) {
      printf("Error loading image '%s' pixel %d\n", fn, i);
      return -1;
    }
    (*buff)[i] = (r & R_MASK) | (g<<G_SHIFT)&G_MASK | (b<<B_SHIFT)&B_MASK;


    /*    if (fread(&c, 3, 1, fp) != 3) {
      printf("Error loading image '%s' pixel %d\n", fn, i);
      return -1;
    }
    (*buff)[i] = c;*/
    // TODO: We are loading 3-bytes into an int32. Verify that they are loaded as expected into the lower 3-bytes
  }

  fclose(fp);
  return 1;
    
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
  /*extern*/ __shared__ int row[512];
  int x = threadIdx.x; // col
  int nx = blockDim.x; // num_threads = number of columns
  int ybase = blockIdx.x * nx; // start of row
  

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


void mod_image(unsigned int num_threads, unsigned int num_blocks, int verbose,
		uint32_t* data, ImageActions_t const mode)
{
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
  case MODE_PIPE_MASK:
  case MODE_MASK:
    key_merge<<<num_blocks, num_threads>>>(gpu_data, mode);
    break;
  case MODE_FLIP_HOR:
    // Third parameter dynamically allocates shared memory
    flip_image_row<<<num_blocks, num_threads/*, num_threads*sizeof(int)*/>>>(gpu_data);
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
  printf("mask_merge CUDA operation took %f ms\n", milliseconds);

  // Report if any errors occurred during CUDA operations
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  
}

int main(int argc, char* argv[])
{
  int c;
  int verbose = 0;
  char *fn = NULL;
  char out_fn[64] = "out.ppm";
  int status;
  unsigned int num_blocks = 0;
  unsigned int num_threads = 0;
  uint32_t *data = NULL;
  uint32_t data_length, height, width;
  MemMode_t memMode = MEM_HOST_PAGEABLE;
  int n = 1;
  int tmp;
  ImageActions_t mode = MODE_PIPE_MASK;
  cudaDeviceProp deviceProp;

  // Get and output basic device information
  if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
    printf("ERROR: Unable to get device properties.\n");
    return -1;
  } else {
    printf("INFO: GPU supports a warpSize of %d, and a maximum of %d threads per block\n",
	   deviceProp.warpSize,
	      deviceProp.maxThreadsPerBlock
	   );
  }

  // Parse input arguments
  while((c = getopt(argc, argv, "phvi:t:n:o:k:m:")) != -1) {
    switch(c) {
    case 'k':
      tmp = atoi(optarg);
      cudaMemcpyToSymbol(chromakey, &tmp, sizeof(int));
      break;
    case 'm':
      // Parse mode: This is easier than bringing in Boost for more user-friendly command-line parsing.
      mode = (ImageActions_t)atoi(optarg);
      break;
    case 'v':
      verbose = 1;
      break;
     case 'b':
      num_blocks = atoi(optarg);
      break;
    case 't':
      num_threads = atoi(optarg);
      break;
    case 'n':
      n = atoi(optarg);
      break;
    case 'i':
      fn = optarg;
      break;
    case 'o':
      strncpy(out_fn,optarg,64);
      break;
    case 'p':
      memMode = MEM_HOST_PINNED;
      break;
    case 'h':
      printf("Usage: \n");
      printf("\t-h     Show this message\n");
      printf("\t-m #   Specify action to be performed.  Currently supported modes are:\n");
      printf("\t          0  Image Mask - Bitwise or each pixel value with provided key\n");
      printf("\t          1  Image Mask - Bitwise and each pixel value with provided key\n");
      printf("\t          2  Image Flip Horizontal*\n");
      printf("\t          3  Image Flip Vertical*\n");
      printf("\t-k #   Specify key value (ie: chromakey mask or cipher value)\n");
      printf("\t-v     Enable verbose output mode.\n");
      printf("\t-i     Select image file to load (ppm format required)\n");
      printf("\t-o     Select output filename (ppm format)\n");
      printf("\t-p     Use pinned host memory (cudaMallocHost) instead of the default pageable (malloc).\n");
      printf("\t-n     Repeat action n times\n");
      printf("\n\n");
      printf("* At this time, these operations are intended to operate where\n");
      printf("  num_threads equals the width of the image. If this is not the case,\n");
      printf("  output may vary. For example, if num_threads is half the image\n");
      printf("  width, then the flip function will flip the left and right halves of\n");
      printf("  the image distinctly as if they are seperate images.  Logic to\n");
      printf("  handle such cases correctly is reserved for future enhancements.\n");
      printf("\n");
      printf("**To keep things simple, at this time operations may assume that images\n");
      printf("** are square (height=width), and works best when they are a power of 2\n");
      printf("   or a multiple of 32 (warp size).\n");
      return 1;
    default:
      printf("ERROR: Option %s is not supported, use -h for usage info.\n", (char)c);
      return -1;
    }
  }

  if (fn == NULL) {
    perror("ERROR: Filename (-i) required for input\n");
    return -1;
  }

  // Load image
  status = load_image(fn, &data, &data_length, &height, &width, memMode);
  if (status < 0) {
    return -1;
  }

  if (num_threads == 0) {
    num_threads = width;
  }
  if (num_blocks == 0) {
    //num_blocks = height;
    num_blocks = (width*height)/num_threads;
  }

  if (num_blocks*num_threads > data_length)
  {
    printf("ERROR: num_blocks&num_threads must be <= image size (%d <= %d)\n", num_blocks*num_threads, data_length);
    return -1;
  }
  printf("Processing image %s fn of size (%d x %d) with %d threads and %d blocks\n",
	 fn, height, width, num_threads, num_blocks);

  // Simple Image Processing: Merge with a mask to "brighten"
  while(n > 0) {
    switch(mode) {
    case MODE_PIPE_MASK:
    case MODE_MASK:
    case MODE_FLIP_HOR:
    case MODE_FLIP_VER:
      mod_image(num_threads, num_blocks, verbose, data, mode);
      break;
    default:
      printf("ERROR: Mode %d not currently supported. Nothing to do\n", mode);
      n = 0;
      break;
    }
    n--;
  }

  // FUTURE: Above processing can be made configurable and/or strung together in multiple phases.
  
  // Write the image back out
  write_image(out_fn, data, width, height);
  
  switch(memMode) {
  case MEM_HOST_PAGEABLE:
    free(data);
    break;
  case MEM_HOST_PINNED:
    cudaFreeHost(data);
    break;
  }
  
	
  return EXIT_SUCCESS;
}
