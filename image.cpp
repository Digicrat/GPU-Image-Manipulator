/* Image Processing and related common utility functions
 *   This file is declared as C++ for convenience, but with a few exceptions, 
 *   most utility functions are largely pure C.
 */
#include "image.hpp"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#ifdef __NVCC__
  #include "cuda_runtime.h"
#endif


/** Write cpu_data as a PPM-formatted image (http://netpbm.sourceforge.net/doc/ppm.html)
 */
void write_image(const char *fn, const uint32_t *buf, const unsigned int width, const unsigned int height)
{
  FILE *f;
  uint8_t c[3];

  printf("Preparing to output image to %s\n", fn);

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

/** Load PPM-format image into memory
 * @param[in] fn Filename to load
 * @param[inout] buff Buffer to load into.  If NULL, buffer will be malloced
 * @param[inout] buf_size Pointer to a variable containing the length of the provided buffer (in int32 elements), if applicable. 
 *               Otherwise, will be set to the length of the allocated buffer.
 * @param[out] length of loaded image (in pixels)
 * @param[out] width of loaded image.
 * @return 1 on success, -1 on failure.
 */
int load_image(const char* fn, uint32_t **buff, uint32_t *buf_length, uint32_t *height, uint32_t *width, MemMode_t mem_mode)
{
  FILE *fp;
  int c, rgb_comp_color, r, g, b;
  uint32_t size;
  char tmp[16];
#ifdef __NVCC__
  cudaError_t cuda_status;
#endif
  
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
#ifdef __NVCC__
    case MEM_HOST_PINNED:
      cuda_status = cudaMallocHost((void **)buff, size );
      if (cuda_status != cudaSuccess) {
	printf("ERROR: CudaMallocHost failed with %x\n", cuda_status);
	return -1;
      }
      break;
#endif
  default:
    fprintf(stderr, "Invalid/unsupported memory mode %d selected\n", mem_mode);
      exit(1);
    }
    *buf_length = size;
    printf("Allocated buffer for image of length %d\n", size);
  } else if (*buf_length != NULL && *buf_length < size ) {
    printf("ERROR: buf_length (%d) undefined or insufficent for image (%d x %d = %d)\n",
	   buf_length, *height, *width, size);
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

#ifdef VERBOSE
    for(unsigned int i = 0; i < 8; i++)
      {
	printf("Data: %08x - %03u %03u %03u\n",
	       (*buff)[i], GET_R((*buff)[i]), GET_G((*buff)[i]), GET_B((*buff)[i]));
      }
#endif

  
  return 1;
    
}

  
