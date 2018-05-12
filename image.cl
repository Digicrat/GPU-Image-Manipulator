
// Constants: Duplicated from image.hpp for the benefit of OpenCL sources
#define RGB_COMPONENT_COLOR 255
#define MAX_COLOR RGB_COMPONENT_COLOR
#define R_SHIFT 0
#define G_SHIFT 8
#define B_SHIFT 16
#define R_MASK 0x000000FF
#define G_MASK 0x0000FF00
#define B_MASK 0x00FF0000
#define S_MASK 0xFF000000 // reserved (ie: alpha channel)

#define GET_R(data) (data & R_MASK)
#define GET_G(data) ((data & G_MASK) >> G_SHIFT)
#define GET_B(data) ((data & B_MASK) >> B_SHIFT)



// Convolution.cl
//
//    This is a simple kernel performing (naive 2D) convolution.
__kernel void convolve(
	const __global  uint * const input,
    __constant uint * const mask,
    __global  uint * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint sum = 0;
    uint maskSum = 0; // NEW?
    for (int r = 0; r < maskWidth; r++)
    {
      int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
         
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
                        maskSum += mask[(r * maskWidth)  + c];
        }
    } 
    
    output[y * inputWidth/*get_global_size(0)*/ + x] = sum;
}

// Sprite Image Animation
__kernel void overlay(
   const __global uint * input,
   const __global uint * sprite,
   const uint sprite_height,
   const uint sprite_width,
   const uint  chromakey, const int sprite_offset,
   __global uint * out
   )
{
   int sprite_x, sprite_y;
   uint sprite_idx;
   uint thread_idx = get_global_id(0) + get_global_id(1)*get_global_size(0);
   sprite_x = get_global_id(0) - sprite_offset;
   sprite_y = get_global_id(1) - sprite_offset;
   sprite_idx = (sprite_y * sprite_height) + sprite_x;

   if (sprite_x < 0 || sprite_y < 0
       || sprite_y > sprite_width
       || sprite_x > sprite_height
       || sprite[sprite_idx] == chromakey
      ) {
      
      out[thread_idx] = input[thread_idx];
   } else {
      out[thread_idx] = sprite[sprite_idx];
   }
}

/** Simple Masking Operation/Demo
 *    Assume a global work size of height*width
 */
__kernel void do_and_mask(
      const __global uint * input,
      __global uint * out,
      const uint chromakey
   )
{
   const uint idx = get_global_id(0);
   out[idx] = input[idx] & chromakey;
}

/** Simple Masking Operation/Demo
 *    Assume a global 2D work size of height,width
 */
__kernel void do_or_mask(
      const __global uint * input,
      __global uint * out,
      const uint chromakey
   )
{
   const uint idx = get_global_id(0) + get_global_id(1)*get_global_size(0);
   out[idx] = input[idx] | chromakey;
}


__kernel void do_flip_hor(
      const __global uint * input,
      __global uint * out,
      const uint chromakey
   )
{
   const uint x = get_global_id(0);
   const uint y = get_global_id(1);
   const uint height = get_global_size(0);
   const uint width = get_global_size(1);
   const uint idx = x + y*width;
   const uint idx2 = (width-x-1) + y*width;
   out[idx] = input[idx2];
}
__kernel void do_flip_ver(
      const __global uint * input,
      __global uint * out,
      const uint chromakey
   )
{
   const uint x = get_global_id(0);
   const uint y = get_global_id(1);
   const uint height = get_global_size(0);
   const uint width = get_global_size(1);
   const uint idx = x + y*width;
   const uint idx2 = x + (height-1-y)*width;
   out[idx] = input[idx2];
}


__kernel void do_steg_en(
   __global uint * data,
   const __global char * msg,
   const uint msg_len,
   const uint chromakey
   )
{
   const uint idx = get_global_id(0);
   unsigned char halfByte;

   if (idx < msg_len*2)
   {
      unsigned int tmp = data[idx];

      // Load the half-byte from source message
      if ((idx & 0x1) == 1) {
       // Odd threads take the upper half-byte
       halfByte = (msg[(idx-1)/2]) >> 4;
    } else {
       // Even threads take the lower half-byte
       halfByte = msg[idx/2];
    }

    // Add cipher (Note: We don't need to mask overflow bytes, since we select bits below)
    halfByte += chromakey;

    // Bit-1 of char to Bit-1 of R
    tmp = tmp ^ (halfByte & 0x1);

    // Bit-2 of char to Bit-1 of G
    tmp = tmp ^ ((halfByte & 0x2) << 7);

    // Bit-3+4 of char to Bit-1+2 of B
    tmp = tmp ^ ((halfByte & 0xC) << 14);

    data[idx] = tmp;
      
   }
   // else nothing to be done
}


char do_steg_de_step(const uint data, const  uint data2, const uint chromakey)
{
   // Difference the two images
   uint tmp = data2 ^ data;

   // Calculate the half-word
   uint rtv = GET_R(tmp) & 0x1; // Bit 1
   rtv |= (GET_G(tmp) & 0x1) << 1; // Bit 2
   rtv |= (GET_B(tmp) & 0x3) << 2; // Bits 3+4
   return (rtv & 0xF);
   
}


__kernel void do_steg_de(
   const __global uint * data,
   const __global uint * data2,
   __global char * msg_out,
   const uint chromakey
   )
{
   const uint idx = get_global_id(0)*2;
#if 1 // Cleaner implementation
   
   msg_out[get_global_id(0)] = (do_steg_de_step(data[idx], data2[idx], chromakey)
                   | (do_steg_de_step(data[idx+1], data2[idx+1], chromakey) << 4)
      );
   
#else // Alternative Implementation using local variables and synchronization
   __local char msg[128]; // Or pass as fn parameter: __local char* msg
   uint tmp, word1, word2;
   __local char msg[128];

  // Difference the two images using XOR
  tmp = data2[idx] ^ data[idx];

  // Calculate the half-word
  tmp2 = GET_R(tmp) & 0x1; // Bit 1
  tmp2 |= (GET_G(tmp) & 0x1) << 1; // Bit 2
  tmp2 |= (GET_B(tmp) & 0x3) << 2; // Bits 3+4
  msg[local_idx] = tmp2;

  // Sync threads
   barrier(CLK_LOCAL_MEM_FENCE);

   
   // Merge the half-words and apply the cipher (to each half)
   tmp =   ((int)(msg[local_idx-1]) - chromakey) & 0xF;
   tmp |= (((int)(msg[local_idx]) - chromakey) & 0xF) << 4;
   
   // Output decrypted character
   msg_out[ (idx-1)/2 ] = tmp;
#endif

}
