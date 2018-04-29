#include "cxxopts.hpp"
#include "main.hpp"
#include <iostream>

using namespace std;

/*** OpenCL Application ***/
class GIMD_cpu : virtual public GIMD_main
{
public:
  int run(int argc, char* argv[])
  {
    // TODO: Move into parent class

    cxxopts::Options options = init_options("OpenCL Test App");
    cxxopts::ParseResult result = parse_options(options, argc, argv);
    cout << "Arguments parsed" << endl;

    do_action();
    
    return 0;
  }

  int convolve(uint32_t *input, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    *output = (uint32_t*)malloc(height*width);
    for(int x = 0; x < width; x++) {
      for(int y = 0; y < height; y++) {
        do_convolve(input, *output, height, width, x, y);
      }
    }
    
    return 1;
  }
void do_convolve(
    const uint * const input,
    uint * const output,
    const uint32_t height, const uint32_t width,
    const int x,
    const int y
                 )
{

    uint sum = 0;
    for (int r = 0; r < mask_width; r++)
    {
      int idxIntmp = (y + r) * width + x;

        for (int c = 0; c < mask_width; c++)
        {
         
			sum += mask[(r * mask_width)  + c] * input[idxIntmp + c];
        }
    } 
    
    output[y * width + x] = sum;
}

  int mod_image_flip_row(uint32_t *data, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    uint32_t row[width];
    for(int y = 0; y < height; y++) {
      uint32_t base = y*width;
      for(int x = 0; x < width; x++) {
        row[x] = data[base+x];
      }
      for(int x = 0; x < width; x++) {
        data[base + width-1 - x] = row[x];
      }
    }
  }
  int mod_image_flip_col(uint32_t *data, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    uint32_t col[height];
    for(int x = 0; x < width; x++) {
      for(int y = 0; y < height; y++) {
        col[y] = data[ y*width + x ];
      }
      for(int y = 0; y < height; y++) {
        data[ y*width + x ] = col[height-y-1];
      }
    }
  }
  
  int mod_image(uint32_t *data, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    switch(mode) {
    case MODE_ADD_RAND_NOISE:
      srand(time(NULL));
      break;
    case MODE_FLIP_HOR:
      return mod_image_flip_row(data, output, height, width);
      break;
    case MODE_FLIP_VER:
      return mod_image_flip_col(data, output, height, width);
      break;
    }
    for(int x = 0; x < width; x++) {
      for(int y = 0; y < height; y++) {
        uint32_t idx = y*width + x;
        switch(mode) {
        case MODE_ADD_RAND_NOISE:
          {
          uint32_t noise = rand();
          char ch_noise = GET_R(noise) % GET_R(host_chromakey);
          noise = SET_R(noise,ch_noise);
          ch_noise = GET_G(noise) % GET_G(host_chromakey);
          noise = SET_G(noise,ch_noise);
          ch_noise = GET_B(noise) % GET_B(host_chromakey);
          noise = SET_B(noise,ch_noise);

          data[idx] += noise;
          }
          break;
        case MODE_OR_MASK:
          data[idx] = data[idx] | host_chromakey;
          break;
        case MODE_AND_MASK:
          data[idx] = data[idx] & host_chromakey;
          break;
        default:
          printf("ERROR: Invalid mode (%d) passed to mod_image function\n", mode);
        }
      }
    }
    return 1;
  }

  int img_sprite_anim(uint32_t *data, uint32_t **output,
                    uint32_t height, uint32_t width)
  {
    if (extra.empty()) {
      throw std::runtime_error("Sprite file must be defined for this operation");
    }
    
    uint32_t image_size = width*height;
    char out_fn[64];
    uint32_t num_images = 0;
    uint32_t *sprite_data = NULL;
    uint32_t sprite_length=0, sprite_height, sprite_width;
    int status;
    uint32_t out_data[height*width];
  
    // Load sprite image
    status = load_image(extra.c_str(),
                        &sprite_data, &sprite_length, &sprite_height, &sprite_width,
                        MEM_HOST_PAGEABLE);
    if (status < 0) {
      printf("ERROR: Unable to load sprite\n");
      return -1;
    }
    // Calculate number of images to generate (must be even to simplify logic)
    num_images = width - (width&1);

    for(int i = 0; i < num_images; i++)
    {
      for(int x = 0; x < width; x++) {
        for(int y = 0; y < height; y++) {
          uint32_t idx = y*height + x;

          for(int sx = 0; sx < sprite_width; sx++) {
            for(int sy = 0; sy < sprite_height; sy++) {
              uint32_t sprite_idx = sy*sprite_height + sx;
              if (sx < 0 || sy < 0
                  || sy > sprite_width || sx > sprite_height
                  || sprite_data[sprite_idx] == host_chromakey) {
                out_data[idx] = data[idx];
              } else {
                out_data[idx] = sprite_data[sprite_idx];
              }
            }
          }
        }
      }
      // Write buffer to disk
      sprintf(out_fn, "%s[%04d].ppm", output_file, i);
      write_image(out_fn, out_data, width, height);

    }
    free_image(sprite_data, memMode);

    return 0;
  }
  int steg_image_en(uint32_t *data, uint32_t **output, uint32_t image_size)
  {
    
    int msgLen = extra.length();
    const char* msg = extra.c_str();
    
    if (msgLen*2 > image_size) {
      printf("ERROR: Message is too long to be encoded in this image\n");
      return -1;
    }
    for(int idx = 0; idx < msgLen*2; idx++) {
      uint32_t tmp = data[idx];
      uint8_t halfByte;
      if (idx & 0x1 == 1) {
        // Odd threads take the upper half-byte
        halfByte = (msg[(idx-1)/2]) >> 4;
      } else {
        // Even threads take the lower half-byte
        halfByte = msg[idx/2];
      }
      
      // Add cypher
      halfByte += host_chromakey;

      // Bit-1 of char to Bit-1 of R
      tmp = tmp ^ (halfByte & 0x1);
      
      // Bit-2 of char to Bit-1 of G
      tmp = tmp ^ ((halfByte & 0x2) << 7);
      
      // Bit-3+4 of char to Bit-1+2 of B
      tmp = tmp ^ ((halfByte & 0xC) << 14);

      data[idx] = tmp;
      
    }
    
    return 1;
  }
  int steg_image_de(uint32_t *data, uint32_t **output,
                    const uint32_t image_size, const uint32_t data_height, const uint32_t data_width )
  {
    std::string msg = "";
    uint8_t tmp2 = 0;
    uint8_t oddByte = 0;
    uint32_t data2_length=0, height, width;
    uint32_t *data2 = NULL; // Start: Encoded image.  End: Decoded message

    // Load Encoded Image
    int status = load_image(output_file.c_str(), &data2, &data2_length, &height, &width, memMode);
    if (status < 0) {
      printf("ERROR: Unable to load encoded image\n");
      return -1;
    }

    // Validate that lengths match
    if (data2_length != image_size) {
      printf("ERROR: Encoded and source images must be of the same size!\n");
      return -1;
    }

    
    for(int x = 0; x < width; x++) {
      for(int y = 0; y < height; y++) {
        uint32_t idx = y*height + x;
        uint32_t tmp = data2[idx] ^ data[idx];
        
        // Calculate the half-word
        tmp2 = GET_R(tmp) & 0x1; // Bit 1
        tmp2 |= (GET_G(tmp) & 0x1) << 1; // Bit 2
        tmp2 |= (GET_B(tmp) & 0x3) << 2; // Bits 3+4

        if (idx & 0x1 == 1) {
          tmp = (oddByte-host_chromakey) & 0xF;
          tmp |= (tmp2-host_chromakey) & 0xF;
          msg += tmp;
        } else {
          oddByte = tmp2;
        }
      }
    }
    
    *output = data2;
    return 1;
  }

};

/*** Program Entry Point ***/
int main(int argc, char* argv[])
{
  // Declare app and initialize
  GIMD_cpu app;

  // Parse Arguments, Execute and run
  return app.run(argc, argv);

}
