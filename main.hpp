#ifndef __MAIN_H__
#define __MAIN_H__

#include "cxxopts.hpp"
#include "image.hpp"
#include <string>
#include <iostream>

typedef enum ImageActions_t {
  // General Purpose Action Selectors (corresponding to functions below)
  MODE_OR_MASK = 0,
  MODE_AND_MASK = 1,
  MODE_FLIP_HOR,
  MODE_FLIP_VER,
  MODE_STEG_EN, // Steganographic Encryption of a Message
  MODE_STEG_DE, // Steganographic Decryption of a Message
  MODE_SPRITE_ANIM,
  MODE_ADD_RAND_NOISE,

  // More General Purpose Actions (TODO: Reorder list and/or remove NPP variants in favor of independent argument)
  MODE_CONVOLUTION,
  MODE_UNSPECIFIED
} ImageActions_t;

class GIMD_main
{
protected:
  int verbose = 0, n = 1;
  int /*ImageActions_t*/ mode = MODE_CONVOLUTION;
  std::string mask_raw = "0 0 0; 0 1 0; 0 0 0";
  int32_t *mask = NULL;
  int mask_height = 3;
  int mask_width = 3;
  std::string input_file = "Lenna.ppm";
  std::string output_file = "Lenna.convolved.ppm";
  MemMode_t memMode = MEM_HOST_PAGEABLE;
  std::string sprite_fn;
  std::string extra;
  uint32_t host_chromakey = 0x11111111;
  
public:
  ~GIMD_main()
  {
    if (mask != NULL)
    {
      //free(mask);
    }
  }
  
  cxxopts::ParseResult parse_options(cxxopts::Options options, int argc, char* argv[])
  {
    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

    // Parse the kernel/mask
    mask = (int32_t*)malloc(sizeof(uint32_t)*mask_width*mask_height);
    std::vector<std::string> elems = split(mask_raw);

    if (elems.size() != mask_height*mask_width) {
      throw std::runtime_error("ERROR: Invalid kernel defined\n");
    } else {
      std::cout << "Convolution Kernel "
                << mask_height << " x " << mask_width << std::endl;
      for(int i = 0; i < (mask_height*mask_width); i++) {
        mask[i] = atoi(elems[i].c_str());
        std::cout << mask[i] << " ";
        if ((i+1) % mask_width == 0) {
          std::cout << std::endl;
        }
      }

      std::cout << std::endl;
    }

    if (input_file.empty()) {
      throw std::runtime_error("Input filename (-i) required (PPM format)");
    }
    if (output_file.empty()) {
      throw std::runtime_error("Output filename (-o) required (PPM format)");
    }

    return result;
  }
  
  cxxopts::Options init_options(std::string appname)
  {
    cxxopts::Options options(appname);

    std::string help_modes =
     "\t          0  Image Mask - Bitwise or each pixel value with provided key\n"
     "\t          1  Image Mask - Bitwise and each pixel value with provided key\n"
     "\t          2  Image Flip Horizontal*\n"
     "\t          3  Image Flip Vertical*\n"
     "\t          4  Steganographic Encryption. Specify ASCII message with '-M'\n"
     "\t                 Optionally use key (-k) as a cipher\n"
     "\t          5  Stegonographic Decryption. ASCII message will be output to STDOUT\n"
     "\t          6  Sprite Animation.\n"
     "\t          7  Add Random noise (via curand) to image.\n"
     "\t          8  NPP Library based AND operation. Equivalent to mode 0\n"
     "\t          9  NPP Library based OR operation. Equivalent to mode 1\n"
     "\t                 Specify the same key as when encoded.\n"
     "\t                 Input image (-i) should be the unaltered image.\n"
     "\t                 Output image (-o) should be the previously altered image.\n"
     "\t         10  GPU-Based 'Naive' Convolution using defined kerne.\n"
      ;

    
    options.add_options()
      ("h,kh", "Convolution Kernel Height", cxxopts::value<int>(mask_height))
      ("w,kw", "Convolution Kernel Width", cxxopts::value<int>(mask_width))
      ("kernel", "Convolution Kernel of defined dimensions (only square kernels are currently supported), ie: \""+mask_raw+"\"", cxxopts::value<std::string>(mask_raw))
      ("k,chromakey", "Specify key value (ie: chromakey mask or cipher value)", cxxopts::value<uint32_t>(host_chromakey))
      //      ("M,message", "ASCII Message to encode in steganographic mode", cxxopts::value<std::string>(msg))
      ("v,verbose", "Toggle verbosity (0 off, 1 on, other values reserved)", cxxopts::value<int>(verbose))
      ("i,input", "Input filename (PPM format)", cxxopts::value<std::string>(input_file))
      ("o,output", "Output filename (PPM format)", cxxopts::value<std::string>(output_file))
      ("m,mode", "Specify action to be performed.  Currently supported modes are:\n"+help_modes, cxxopts::value<int>(mode))
      ("n", "Number of repetitions", cxxopts::value<int>(n))
      ("M,msg,sprite", "Extra String Argument; Steganographic encoding message or sprite filename", cxxopts::value<std::string>(extra))
      ;

    return options;
  }

  void do_action()
  {
    uint32_t *image = NULL;
    uint32_t *output = NULL;
    uint32_t height, width, image_size, outHeight, outWidth;
    int status;

    load_image(input_file.c_str(), &image, &image_size, &height, &width, memMode);

    if(image == NULL)
    {
      throw std::runtime_error("Unable to load input file");
    }

    try {
      switch(mode) {
      case MODE_OR_MASK:
      case MODE_AND_MASK:
      case MODE_FLIP_HOR:
      case MODE_FLIP_VER:
      case MODE_ADD_RAND_NOISE:
        status = mod_image(image, &output, height, width );
        break;
      case MODE_SPRITE_ANIM:
        status = img_sprite_anim(image, &output, height, width);
        break;
      case MODE_STEG_DE:
        status = steg_image_de(image, &output, image_size, height, width);
        break;
      case MODE_STEG_EN:
        status = steg_image_en(image, &output, image_size);
        break;
      case MODE_CONVOLUTION:
        status = convolve(image, &output, height, width );
      default:
        throw std::runtime_error("Invalid or unsupported mode ("+std::to_string(mode)+") specified.");
      }
      
      // All action functions will return -1 on error, 0 on success, 1 on success with image data to be output.
      if( status > 0) {
        if (output == NULL) {
          write_image(output_file.c_str(), image, width, height); // TODO/FIXME: Support arbitrary output size
        } else {
          write_image(output_file.c_str(), output, width, height); // TODO/FIXME: Support arbitrary output size
          free_image(output, memMode);
        }
      } else {
        throw std::runtime_error("ERROR: Kernel Failed");
      }
    } catch( const std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }

    // Cleanup
    free_image(image, memMode);
    
  }
  

  /*** Virtual Action Functions 
       All of the following functions return:
         0 on success. No farther action needed.
         -1 on error.
         1 on success. New image data available for output.
  ***/
  /** Simple (Naive) Convolution Example 
   * Based on https://github.com/bgaster/opencl-book-samples/blob/master/src/Chapter_3/OpenCLConvolution/Convolution.cl
   *
   * This simple implementation is not optimized (and therefore more readable).
   *
   * This simple implementation requires image and mask to each be square. Image must be small enough to run
   *   one image row per GPU block.
   **/
  virtual int convolve(uint32_t *input, uint32_t **output, const uint32_t height, const uint32_t width )=0  ;

  /** Dispatch function for all actions taking image input and outputting one of equal size */
  virtual int mod_image(uint32_t *input, uint32_t **output, const uint32_t height, const uint32_t width )=0  ;

/* Simple sprite animation.  The given sprite image will be overlaid
 *  on the base image. The first image will start at position 0,0, with
 *  the starting column incremented for each pass.  One image will be generated for each width-1 pixels.
 * The resulting images can be converted into an animated gif using the convert tool:
 *  convert -delay 20 -loop 0 out_fn* out_fn.gif
 *  WARNING: This conversion process can be slow. Ideally, this could be sped up
 *   by utilizing CUDA to directly convert files into GIF format and letting the
 *   CPU assemble the results into an animated GIF ... but one step at a time.
 */
  virtual int img_sprite_anim(uint32_t *input, uint32_t **output,
                              const uint32_t height, const uint32_t width)=0;

  // Simple steganographic message en/decryption
  virtual int steg_image_en(uint32_t *input, uint32_t **output,
                              const uint32_t image_size)=0;
  virtual int steg_image_de(uint32_t *input, uint32_t **output,
                            const uint32_t image_size, const uint32_t height, const uint32_t width)=0;

};


#endif
