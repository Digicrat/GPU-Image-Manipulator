
#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <stdint.h>
  
typedef enum MemMode_t {
  MEM_HOST_PAGEABLE,
  MEM_HOST_PINNED
} MemMode_t;

/* Image Generation Configuration
 *  We'll use a simple RGB color scheme
 *  This can be extended to other schemes (ie: sRGB, IAB) if needed later
 */
// 10-bit color space -- good in theory, but harder to display in a useful format
// Note: We still reserve 10-bits per channel, but only use 8 when outputting
#define RGB_COMPONENT_COLOR 255
#define MAX_COLOR RGB_COMPONENT_COLOR
#define R_SHIFT 0
#define G_SHIFT 8
#define B_SHIFT 16
#define R_MASK 0x000000FF
#define G_MASK 0x0000FF00
#define B_MASK 0x00FF0000
#define S_MASK 0xFF000000 // reserved (ie: alpha channel)

#define SET_R(data,value) ((value&0xFF)|(data&~R_MASK))
#define SET_G(data,value) (((value&0xFF)<<G_SHIFT)|(data&~G_MASK))
#define SET_B(data,value) (((value&0xFF)<<B_SHIFT)|(data&~B_MASK))

#define GET_R(data) (data & R_MASK)
#define GET_G(data) ((data & G_MASK) >> G_SHIFT)
#define GET_B(data) ((data & B_MASK) >> B_SHIFT)

#define GET_Rxy(buf,x,y) (GET_R(buf[y*width+x]))
#define GET_Gxy(buf,x,y) (GET_G(buf[y*width+x]))
#define GET_Bxy(buf,x,y) (GET_B(buf[y*width+x]))


int load_image(const char* fn, uint32_t **buff, uint32_t *buf_length, uint32_t *height, uint32_t *width, MemMode_t mem_mode);
void write_image(const char *fn, const uint32_t *buf, const unsigned int width, const unsigned int height);

#include <regex>
/** String split utility1                                                                                                     
 * From https://stackoverflow.com/questions/9435385/split-a-string-using-c11                                                  
 * NOTE: Requires C++11 and GCC >= 4.9                                                                                        
 */
inline std::vector<std::string> split(const std::string& input /*, const string& regex*/) {
  const std::string regex = "[\\s,;]+";
  // passing -1 as the submatch index parameter performs splitting
  std::regex re(regex);
  std::sregex_token_iterator
    first{input.begin(), input.end(), re, -1},
    last;
    return {first, last};
}


#endif
