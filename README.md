# GPU Image Manipulation Demonstrator (GIMD)

Not to be confused with GIMP, GIMD is an assorted collection of image manipulation demos.  Functionality is in the spirit of a lightweight ImageMagick, but with the primary goal to serve as a complete example and introduction to GPU programming.

The goal is to provide functional image manipulation tools that demonstrate a variety of GPU programming options, with comparitive timing metrics when available for CUDA, OpenCL, and native CPU code.

## Usage
See the Makefile "modX" targets for examples, or run the executables with "--help"

## File Description

- *.png, *.jpg - Example input images (TODO: Document where images were downloaded from)
- Makefile - Master build configuration.  "make" will build all binaries while "make modX" will execute dependencies corresponding to the appropriate module.
-- TODO: List supported modules here, with descriptions
- image.cu - CUDA Application Methods
- imagel.cl - OpenCL Sources
- image.hpp, image.cpp - Common CPU Library Functions
- cxxopts.hpp - Library for command line argument parsing from https://github.com/jarro2783/cxxopts
- main.hpp - Prototypes and settings for main application functions (independent of target)


# References & Resources
This application is a class project for EN605.417, with related source code at https://github.com/JHU-EP-Intro2GPU/EN605.417.FA

Command Line Argument Parsing from https://github.com/jarro2783/cxxopts
  # TODO: Convert this into a submodule

Simple Convolution Examples from https://github.com/bgaster/opencl-book-samples/blob/master/src/Chapter_3/OpenCLCo

PPM Image Format - http://netpbm.sourceforge.net/doc/ppm.html

C++ RegEx String Split Example/reference from https://stackoverflow.com/questions/9435385/split-a-string-using-c11

OpenCL:
- https://khronos.org
- TI OpenCL Documentation, includes good overviews on OpenCL in general: http://downloads.ti.com/mctools/esd/docs/opencl/intro.html
-- For example: http://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html
- C++ API: http://github.khronos.org/OpenCL-CLHPP/index.html
- Examples: https://github.com/Dakkers/OpenCL-examples/blob/master
- More examples: https://www.eriksmistad.no/using-the-cpp-bindings-for-opencl/
- Community Tips: https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor

CUDA:
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA GDB Documentation: https://docs.nvidia.com/cuda/cuda-gdb/index.html
- NPP Library: http://docs.nvidia.com/cuda/npp/index.html
- Thrust Library: https://docs.nvidia.com/cuda/thrust/index.html
- NvGraph Library: https://docs.nvidia.com/cuda/nvgraph/index.html
- CuRand Library: https://docs.nvidia.com/cuda/curand/host-api-overview.html
- CuFFT Library: https://docs.nvidia.com/cuda/cufft/index.html
- Blog Entries:
- - Pageable Memory - https://devtalk.nvidia.com/default/topic/402801/pageable-and-non-pageable-memory/
-- Unified Memory: https://devblogs.nvidia.com/unified-memory-cuda-beginners/
-- Streams: https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
-- Cuda C++11 Features: https://devblogs.nvidia.com/power-cpp11-cuda-7/

# System Setup Notes
To use CUDA, one must of course have a compatible graphics card.

Before succumbing to the inevitable and switching to a less powerful GPU on my Linux machine, I started developing under Windows 10 using the command line interface beneath the Ubuntu Bash shell.  This environment works and has some benefits, but also tends to be buggier than running directly under Linux.  Note that for this configuration to work, one must install MS Visual Studio in addition to the nVidia CUDA Toolkit.

## Ubuntu Quick Start Guide
- sudo apt-get update && sudo apt-get upgrade
- Install latest driver from http://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/XFree86/Linux-x86_64/390.25/NVIDIA-Linux-x86_64-390.25.run&lang=us&type=TITAN
-- sudo service lightdm stop
--- If running XFCE.  Substitute for KDE or Gnome as appropriate. This is necessary to run the installer.
-- sudo ./NVIDIA-Linux-x86_64-xxx.xx.run
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal
- Install OpenCL Headers (Optional: You may also manually specify the correct include path)
-- sudo apt-get install opencl-headers

