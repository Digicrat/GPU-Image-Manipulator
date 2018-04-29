#define OPENCL_SRC_FILE "image.cl"

// Enable Exceptions (minimize need for explicit error handling)
#define CL_HPP_ENABLE_EXCEPTIONS 

#include "main.hpp"

#include <iostream>
#include <fstream>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

/*** Static OpenCL Utilities ***/
/* Error Code Lookup
 *  NOTE: If boost.compute is available we could use opencl_error::to_string()
 *   unfortunately that doesn't want to install neatly on this system, so:
 * Adapted From https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
 */
const char *getErrorString(cl_int error)
{
  switch(error){
    // run-time and JIT compiler errors
  case CL_SUCCESS: return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
 case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
 case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
 case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
 case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
 case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
 case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
 case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
 case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
 case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
 case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
 case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
 case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
 case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
 case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
 case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
 case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
 case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
 case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
 case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
 case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
 case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
 case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
 case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
 case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
 case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
 case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
 case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
 case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
 case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
 case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
 case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
 case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
 case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
 case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
 case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
 case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
 case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
 case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
 case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  default: return "CL_UNKNOWN_ERROR";
  }
}
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
      std::cerr << "ERROR: " <<  name << " (" << err << " = " << getErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}



/*** OpenCL Application ***/
class GIMD_OpenCL : virtual public GIMD_main
{
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  std::vector<cl::Device> all_devices;
 public:
  
  /** Constructor verifies device requirements and initializes hardware references */
  GIMD_OpenCL()
  {
    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
      throw std::runtime_error("No platforms found. Check OpenCL installation!");
      exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
      std::cout<<" No devices found. Check OpenCL installation!\n";
      exit(1);
    } 
    std::cout << "Found " << all_devices.size() << " OpenCL devices." << std::endl;
    
    // Ideally we would query if there is more than one device to find the most suitable
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    
    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    context = cl::Context({default_device});
    
    // create a queue (a queue of commands that the GPU will execute)
    queue = cl::CommandQueue(context, default_device, CL_QUEUE_PROFILING_ENABLE);

    
  }

  int run(int argc, char* argv[])
  {
    // TODO: Move into parent class

    cxxopts::Options options = init_options("OpenCL Test App");
    cxxopts::ParseResult result = parse_options(options, argc, argv);
    cout << "Arguments parsed" << endl;

    // TODO: Sanity checks

    load_program(OPENCL_SRC_FILE);
    
    do_action();
    
  }

  int convolve(uint32_t *input, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    uint32_t outHeight, outWidth;
    *output = run_img_kernel("convolve",
                             input, height*width, height, width,
                             NULL, NULL // TODO
                                      );
    return 1;
  }

  int mod_image(uint32_t *input, uint32_t **output, const uint32_t height, const uint32_t width )
  {
    return -1; // TODO
  }

  int img_sprite_anim(uint32_t *data, uint32_t **output,
                      uint32_t height, uint32_t width)
  {
    return -1; // TODO
  }
  int steg_image_en(uint32_t *data, uint32_t **output, uint32_t image_size)
  {
    return -1;
  }
  int steg_image_de(uint32_t *data, uint32_t **output,
                    const uint32_t image_size, const uint32_t data_height, const uint32_t data_width )
  {
    return -1;
  }

  void load_program(string fn)
  {
  cout << "Loading program from CL file " << fn << endl;
  ifstream sourceFile(fn);
  string sourceCode(
                    istreambuf_iterator<char>(sourceFile),
                    (istreambuf_iterator<char>()));
  cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length()+1));

  // Make program of the source code in the context
  program = cl::Program(context, source);

  // Build program for these specific devices
  if (program.build(all_devices) != CL_SUCCESS) {
    cout << "Error building: ";// << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices) << endl;
    exit(1);
  }
}
 
 // Test Kernel from github.  This always returns 0s for some reason.
uint32_t *run_img_kernel(string fcn,
uint32_t *inputSignal, uint32_t signal_size, uint32_t inputSignalHeight, uint32_t inputSignalWidth,
uint32_t *outHeight, uint32_t *outWidth)
{
  cout << "Running function " << fcn << endl;
  const size_t mask_size = sizeof(uint32_t) * mask_height * mask_width;
  const unsigned int outputSignalWidth  = inputSignalWidth-mask_width+1;
  const unsigned int outputSignalHeight = inputSignalHeight-mask_height+1;
  const size_t output_size = sizeof(uint32_t)*outputSignalHeight*outputSignalWidth;
  uint32_t *outputSignal = (uint32_t*)malloc(outputSignalWidth*outputSignalHeight*sizeof(uint32_t));

 if (outHeight != NULL) { *outHeight = outputSignalHeight; }
 if (outWidth != NULL) { *outWidth = outputSignalWidth; }
  
  // create buffers on device (allocate space on GPU)
  cl::Buffer buffer_input(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, signal_size, inputSignal);
  cl::Buffer buffer_mask(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, mask_size, mask);
  cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, output_size);

  // Prepare and run the kernel
  cl_int err;
  cl::Kernel kernel(program, fcn.c_str(), &err);
  checkErr(err, "kernel");
  
  cl::Event evt;
  checkErr(kernel.setArg(0, buffer_input),"setArg0");
  checkErr(kernel.setArg(1, buffer_mask),"setArg1");
  checkErr(kernel.setArg(2, buffer_output),"setArg2");
  checkErr(kernel.setArg(3, sizeof(cl_uint), &inputSignalWidth),"setArg3");
  checkErr(kernel.setArg(4, sizeof(cl_uint), &mask_width),"setArg4");
  checkErr( queue.enqueueNDRangeKernel(kernel,
              cl::NullRange, // Offset
              cl::NDRange(inputSignalWidth*inputSignalHeight), // Global Work size
              cl::NDRange(inputSignalWidth), // Local work size
              NULL, // Optional Events (?)
              &evt // Optional Event pointer
                                       ), "enqueueNDRangeKernel");

  // read result from GPU to here
  checkErr(queue.enqueueReadBuffer(buffer_output,
                          CL_TRUE, // Blocking Flag
                          0,
                          output_size,
                                     outputSignal), "enqueueReadBuffer");
#ifndef VERBOSE
  // Output the first line of input buffer
  std::cout << "Input Signal First Row:" << endl;
  for (int y = 0; y < inputSignalWidth; y++)
  {
      std::cout << inputSignal[y] << " ";
  }

  // Output the first line of output buffer
  std::cout << std::endl << std::endl << "Output Signal First Row:" << endl;;
  for (int y = 0; y < outputSignalWidth; y++)
  {
      std::cout << outputSignal[y] << " ";
  }
  std::cout << std::endl;
#endif
  
  std::cout << "Executed program succesfully with"
            << " input signal " << inputSignalHeight << " x " << inputSignalWidth
            << " output signal " << outputSignalHeight << " x " << outputSignalWidth
            << " in "
            << (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()
                - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
            << " ns." << std::endl;

return (uint32_t*)&outputSignal[0];
}

  
};

/*** Program Entry Point ***/
int main(int argc, char* argv[])
{
  // Declare app and initialize
  GIMD_OpenCL app;

  // Parse Arguments, Execute and run
  app.run(argc, argv);
}
