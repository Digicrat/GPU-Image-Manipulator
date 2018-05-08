#define OPENCL_SRC_FILE "image.cl"

// Enable Exceptions (minimize need for explicit error handling)
#define CL_HPP_ENABLE_EXCEPTIONS 

#include <iostream>
#include <fstream>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

#include "main.hpp"

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
      throw std::runtime_error("OpenCL ERROR Thrown");
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
    *output = run_img_kernel("convolve",
                             input, height*width, height, width,
                             NULL, NULL // TODO
                                      );
    return 1;
  }

  int mod_image(uint32_t *input, uint32_t **output, const uint32_t height, const uint32_t width )
  {
     string fcn;
     size_t imageSize = width*height*sizeof(uint32_t);
     cl::NDRange global_work(width,height);
     cl::NDRange local_work = cl::NullRange;
     
      // create buffers on device (allocate space on GPU)
      cl::Buffer buffer_input(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              imageSize,
                              input);
      cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, imageSize);

      // Select function
      switch(mode)
      {
      case MODE_AND_MASK:
         fcn = "do_and_mask";
         // Use simple global work size of width*height
         global_work = cl::NDRange(height*width);
         break;
      case MODE_OR_MASK:
         fcn = "do_or_mask";
         // Use default work size of width,height
         break;
      case MODE_FLIP_HOR:
         fcn = "do_flip_hor";
         break;
      case MODE_FLIP_VER:
         fcn = "do_flip_ver";
         break;
      default:
         printf("ERROR: Unsupported mode\n");
         return -1;
      }
      
      // Prepare selected kernel
      cl_int err;
      cl::Kernel kernel(program, fcn.c_str(), &err);
      checkErr(err, "kernel");

      // Set Arguments
      checkErr(kernel.setArg(0, buffer_input),"setArg0");
      checkErr(kernel.setArg(1, buffer_output),"setArg1");
      checkErr(kernel.setArg(2, host_chromakey),"setArg2");

      // Define event (used here for timing metrics)
      cl::Event evt;

      // Enqueue Kernel
      checkErr( queue.enqueueNDRangeKernel(kernel,
                                           cl::NullRange, // Offset
                                           global_work, // Global Work size
                                           local_work, // Local work size
              NULL, // Optional Events
              &evt // Optional Event pointer
                   ), "enqueueNDRangeKernel");

      // read result from GPU
      checkErr(queue.enqueueReadBuffer(buffer_output,
                                       CL_TRUE, // Blocking Flag
                                       0,
                                       imageSize,
                                       input), "enqueueReadBuffer");
      std::cout << "Executed program succesfully in "
                << (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()
                    - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                << " ns." << std::endl;
    return 1;
  }

   /** This application demonstrates enqueing of multiple commands with
         event handling to generate an series of images that can later
         be merged into an animated gif.
    */
  int img_sprite_anim(uint32_t *input, uint32_t **output,
                      uint32_t height, uint32_t width)
  {
     int status;
     uint32_t *sprite_data = NULL;
     uint32_t sprite_length=0, sprite_height, sprite_width;
     int imageSize = height*width*sizeof(uint32_t);

     // Load sprite
     status = load_image(extra.c_str(),
                        &sprite_data, &sprite_length, &sprite_height, &sprite_width,
                        MEM_HOST_PAGEABLE);
     if (status < 0) {
        printf("ERROR: Unable to load sprite\n");
        return -1;
     }

     // Calculate number of images to generate (make it even for consistency)
     int num_images = n;
     if (num_images <= 1)
     {
        num_images = width - (width&1);
     }
     else
     {
        num_images -= num_images&1;
     }

     
     // create buffers on device (allocate space on GPU)
     cl::Buffer buffer_input(context,
                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             imageSize,
                             input);
     cl::Buffer buffer_sprite(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sprite_height*sprite_width*sizeof(uint32_t),
                              sprite_data);
     
     // Prepare the kernel
     cl_int err;
     cl::Kernel kernel(program, "overlay", &err);
     checkErr(err, "kernel");
     checkErr(kernel.setArg(0, buffer_input),"setArg0");
     checkErr(kernel.setArg(1, buffer_sprite),"setArg1");
     checkErr(kernel.setArg(2, sprite_height),"setArg2");
     checkErr(kernel.setArg(3, sprite_width),"setArg3");
     checkErr(kernel.setArg(4, host_chromakey), "setArg4");

     /** Create an alternate command queue with the OUT_OF_ORDER property
      *   to allow potentially asynchronous execution.
      */
     cl::CommandQueue aqueue = cl::CommandQueue(
        context,
        all_devices[0],
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE
        );
     
     // Prepare num_images output buffers and events, and enqueue
     std::vector<cl::Event> eventList;
     std::vector<cl::Event> timingEvents;
     std::vector<cl::Buffer> bufList;
     std::vector<uint32_t*> localBufList;
     for(int i = 0; i < num_images; i++)
     {
        cl::Buffer buf(context,
                       CL_MEM_WRITE_ONLY,
                       imageSize);
        cl::Event timingEvt, evt;
        checkErr(kernel.setArg(5, i),"setArg5");
        checkErr(kernel.setArg(6, buf),"setArg6");

        checkErr( aqueue.enqueueNDRangeKernel(
                     kernel,
                     cl::NullRange, // Offset
                     cl::NDRange(height,width),
                     cl::NullRange,
                     NULL, // Optional Event List
                     &timingEvt // Optional Event pointer
                     ), "enqueueNDRangeKernel");

        // read result from GPU.  Read will be non-blocking with an event to signal completion
        uint32_t *outBuf = (uint32_t*)malloc(imageSize);
        std::vector<cl::Event> events;
        events.push_back(timingEvt);
        checkErr(aqueue.enqueueReadBuffer(
                    buf,
                    CL_FALSE, // Blocking Flag
                    0, // offset
                    imageSize,
                    outBuf,
                    &events,
                    &evt
                    ), "enqueueReadBuffer");

        
        // Add to vector
        eventList.push_back(evt);
        bufList.push_back(buf);
        localBufList.push_back(outBuf);
        timingEvents.push_back(timingEvt);
     }

     for( int i = 0; i < num_images; i++)
     {
        cl::Event evt = eventList[i];
        cl::Event timingEvt = timingEvents[i];
        cl::Buffer buf = bufList[i];
        uint32_t *out = localBufList[i];
        char out_fn[64];

        // Wait on event
        evt.wait();

        // Write Image
        sprintf(out_fn, "%s[%04d].ppm", output_file.c_str(), i);
        write_image(out_fn, out, width, height);

        // Log timing metrics
        // TODO: Use common fn to log to file
        printf("Image %d took %f ns\n",
               i,
               (timingEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>()
                - timingEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
           );
        
        // Free resources
        free_image(out, memMode);
     }
     
     free_image(sprite_data, memMode);
     return 0;
  }
   
  int steg_image_en(uint32_t *data, uint32_t **output, uint32_t imageSize)
  {
    int msgLen = extra.length();
    
    // create buffers on device (allocate space on GPU)
    cl::Buffer buffer(context,
                            CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                            imageSize,
                            data);
    cl::Buffer buffer_msg(context,
                          CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                          extra.length(),
                          const_cast<char*>(extra.c_str())
       );

    // Prepare selected kernel
    cl_int err;
    cl::Kernel kernel(program, "do_steg_en", &err);
    checkErr(err, "kernel");
      
    // Set Arguments
    checkErr(kernel.setArg(0, buffer),"setArg0");
    checkErr(kernel.setArg(1, buffer_msg),"setArg1");
    checkErr(kernel.setArg(2, msgLen),"setArg2");
    checkErr(kernel.setArg(3, host_chromakey),"setArg3");

    // Define event (used here for timing metrics)
    cl::Event evt;
    
    // Enqueue Kernel
    checkErr( queue.enqueueNDRangeKernel(kernel,
                                         cl::NullRange,
                                         cl::NDRange(imageSize/sizeof(uint32_t)),
                                         cl::NullRange,
                                         NULL,
                                         &evt
                 ), "enqueueNDRangeKernel");
    
    // read result from GPU
    checkErr(queue.enqueueReadBuffer(buffer,
                                     CL_TRUE, // Blocking Flag
                                     0,
                                     imageSize,
                                     data), "enqueueReadBuffer");
    std::cout << "Executed program succesfully in "
              << (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()
                  - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
              << " ns." << std::endl;
    return 1;
  }
  int steg_image_de(uint32_t *input, uint32_t **output,
                    const uint32_t imageSize, const uint32_t height, const uint32_t width )
  {
     int status = 0;
     uint32_t *enc_data = NULL;
     uint32_t enc_length=0, enc_height, enc_width;

     // Input file is original (base) image.  Copy it into a buffer.
     cl::Buffer buffer_input(context,
                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             imageSize,
                             input);
     
     // "Output" file is the encoded image.  Load it and copy it into a buffer
     status = load_image(output_file.c_str(),
                         &enc_data, &enc_length, &enc_height, &enc_width,
                         MEM_HOST_PAGEABLE);
     if (status < 0) {
        printf("ERROR: Unable to load sprite\n");
        return -1;
     }
     else if (height != enc_height || width != enc_width)
     {
        printf("ERROR: Encoded and original image must be of the same dimensions\n");
        free(enc_data);
        return -1;
     }

     // Message decoding buffer
     size_t msg_len = height*width/2;
     char *msg = (char*)malloc(msg_len);
     cl::Buffer buffer_msg(context,
                           CL_MEM_WRITE_ONLY,
                           msg_len);
     
     try
     {
        // Encoded Image Buffer
        cl::Buffer buffer_enc(context,
                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             imageSize,
                             enc_data);
        
        // Prepare kernel
        cl_int err;
        cl::Kernel kernel(program, "do_steg_de", &err);
        checkErr(err, "kernel");

        // Set Arguments
        checkErr(kernel.setArg(0, buffer_input),"setArg0");
        checkErr(kernel.setArg(1, buffer_enc),"setArg1");
        checkErr(kernel.setArg(2, buffer_msg),"setArg2");
        checkErr(kernel.setArg(3, host_chromakey),"setArg3"); // Cipher key

        // Enqueue Kernel
        cl::Event evt;
        checkErr( queue.enqueueNDRangeKernel(kernel,
                                         cl::NullRange, // Offset
                                             cl::NDRange(width,height), // Global Work Size
                                         cl::NullRange, // Local Work Size
                                         NULL, // Optional Events
                                         &evt // Optional Event pointer
                 ), "enqueueNDRangeKernel");
    
    // read result from GPU
    checkErr(queue.enqueueReadBuffer(buffer_msg,
                                     CL_TRUE, // Blocking Flag
                                     0,
                                     msg_len,
                                     msg), "enqueueReadBuffer");

    std::cout << "Executed program succesfully in "
              << (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()
                  - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
              << " ns." << std::endl;
    printf("Decoded message reads: %s\n", msg);


     }
     catch( std::runtime_error e )
     {
        printf("ERROR During Steganographic Decoding\n");
        status = -1;
     }
        
     free_image(enc_data, memMode);
     return status; // No image to write back out
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
        cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(all_devices[0]) << endl;
        exit(1);
     }
  }
 
   uint32_t *run_img_kernel(string fcn,
                            uint32_t *inputSignal, uint32_t signal_size,
                            uint32_t inputSignalHeight, uint32_t inputSignalWidth,
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
