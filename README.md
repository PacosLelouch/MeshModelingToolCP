# AA Shape-Up - A Maya Plug-in Development Tool

A Maya mesh modeling tool with constraint projection. (CIS-660 Authoring Tool Project)



## How to Build

### Requirement

- Maya (>= 2020, for 2020: https://www.autodesk.com/developer-network/platform-technologies/maya)
- Visual Studio 2019
- Maya Environment (for 2020: http://help.autodesk.com/view/MAYAUL/2020/ENU/?guid=__developer_Maya_SDK_MERGED_Setting_up_your_build_Windows_environment_64_bit_html)
- CUDA (optional, >= 11, with CUDA_PATH system variable)



### Build

1. Build project. `_CPP` for CPU-only version. `_CUDA` for CUDA enabled version.



## For Developers

### How to Add Source Files

You can just add your non-CUDA source file to `_CPP` project, and only add your CUDA source file to `_CUDA` project. For some functions that need to define a non-CUDA version, you need to place them in source files in `CPPOnly` directory. 
