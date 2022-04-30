# AA Shape-Up - A Maya Plug-in Development Tool

A Maya mesh modeling tool with constraint projection. (CIS-660 Authoring Tool Project)



## How to Build

### Requirement

- Maya (>= 2020, for 2020: https://www.autodesk.com/developer-network/platform-technologies/maya)
- Visual Studio 2019
- Maya Environment (for 2020: http://help.autodesk.com/view/MAYAUL/2020/ENU/?guid=__developer_Maya_SDK_MERGED_Setting_up_your_build_Windows_environment_64_bit_html)
- CUDA (optional, >= 11, with CUDA_PATH system variable)



### Something to Check

1. In Maya, input command `about -api` in MEL. Update Maya if the result is smaller than `20200400`.



### Build

1. Build project. `AAShapeUp_Maya_CPP` for CPU-only version. `AAShapeUp_Maya_CUDA` for CUDA enabled version.
   `AAShapeUp_Viewr_CPP` for standalone OpenGL viewer.
1. Copy all the `.mel` files to the `../scripts/` directory related to your custom Maya plugin directory (`plug-ins/`).



## For Developers

### Project Dependencies

1. `AAShapeUp_Maya_CPP`->`AAShapeUp_Extension_NoCUDAExport`->`AAShapeUp_Core_CPP`
2. `AAShapeUp_Maya_CUDA`->`AAShapeUp_Extension_CUDA`->`AAShapeUp_Core_CPP`
3. `AAShapeUp_Viewer_CPP`->`AAShapeUp_Extension_NoCUDAExport`->`AAShapeUp_Core_CPP`



### Attention

1. Copy the reference in `AAShapeUp_Maya_CUDA` project if you create new files in `AAShapeUp_Maya_CPP`.



