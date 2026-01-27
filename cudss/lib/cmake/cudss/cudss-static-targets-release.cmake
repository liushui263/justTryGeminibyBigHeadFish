#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cudss_static" for configuration "Release"
set_property(TARGET cudss_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cudss_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${cudss_LIBRARY_DIR}/libcudss_static.a"
  )

list(APPEND _cmake_import_check_targets cudss_static )
list(APPEND _cmake_import_check_files_for_cudss_static "${cudss_LIBRARY_DIR}/libcudss_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
