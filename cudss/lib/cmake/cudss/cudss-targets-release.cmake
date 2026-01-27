#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cudss" for configuration "Release"
set_property(TARGET cudss APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cudss PROPERTIES
  IMPORTED_LOCATION_RELEASE "${cudss_LIBRARY_DIR}/libcudss.so.0.7.1"
  IMPORTED_SONAME_RELEASE "libcudss.so.0"
  )

list(APPEND _cmake_import_check_targets cudss )
list(APPEND _cmake_import_check_files_for_cudss "${cudss_LIBRARY_DIR}/libcudss.so.0.7.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
