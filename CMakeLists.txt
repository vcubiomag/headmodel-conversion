cmake_minimum_required(VERSION 3.21)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Source-tree config so CGAL's own install() rules don't leak headers/docs into the wheel.
find_package(CGAL REQUIRED
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/external/cgal" NO_DEFAULT_PATH)

# OCCT (frozen at V8_0_0_p1) is consumed via add_subdirectory. EXCLUDE_FROM_ALL both ignores OCCT's
# install() rules (no leakage into the wheel) and builds toolkits only on demand. Options must be
# forced before add_subdirectory for OCCT's option() calls to see them. PIC is required so the
# static libs link into the shared extension on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_LIBRARY_TYPE "Static" CACHE STRING "" FORCE)

# Disabled so a box without Tcl/OpenGL can still configure; toolkits TKDESTEP pulls are configured
# on demand regardless.
foreach(_m Draw Visualization ApplicationFramework DETools)
  set(BUILD_MODULE_${_m} OFF CACHE BOOL "" FORCE)
endforeach()

foreach(_u FREETYPE OPENGL GLES2 FREEIMAGE RAPIDJSON DRACO VTK OPENVR TBB FFMPEG TK)
  set(USE_${_u} OFF CACHE BOOL "" FORCE)
endforeach()
set(BUILD_DOC_Overview OFF CACHE BOOL "" FORCE)
set(BUILD_Inspector OFF CACHE BOOL "" FORCE)
set(BUILD_SAMPLES_MFC OFF CACHE BOOL "" FORCE)
set(BUILD_SAMPLES_QT OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST OFF CACHE BOOL "" FORCE)

add_subdirectory(external/opencascade EXCLUDE_FROM_ALL)

# When embedded, OCCT generates Standard_Version.hxx into its own binary subdir instead of the
# aggregated include dir the rest of its headers (and its include path) use.
file(COPY
  "${CMAKE_CURRENT_BINARY_DIR}/external/opencascade/${INSTALL_DIR_INCLUDE}/Standard_Version.hxx"
  DESTINATION "${CMAKE_BINARY_DIR}/${INSTALL_DIR_INCLUDE}")

# TKDESTEP's EXTERNLIB drags its whole CAF/registry/visualization closure (~23 toolkits) into the
# build via CMake build-ordering, though a static archive doesn't need them. We call only the
# non-CAF STEPControl_Writer, so drop those deps, leaving just the 11 toolkits below. Done here, not
# in the submodule, so it survives a fresh `submodule update`.
set_target_properties(TKDESTEP PROPERTIES LINK_LIBRARIES "" INTERFACE_LINK_LIBRARIES "")

# MMG: robust conformal multi-material tetrahedral remeshing (libmmg3d_a static only). It coarsens
# the CHARM tet mesh preserving every material interface and is non-self-intersecting by
# construction, which per-patch surface decimation cannot guarantee for the folded tissues. Only the
# 3D static library is built -- no CLI apps (EXCLUDE_FROM_ALL, nothing depends on the executable), no
# MMG2D/MMGS, no optional backends. Options are forced before add_subdirectory so MMG's option()
# calls see them; libmmg3d_a propagates its own include dir to whatever links it.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_MMG3D ON CACHE BOOL "" FORCE)
set(BUILD_MMG2D OFF CACHE BOOL "" FORCE)
set(BUILD_MMGS OFF CACHE BOOL "" FORCE)
set(BUILD_MMG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(USE_SCOTCH OFF CACHE BOOL "" FORCE)
set(USE_ELAS OFF CACHE BOOL "" FORCE)
set(USE_VTK OFF CACHE BOOL "" FORCE)
set(USE_POINTMAP OFF CACHE BOOL "" FORCE)
add_subdirectory(external/mmg EXCLUDE_FROM_ALL)

# Ordered dependents-first for single-pass (ELF) linkers.
set(OCCT_LINK
  TKDESTEP TKXSBase TKShHealing TKTopAlgo TKGeomAlgo TKBRep TKGeomBase
  TKG3d TKG2d TKMath TKernel)

nanobind_add_module(_mesher_ext LTO FREE_THREADED src/headmodel_conversion/mesher/bindings.cpp)

# OCCT sets C++17 in its own scope; pin the extension to the project's C++20.
set_target_properties(_mesher_ext PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# OCCT exposes headers only via a global include_directories(), not per-target (INSTALL_DIR_INCLUDE
# is include/opencascade on Unix, inc on Windows).
target_include_directories(_mesher_ext PRIVATE "${CMAKE_BINARY_DIR}/${INSTALL_DIR_INCLUDE}")

# add_dependencies because archive-file linking doesn't imply build order. We link archive files,
# not the TKDESTEP target (whose interface would re-pull the CAF stack); --start-group on ELF makes
# intra-set link order irrelevant (ld64/MSVC resolve multi-pass and reject the flag).
add_dependencies(_mesher_ext ${OCCT_LINK})
set(OCCT_LINK_FILES "")
foreach(_tk ${OCCT_LINK})
  list(APPEND OCCT_LINK_FILES "$<TARGET_FILE:${_tk}>")
endforeach()
if(UNIX AND NOT APPLE)
  set(OCCT_LINK_FILES -Wl,--start-group ${OCCT_LINK_FILES} -Wl,--end-group)
endif()

target_link_libraries(_mesher_ext PRIVATE CGAL::CGAL libmmg3d_a ${OCCT_LINK_FILES})

# System libs TKernel needs, not inherited through archive-file linking.
if(APPLE)
  target_link_libraries(_mesher_ext PRIVATE "-framework CoreFoundation")
elseif(UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(_mesher_ext PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

install(TARGETS _mesher_ext LIBRARY DESTINATION headmodel_conversion/mesher
        COMPONENT headmodel_ext)

nanobind_add_stub(
  _mesher_ext_stub
  MODULE _mesher_ext
  OUTPUT mesher/_mesher_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_mesher_ext>
  DEPENDS _mesher_ext
  MARKER_FILE mesher/py.typed
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/mesher/_mesher_ext.pyi
  ${CMAKE_CURRENT_BINARY_DIR}/mesher/py.typed
  DESTINATION headmodel_conversion/mesher
  COMPONENT headmodel_ext
)
