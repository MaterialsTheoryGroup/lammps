cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

target_link_libraries(lammps PRIVATE -L${LIBREPOSE_DIR} repose -L${LIBREPOSE_DIR}/../../airss/lib symspg lapack blas)
target_include_directories(lammps PRIVATE ${LIBREPOSE_DIR})
