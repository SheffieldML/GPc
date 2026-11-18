PROJECT( CMatrix )

OPTION( USE_HDF5 "Build with HDF5 support." )

SET( CMAKE_CXX_FLAGS    "-fPIC" )
SET( CMAKE_C_FLAGS      "-fPIC" )

SET( MLPROJECTS_BASE_DIR /home/psauer/software/sources/mlprojects )

if(USE_HDF5)
    ADD_DEFINITIONS(-D_HDF5)
    FIND_PACKAGE( HDF5 )
endif(USE_HDF5)

INCLUDE_DIRECTORIES( . ${HDF5_INCLUDE_DIR}
                       ${MLPROJECTS_BASE_DIR}/ndlutil/src
                       ${MLPROJECTS_BASE_DIR}/matrix/src
                       ${MLPROJECTS_BASE_DIR}/branches/helper/src
                       /usr/include
                       /usr/local/include        #for boost
                       /usr/local/include/eigen3 #for eigen3
                    )

LINK_DIRECTORIES( /usr/lib /usr/local/lib ${HDF5_LIBRARY_DIRS} )

SET( CMatrix_sources
    ${MLPROJECTS_BASE_DIR}/ndlutil/src/CNdlInterfaces.h
    ${MLPROJECTS_BASE_DIR}/matrix/src/CMatrix.h                  ${MLPROJECTS_BASE_DIR}/matrix/src/CMatrix.cpp

    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlassert.h               ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlassert.cpp
    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlexceptions.h

    ${MLPROJECTS_BASE_DIR}/ndlutil/src/lapack.h
    #${MLPROJECTS_BASE_DIR}/ndlutil/src/f2c.h
    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlfortran.h              ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlfortran.c
    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.h              ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.cpp
    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlutil.h                 ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlutil.cpp
    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.h              ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.cpp
)
 
ADD_DEFINITIONS( -D_FORTRAN_MAIN_FIX )
ADD_LIBRARY( cmatrix STATIC ${CMatrix_sources} )
TARGET_LINK_LIBRARIES( cmatrix hdf5_hl hdf5 f2c m lapack blas )

SUBDIRS( test )
