# Prefer Homebrew g++/gfortran on Apple so C++ and Fortran share an ABI (make.osx).
# Include before project(). Honours CXX/FC env and explicit CMAKE_*_COMPILER.

option(GPC_PREFER_HOMEBREW_GCC "Prefer Homebrew g++/gfortran on Apple (CIP-0020)" ON)
if(APPLE AND GPC_PREFER_HOMEBREW_GCC AND NOT DEFINED ENV{CXX} AND NOT DEFINED ENV{FC})
  if(NOT CMAKE_CXX_COMPILER AND NOT CMAKE_Fortran_COMPILER)
    execute_process(
      COMMAND brew --prefix gcc
      OUTPUT_VARIABLE _GPC_GCC_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_GPC_GCC_PREFIX)
      # Use /bin/ls so shell aliases (ls -F) cannot append type suffixes.
      execute_process(
        COMMAND sh -c "/bin/ls '${_GPC_GCC_PREFIX}/bin'/g++-[0-9]* 2>/dev/null | sort -V | tail -1"
        OUTPUT_VARIABLE _GPC_GXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      execute_process(
        COMMAND sh -c "/bin/ls '${_GPC_GCC_PREFIX}/bin'/gfortran-[0-9]* 2>/dev/null | sort -V | tail -1"
        OUTPUT_VARIABLE _GPC_GFORTRAN
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(_GPC_GXX AND _GPC_GFORTRAN)
        set(CMAKE_CXX_COMPILER "${_GPC_GXX}" CACHE FILEPATH "C++ compiler" FORCE)
        set(CMAKE_Fortran_COMPILER "${_GPC_GFORTRAN}" CACHE FILEPATH "Fortran compiler" FORCE)
        set(CMAKE_C_COMPILER "${_GPC_GXX}" CACHE FILEPATH "C compiler" FORCE)
      endif()
    endif()
  endif()
endif()
