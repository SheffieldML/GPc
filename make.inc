#
#
#  The machine (platform) identifier to append to the library names
#
SHELL = /bin/sh

PLAT = _LINUX
LAPACKLIBS = -llapack -lblas
F2CLIB = -lgfortran
# commands to make statically&dynamically linked libraries and their extensions
LIBSCOMMAND = ar rcs
LIBDCOMMAND = gcc -shared -o
LIBSEXT = .a
LIBDEXT = .so

# Uncomment these lines to set up the MATLAB interface.
#MATLABLIBS = -lmat -lmx
#MATLABBASE=/usr/local/pkg/matlab/
#MATLABRUNLIBDIR=${MATLABBASE}/bin/glnx86
#MATLABLIBDIR=${MATLABBASE}/bin/glnx86
#MATLABINCLUDE=${MATLABBASE}/extern/include
#MATLABXLINKFLAG=-L${MATLABLIBDIR} -Xlinker -rpath -Xlinker ${MATLABRUNLIBDIR} 
#MATLABFLAG=-D_NDLMATLAB

# Optional libmatio path for reading numeric .mat fixtures without MATLAB (CIP-0001).
# Enable with: make USE_MATIO=1 testMatrix
ifeq ($(USE_MATIO),1)
MATIOFLAG = -D_NDLMATIO
MATIOLIBS = -lmatio
endif

# Default to the project root so -I is never empty when MATLAB is disabled.
# When enabling MATLAB above, set MATLABINCLUDE and it will be used instead.
ifeq ($(strip $(MATLABINCLUDE)),)
INCLUDE = .
else
INCLUDE = $(MATLABINCLUDE)
endif
LD=g++
LDFLAGS = ${MATLABLIBS} ${MATIOLIBS} ${LAPACKLIBS} ${F2CLIB} -lc ${MATLABXLINKFLAG}
CC=g++
# C++17: dynamic exception specifications removed from the sources (CIP-0002).
CCFLAGS = -std=c++17 -ggdb -DDBG -fPIC -I$(INCLUDE) -D$(PLAT) $(MATLABFLAG) $(MATIOFLAG)
#CCFLAGS = -std=c++17 -O3 -fPIC -I$(INCLUDE) -D$(PLAT) $(MATLABFLAG) $(MATIOFLAG)
FC=gfortran
FCFLAGS = -funroll-all-loops -O3 -fPIC

