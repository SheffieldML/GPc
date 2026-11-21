# CIP-001D Phase A: build gpc._gpc against libgp from the repo root.
#
# Usage (from python/):
#   make          # builds ../libgp.a if needed, then gpc/_gpc*.so
#   make test     # pytest against the built extension
#   make clean

ROOT := ..
include $(ROOT)/make.inc

PY ?= $(CURDIR)/.venv/bin/python
PYBIND11_INCLUDE := $(shell $(PY) -c 'import pybind11; print(pybind11.get_include())')
NUMPY_INCLUDE := $(shell $(PY) -c 'import numpy; print(numpy.get_include())')
PYTHON_INCLUDE := $(shell $(PY) -c 'import sysconfig; print(sysconfig.get_path("include"))')
EXT_SUFFIX := $(shell $(PY) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

MODULE := gpc/_gpc$(EXT_SUFFIX)
OBJS := src/module.o src/matrix_numpy.o src/wrap_gp.o src/wrap_gplvm.o src/wrap_deepgp.o
REF_FTC := ref_ftc_objective

# Reuse root CCFLAGS (includes -fPIC, -std=c++17) plus pybind/numpy.
PY_CCFLAGS = $(CCFLAGS) -I$(ROOT) -I$(PYBIND11_INCLUDE) -I$(NUMPY_INCLUDE) -I$(PYTHON_INCLUDE) -Isrc
# On macOS, leave Python symbols undefined until dlopen (pybind11 convention).
ifeq ($(PLAT),_OSX)
PY_UNDEF = -undefined dynamic_lookup
endif
# Prefer static libgp.a so the extension does not depend on DYLD_LIBRARY_PATH
# for a sibling libgp.dylib (macOS dyld searches -lgp as .dylib first).
PY_LDFLAGS = $(ROOT)/libgp$(LIBSEXT) $(LDFLAGS) $(PY_UNDEF)
REF_CCFLAGS = $(CCFLAGS) -I$(ROOT)
REF_LDFLAGS = $(ROOT)/libgp$(LIBSEXT) $(LDFLAGS)

.PHONY: all test clean libgp

all: $(MODULE)

libgp:
	$(MAKE) -C $(ROOT) libgp$(LIBSEXT)

$(ROOT)/libgp$(LIBSEXT):
	$(MAKE) -C $(ROOT) libgp$(LIBSEXT)

src/module.o: src/module.cpp src/matrix_numpy.h src/wrap_gp.h src/wrap_gplvm.h src/wrap_deepgp.h
	$(CC) -c src/module.cpp -o src/module.o $(PY_CCFLAGS)

src/matrix_numpy.o: src/matrix_numpy.cpp src/matrix_numpy.h
	$(CC) -c src/matrix_numpy.cpp -o src/matrix_numpy.o $(PY_CCFLAGS)

src/wrap_gp.o: src/wrap_gp.cpp src/wrap_gp.h src/matrix_numpy.h
	$(CC) -c src/wrap_gp.cpp -o src/wrap_gp.o $(PY_CCFLAGS)

src/wrap_gplvm.o: src/wrap_gplvm.cpp src/wrap_gplvm.h src/matrix_numpy.h
	$(CC) -c src/wrap_gplvm.cpp -o src/wrap_gplvm.o $(PY_CCFLAGS)

src/wrap_deepgp.o: src/wrap_deepgp.cpp src/wrap_deepgp.h src/matrix_numpy.h
	$(CC) -c src/wrap_deepgp.cpp -o src/wrap_deepgp.o $(PY_CCFLAGS)

$(MODULE): $(OBJS) $(ROOT)/libgp$(LIBSEXT)
	$(LD) ${XLINKERFLAGS} -shared -o $(MODULE) $(OBJS) $(PY_LDFLAGS)

$(REF_FTC): ref_ftc_objective.cpp $(ROOT)/libgp$(LIBSEXT)
	$(CC) ref_ftc_objective.cpp -o $(REF_FTC) $(REF_CCFLAGS) $(REF_LDFLAGS)

test: $(MODULE) $(REF_FTC)
	cd $(CURDIR) && $(PY) -m pytest -q tests

clean:
	rm -f src/*.o $(MODULE) $(REF_FTC) ref_ftc_objective.o
