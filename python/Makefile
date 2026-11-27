# CIP-001D Phase A: build gpc._gpc against libgp from the repo root.
#
# Usage (from python/):
#   make          # builds ../libgp.a if needed, then gpc/_gpc*.so
#   make test     # pytest against the built extension
#   make clean

ROOT := ..
# Same platform selection as the top-level makefile: make.inc if present, else make.linux.
ifeq ($(wildcard $(ROOT)/make.inc),$(ROOT)/make.inc)
include $(ROOT)/make.inc
else
include $(ROOT)/make.linux
endif

PY ?= $(CURDIR)/.venv/bin/python
PYBIND11_INCLUDE := $(shell $(PY) -c 'import pybind11; print(pybind11.get_include())')
NUMPY_INCLUDE := $(shell $(PY) -c 'import numpy; print(numpy.get_include())')
PYTHON_INCLUDE := $(shell $(PY) -c 'import sysconfig; print(sysconfig.get_path("include"))')
EXT_SUFFIX := $(shell $(PY) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

MODULE := gpc/_gpc$(EXT_SUFFIX)
OBJS := src/module.o src/matrix_numpy.o src/wrap_gp.o src/wrap_gplvm.o src/wrap_deepgp.o src/wrap_likelihood.o src/wrap_interchange.o
REF_FTC := ref_ftc_objective

# Reuse root CCFLAGS (includes -fPIC, -std=c++17) plus pybind/numpy.
PY_CCFLAGS = $(CCFLAGS) -MMD -MP -I$(ROOT) -I$(PYBIND11_INCLUDE) -I$(NUMPY_INCLUDE) -I$(PYTHON_INCLUDE) -Isrc
# On macOS, leave Python symbols undefined until dlopen (pybind11 convention).
ifeq ($(PLAT),_OSX)
PY_UNDEF = -undefined dynamic_lookup
endif
# Prefer static libgp.a so the extension does not depend on DYLD_LIBRARY_PATH
# for a sibling libgp.dylib (macOS dyld searches -lgp as .dylib first).
PY_LDFLAGS = $(ROOT)/libgp$(LIBSEXT) $(LDFLAGS) $(PY_UNDEF)
REF_CCFLAGS = $(CCFLAGS) -I$(ROOT)
REF_LDFLAGS = $(ROOT)/libgp$(LIBSEXT) $(LDFLAGS)

.PHONY: all test clean libgp FORCE

all: $(MODULE)

libgp:
	$(MAKE) -C $(ROOT) libgp$(LIBSEXT)

# FORCE makes the root make decide whether libgp is stale; without a prerequisite
# an existing libgp.a would never be rebuilt after a root source change.
$(ROOT)/libgp$(LIBSEXT): FORCE
	$(MAKE) -C $(ROOT) libgp$(LIBSEXT)

FORCE:

# Header dependencies come from -MMD (src/*.d, also covering ../*.h).
# Rebuild when the compile flags change (the stamp is owned by the root makefile).
$(ROOT)/.buildflags: FORCE
	$(MAKE) -C $(ROOT) .buildflags

src/%.o: src/%.cpp $(ROOT)/.buildflags
	$(CC) -c $< -o $@ $(PY_CCFLAGS)

-include $(OBJS:.o=.d)

$(MODULE): $(OBJS) $(ROOT)/libgp$(LIBSEXT)
	$(LD) ${XLINKERFLAGS} -shared -o $(MODULE) $(OBJS) $(PY_LDFLAGS)

$(REF_FTC): ref_ftc_objective.cpp $(ROOT)/libgp$(LIBSEXT)
	$(CC) ref_ftc_objective.cpp -o $(REF_FTC) $(REF_CCFLAGS) $(REF_LDFLAGS)

test: $(MODULE) $(REF_FTC)
	cd $(CURDIR) && $(PY) -m pytest -q tests

clean:
	rm -f src/*.o src/*.d $(MODULE) $(REF_FTC) ref_ftc_objective.o
