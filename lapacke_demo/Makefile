# Makefile for lapacke_demo.c with user's choice of LAPACKE implementation.
# optional dynamic linking to GSL if desired. MKL cannot be linked with GSL,

# C compiler and flags
CC                     = gcc
CFLAGS                 = -Wall -g
# set to 1 to include GSL headers and link GSL
GSL_INCLUDE           ?=
# location of GSL install path. ignored if GSL_INCLUDE is 0
GSL_PATH              ?= /usr/local
# location of OpenBLAS, LAPACKE, MKL install paths
OPENBLAS_PATH         ?= /opt/OpenBLAS
LAPACKE_PATH          ?= /usr
MKL_PATH              ?= /usr
# name of build targets linking OpenBLAS, LAPACKE, MKL respectively
tgt_openblas           = demo_openblas
tgt_lapacke            = demo_lapacke
tgt_mkl                = demo_mkl
# include lines for GSL, LAPACKE, OpenBLAS, MKL
gsl_include_line       = -I$(GSL_PATH)/include
lapacke_include_line   = -I$(LAPACKE_PATH)/include \
	-I$(LAPACKE_PATH)/LAPACKE/include
openblas_include_line  = -I$(OPENBLAS_PATH)/include
mkl_include_line       = -I$(MKL_PATH)/include -I$(MKL_PATH)/include/mkl
# linker lines for GSL, LAPACKE, OpenBLAS, MKL. -lm may not be needed in
# gsl_link_line for some systems. Intel MKL link line courtesy of the Intel
# Math Kernel Library Link Line Advisor. requires libpthread, targets x86-64.
gsl_link_line          = -L$(GSL_PATH)/lib \
	-L$(GSL_PATH)/lib/x86_64-linux-gnu -Wl,-rpath,$(GSL_PATH)/lib \
	-Wl,-rpath,$(GSL_PATH)/lib/x86_64-linux-gnu -lgsl -lm
lapacke_link_line      = -L$(LAPACKE_PATH) -L$(LAPACKE_PATH)/lib \
	-L$(LAPACKE_PATH)/lib/x86_64-linux-gnu -Wl,-rpath,$(LAPACKE_PATH) \
	-Wl,-rpath,$(LAPACKE_PATH)/lib \
	-Wl,-rpath,$(LAPACKE_PATH)/lib/x86_64-linux-gnu -llapacke
openblas_link_line     = -L$(OPENBLAS_PATH)/lib \
	-Wl,-rpath,$(OPENBLAS_PATH)/lib -lopenblas
mkl_link_line          = -Wl,--no-as-needed \
	-L$(MKL_PATH)/lib/x86_64-linux-gnu -L$(MKL_PATH)/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# phony targets
.PHONY: dummy

# triggered if no target is provided
dummy:
	@echo "Target must be one of $(tgt_openblas), $(tgt_lapacke), $(tgt_mkl)"

# build using OpenBLAS
$(tgt_openblas): lapacke_demo.c
# only include + link GSL if GSL_INCLUDE is 1
ifeq ($(GSL_INCLUDE), 1)
	$(CC) $(CFLAGS) -DLAPACKE_INCLUDE -DGSL_INCLUDE $(gsl_include_line) \
	$(openblas_include_line) -o lapacke_demo lapacke_demo.c \
	$(gsl_link_line) $(openblas_link_line)
else
	$(CC) $(CFLAGS) -DLAPACKE_INCLUDE $(openblas_include_line) -o \
	lapacke_demo lapacke_demo.c $(openblas_link_line)
endif

# build using LAPACKE. note that we add multiple include and lib directories
# based off LAPACKE_PATH prefix to handle several different install cases.
$(tgt_lapacke): lapacke_demo.c
ifeq ($(GSL_INCLUDE), 1)
	$(CC) $(CFLAGS) -DLAPACKE_INCLUDE -DGSL_INCLUDE $(gsl_include_line) \
	$(lapacke_include_line) -o lapacke_demo lapacke_demo.c \
	$(gsl_link_line) $(lapacke_link_line)
	
else
	$(CC) $(CFLAGS) -DLAPACKE_INCLUDE $(lapacke_include_line) -o lapacke_demo \
	lapacke_demo.c $(lapacke_link_line)
endif

# build using Intel MKL. note that we add multiple include and lib directories
# based off MKL_PATH prefix to handle several different install cases.
$(tgt_mkl): lapacke_demo.c
# tab required for the $(error) call, else it always triggers if GSL_INCLUDE=1
ifeq ($(GSL_INCLUDE), 1)
	$(error cannot pass -DGSL_INCLUDE with -DMKL_INCLUDE)
else
	$(CC) $(CFLAGS) -DMKL_INCLUDE $(mkl_include_line) -m64 -o lapacke_demo \
	lapacke_demo.c $(mkl_link_line)
endif