export C_INCLUDE_PATH=/usr/lib/openmpi/include

# compiler choice
CC    = ${CONDA_PREFIX}/bin/x86_64-conda_cos6-linux-gnu-gcc

all: fastmodules

.PHONY : fastmodules

qhull:
	make -C qhull/src

voboz:
	make -C src all

fastmodules:
	python python_tools/setup.py build_ext --inplace
	mv fastmodules*.so python_tools/.

clean:
	make -C src clean
	make -C qhull/src cleanall
	rm -f bin/*
	rm -f python_tools/*.*o
	rm -f python_tools/fastmodules.c
	rm -f python_tools/fastmodules*.so
	rm -f python_tools/*.pyc
