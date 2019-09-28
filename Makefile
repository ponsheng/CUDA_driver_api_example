EXE=cuda_driver_example

all: $(EXE)

kernel.ptx: kernel.cu
	nvcc $< --ptx -o $@

cuda_driver_example: cuda_driver_example.cpp kernel.ptx
	nvcc $< -o $@ -lcuda

clean:
	rm -f $(EXE) kernel.ptx


