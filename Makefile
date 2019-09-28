EXE=driver_api unified_memory

all: $(EXE)

kernel.ptx: kernel.cu
	nvcc $< --ptx -o $@

%: %.cpp kernel.ptx
	nvcc $< -o $@ -lcuda

clean:
	rm -f $(EXE) kernel.ptx


