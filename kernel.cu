// Vector addition (device code)

extern "C" __global__ void Sum(int *a, int *b, int *c, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}
