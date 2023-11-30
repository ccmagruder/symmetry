// Copyright 2023 Caleb Magruder

#include <stdio.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

void cudaHelloCUDA(){
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
}
