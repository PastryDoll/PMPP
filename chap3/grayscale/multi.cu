#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "commons/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "commons/stb_image_write.h"

#define CHANNELS 3

void colortoGrayscaleConvertion_CPU(unsigned char* data, int width, int height, int num_channels)
{
    float cr = 0.21, cg = 0.72, cb = 0.07;
    if (num_channels != 3) 
    {
        printf("Not supported number of channels: %i. Currently we only support 3\n", num_channels);
        return; 
    }
    for (int i = 0; i < width * height; i++) 
    {
        int index = i * num_channels;
        
        unsigned char r = data[index];        
        unsigned char g = data[index + 1];    
        unsigned char b = data[index + 2];    

        unsigned char gray = (unsigned char)(r * cr + g * cg + b * cb);

        data[index] = gray;          
        data[index + 1] = gray;      
        data[index + 2] = gray;      
    }

}
__global__
void colortoGrayscaleConvertion_GPU(u_char* Pout, u_char* Pin, int width, int height)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int grayOffset = row*width + col;

        int rgbOffset = grayOffset*CHANNELS;
        u_char r = Pin[rgbOffset];
        u_char g = Pin[rgbOffset + 1];
        u_char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f*r + 0.71*g + 0.07f*b;
    }
    return;
}

void run_kernel(u_char* Pout_h, u_char* Pin_h, int width, int height)
{
    u_char *Pout_d, *Pin_d;
    int size = width*height*sizeof(u_char);
    cudaMalloc((void **) &Pout_d, size);
    cudaMalloc((void **) &Pin_d, size*CHANNELS);

    cudaMemcpy(Pin_d, Pin_h, size*CHANNELS, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/32.0),ceil(height/32.0),1);
    dim3 dimBlock(32,32,1);
    colortoGrayscaleConvertion_GPU<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
    cudaFree(Pout_d);
    cudaFree(Pin_d);
}

int main() 
{
    const char* filename = "./8k.jpg";
    int x,y,n;
    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
    if (x == 0) {printf("Error loading image from %s\n", filename);}
    
    printf("Loaded image with height: %i, width: %i, channels: %i\n", y,x,n);

    clock_t start = clock();
    colortoGrayscaleConvertion_CPU(data,x,y,n);
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC * 1000;

    printf("CPU time: %.2f ms\n", time);

    const char* output_filename = "./sample_gray_cpu.jpg";
    if (!stbi_write_jpg(output_filename, x, y, n, data, 100)) {
        printf("Failed to save the image to %s\n", output_filename);
        return 1;
    }
    printf("Image saved successfully to %s\n", output_filename);

    u_char* Pout_h = (u_char*)malloc( x*y*sizeof(u_char));

    clock_t start_g = clock();
    run_kernel(Pout_h,data,x,y);
    clock_t end_g = clock();
    float time_g = (float)(end_g - start_g) / CLOCKS_PER_SEC * 1000;

    printf("GPU time: %.2f ms\n", time_g);

    output_filename = "./sample_gray_gpu.jpg";
    if (!stbi_write_jpg(output_filename, x, y, 1, Pout_h, 100)) {
        printf("Failed to save the image to %s\n", output_filename);
        return 1;
    }

    printf("Image saved successfully to %s\n", output_filename);
    stbi_image_free(data);
    stbi_image_free(Pout_h);
    
    return 0;
}