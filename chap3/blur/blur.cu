#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "commons/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "commons/stb_image_write.h"

#define BLUR_KERNEL_SIZE 7 

void blur_CPU(u_char* Pout, u_char* Pin, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < height; col++)
        {
            int pixVal = 0;
            int pixels = 0;
    
            for (int blurRow =-BLUR_KERNEL_SIZE; blurRow < BLUR_KERNEL_SIZE+1; ++blurRow)
            {
                for (int blurCol =-BLUR_KERNEL_SIZE; blurCol < BLUR_KERNEL_SIZE+1; ++blurCol)
                {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol; 
    
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                    {
                        pixVal += Pin[curRow*width + curCol];
                        ++pixels;
                    }
                }
            }
    
            Pout[row*width + col] = (u_char)(pixVal/pixels);
        }
    }
}

__global__
void blur_GPU(u_char* Pout, u_char* Pin, int width, int height)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow =-BLUR_KERNEL_SIZE; blurRow < BLUR_KERNEL_SIZE+1; ++blurRow)
        {
            for (int blurCol =-BLUR_KERNEL_SIZE; blurCol < BLUR_KERNEL_SIZE+1; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol; 

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    pixVal += Pin[curRow*width + curCol];
                    ++pixels;
                }
            }
        }

        Pout[row*width + col] = (u_char)(pixVal/pixels);
    }
}

void run_kernel(u_char* Pout_h, u_char* Pin_h, int width, int height)
{
    u_char *Pout_d, *Pin_d;
    int size = width*height*sizeof(u_char);
    cudaMalloc((void **) &Pout_d, size);
    cudaMalloc((void **) &Pin_d, size);

    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/32.0),ceil(height/32.0),1);
    dim3 dimBlock(32,32,1);
    blur_GPU<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
    cudaFree(Pout_d);
    cudaFree(Pin_d);
}

void load_and_run(const char* filename, void (*run)(u_char* Pout_h, u_char* Pin_h, int width, int height), bool gpu)
{
    int x,y,n;
    unsigned char *data = stbi_load(filename, &x, &y, &n, 1);
    if (x == 0) {printf("Error loading image from %s\n", filename);}
    
    printf("Loaded image with height: %i, width: %i, channels: %i\n", y,x,n);

    u_char* Pout_h = (u_char*)malloc( x*y*sizeof(u_char));

    clock_t start = clock();
    run(Pout_h,data,x,y);
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC * 1000;

    if (gpu) printf("GPU time: %.2f ms\n", time);
    else printf("CPU time: %.2f ms\n", time);

    const char* output_filename;
    if (gpu) output_filename = "./sample_gray_gpu.jpg";
    else output_filename = "./sample_gray_cpu.jpg";
    
    if (!stbi_write_jpg(output_filename, x, y, 1, Pout_h, 100)) {
        printf("Failed to save the image to %s\n", output_filename);
        return;
    }

    printf("Image saved successfully to %s\n", output_filename);
    
    stbi_image_free(data);
    stbi_image_free(Pout_h);

}

int main() 
{
    load_and_run("8k_bw.jpg", run_kernel, true);
    load_and_run("8k_bw.jpg", blur_CPU, false);
    return 0;
}