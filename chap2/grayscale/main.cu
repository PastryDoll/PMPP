#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "commons/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "commons/stb_image_write.h"

void ModifyToGrayScale(unsigned char* data, int width, int height, int num_channels)
{
    float cr = 0.21, cg = 0.72, cb = 0.07;
    if (num_channels != 3) 
    {
        printf("Not supported number of channels: %i. Currently we only support 3", num_channels);
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

int main() 
{
    const char* filename = "../assets/sample_color.jpg";
    int x,y,n;
    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
    if (x == 0) {printf("Error loading image from %s", filename);}
    
    printf("Loaded image with height: %i, width: %i, channels: %i\n", y,x,n);

    ModifyToGrayScale(data,x,y,n);
    const char* output_filename = "../assets/sample_gray.jpg";
    if (!stbi_write_jpg(output_filename, x, y, n, data, 100)) {
        printf("Failed to save the image to %s\n", output_filename);
        return 1;
    }

    printf("Image saved successfully to %s\n", output_filename);
    stbi_image_free(data);
    
    return 0;
}