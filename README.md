# Bilinear interpolation
This repo holds three implementations of the same bilinear interpolation algorithm:
* CPU
* GPU (CUDA)
* Many-Core Epiphany chip

The code is used mainly as reference to the blog post:
> http://mkaczanowski.com/parallella-part-10-power-efficiency

# Usage
Each implemenation is configured by cmake, so follow the same steps:
```
# install deps
pacman -S sdl2 sdl2_image opencv libpng cmake

# clone repo & build
git clone https://github.com/mkaczanowski/interpolation
cd interpolation/cpu

mkdir build && cd build
cmake ../
make

# select 105th frame
./interpolation 105
file output.png
```
