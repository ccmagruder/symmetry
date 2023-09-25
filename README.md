This project reproduces some of the figures found in "Symmetry in Chaos: A Search for Pattern in Mathematics, Second Edition" by Michael Field and Martin Golubitsky. If you haven't had the opportunity to read their book, it's fascinating and its images otherworldy. I loved the book and found reproducing its pictures irresistible. 

# Image Gallery

Reproduced images can be found on this project's site [https://ccmagruder.github.io/symmetry](https://ccmagruder.github.io/symmetry)

[![Image](/SymmetryInChaos.png "Symmetry In Chaos Image")](https://ccmagruder.github.io/symmetry)

# Installing Symmetry

## Docker Compose

```sh
docker compose run --rm symmetry symmetry run config/fig1-10.json images/im1-10.pgm
```

## Building From Source
```sh
git clone https://github.com/ccmagruder/symmetry.git
cd Symmetry
cmake -DCMAKE_BUILD_TYPE=Release -B build .
cmake --build build
```

## Screen Resolutions
1. 5120 x 2160 21:9 LG HDR 5K
1. 2560 x 1440 16:9 DELL U2722DE
1. 1920 x 1080 16:9 DELL S2719H
1. 1920 x 1080 16:9 ACER R271

## Wallpaper Resolutions

1. 3840 x 2160 16:9 Ultra HD, UHD, 4K, 2160p
1. 2560 x 1440 16:9 Quad HD, QHD, 2.5K, 1440p
1. 1920 x 1080 16:9 Full HD, FHD, 2K, 1080p

