# LPSolver on GPU

Perform simplex method on GPU.

## Build

-   Requirements
    -   or-tools

```sh
$ cd externals && wget https://github.com/google/or-tools/releases/download/v7.8/or-tools_ubuntu-18.04_v7.8.7959.tar.gz
$ tar -zxvf tar -xzvf or-tools_ubuntu-18.04_v7.8.7959.tar.gz
$ cd ../build && cmake ..
$ make
```

## Benchmark

### Minimum distance problem

LPSolver on GPU is currently much slower than GLOP.

```sh
$ cd dataset && wget https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt
$ cd ../build/benchmark/minimum_distance/
$ ./minimum_distance ../../../dataset/sgb-words.txt 10 1000
```
