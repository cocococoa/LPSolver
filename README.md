# LPSolver

Perform Simplex method on GPU

## Build

```sh
$ cd build && cmake ..
$ make
```

## Benchmark

### Minimum distance problem

```sh
$ cd dataset && wget https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt
$ cd ../build/benchmark/minimum_distance/
$ ./minimum_distance ../../../dataset/sgb-word.txt 10 2000
```
