#include "solver.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>

namespace solver
{
    void check_error(const std::string& title)
    {
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            auto name = cudaGetErrorName(error);
            std::cerr << title << ": ";
            std::cerr << name << std::endl;
            exit(1);
        }
    }
    void print_fractional(const std::string& title, const Fractional* d_ptr, std::size_t sz)
    {
        std::vector<Fractional> vec(sz);
        cudaMemcpy(vec.data(), d_ptr, sizeof(Fractional) * sz, cudaMemcpyDeviceToHost);
        std::cerr << title << ": ";
        for (const auto& e : vec) std::cerr << e.x << "/" << e.y << ", ";
        std::cerr << std::endl;
    }
    template <typename T> void print(const std::string& title, const T* d_ptr, std::size_t sz)
    {
        std::vector<T> vec(sz);
        cudaMemcpy(vec.data(), d_ptr, sizeof(T) * sz, cudaMemcpyDeviceToHost);
        std::cerr << title << ": ";
        for (const auto& e : vec) std::cerr << e << ", ";
        std::cerr << std::endl;
    }
    void print_lp(const LinearProblem* d_ptr)
    {
        LinearProblem lp;
        cudaMemcpy(&lp, d_ptr, sizeof(LinearProblem), cudaMemcpyDeviceToHost);
        print_fractional("mat", lp.mat, lp.n * lp.m);
        print_fractional("obj", lp.obj, lp.n);
        print_fractional("con", lp.con, lp.m);
        print("bas", lp.basis, lp.m);
        print("non", lp.non_basis, lp.n);
        std::cerr << "v  : " << lp.v.x << "/" << lp.v.y << std::endl;
    }

    void decodeSolution(const LinearProblem& lp, Fractional* vec)
    {
        const auto n = lp.n;
        const auto m = lp.m;
        std::vector<std::uint32_t> basis(m);
        std::vector<std::uint32_t> non_basis(n);
        std::vector<Fractional> con(m);
        cudaMemcpy(basis.data(), lp.basis, sizeof(std::uint32_t) * m, cudaMemcpyDeviceToHost);
        cudaMemcpy(non_basis.data(), lp.non_basis, sizeof(std::uint32_t) * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(con.data(), lp.con, sizeof(Fractional) * m, cudaMemcpyDeviceToHost);
        for (std::size_t s = 0; s < n; ++s) {
            const auto nb = non_basis[s];
            if (nb < n) vec[nb] = make_int2(0, 0);
        }
        for (std::size_t s = 0; s < m; ++s) {
            const auto b = basis[s];
            if (b < n) vec[b] = con[s];
        }
    }

    template <typename T>
    __forceinline__ __host__ __device__ void kernelMemcpy(T* dst, const T* __restrict__ src, uint32_t size)
    {
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) dst[i] = src[i];
    }

    LPSolver::LPSolver(const std::uint32_t n, const std::uint32_t m, const Fractional* mat, const Fractional* obj,
                       const Fractional* con, const Fractional v)
        : n_(n), m_(m), optimal_(false), value_(v), answer_(n, make_int2(0, 1))
    {
        cudaMalloc(&d_mat1_, sizeof(Fractional) * n_ * m_);
        cudaMalloc(&d_mat2_, sizeof(Fractional) * n_ * m_);
        cudaMalloc(&d_obj1_, sizeof(Fractional) * n_);
        cudaMalloc(&d_obj2_, sizeof(Fractional) * n_);
        cudaMalloc(&d_con1_, sizeof(Fractional) * m_);
        cudaMalloc(&d_con2_, sizeof(Fractional) * m_);
        cudaMalloc(&d_basis1_, sizeof(std::uint32_t) * m_);
        cudaMalloc(&d_basis2_, sizeof(std::uint32_t) * m_);
        cudaMalloc(&d_non_basis1_, sizeof(std::uint32_t) * n_);
        cudaMalloc(&d_non_basis2_, sizeof(std::uint32_t) * n_);
        cudaMalloc(&d_lp1_, sizeof(LinearProblem));
        cudaMalloc(&d_lp2_, sizeof(LinearProblem));

        cudaMemcpy(d_mat1_, mat, sizeof(Fractional) * n_ * m_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat2_, mat, sizeof(Fractional) * n_ * m_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_obj1_, obj, sizeof(Fractional) * n_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_obj2_, obj, sizeof(Fractional) * n_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_con1_, con, sizeof(Fractional) * m_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_con2_, con, sizeof(Fractional) * m_, cudaMemcpyHostToDevice);
        std::vector<std::uint32_t> basis(m_);
        std::iota(basis.begin(), basis.end(), n_);
        cudaMemcpy(d_basis1_, basis.data(), sizeof(std::uint32_t) * m_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis2_, basis.data(), sizeof(std::uint32_t) * m_, cudaMemcpyHostToDevice);
        std::vector<std::uint32_t> non_basis(n_);
        std::iota(non_basis.begin(), non_basis.end(), 0);
        cudaMemcpy(d_non_basis1_, non_basis.data(), sizeof(std::uint32_t) * n_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_non_basis2_, non_basis.data(), sizeof(std::uint32_t) * n_, cudaMemcpyHostToDevice);
        LinearProblem lp1{n_, m_, d_mat1_, d_obj1_, d_con1_, v, d_basis1_, d_non_basis1_};
        LinearProblem lp2{n_, m_, d_mat2_, d_obj2_, d_con2_, v, d_basis2_, d_non_basis2_};
        cudaMemcpy(d_lp1_, &lp1, sizeof(LinearProblem), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lp2_, &lp2, sizeof(LinearProblem), cudaMemcpyHostToDevice);
    }

    __global__ void findExchangeIndexPre(const LinearProblem* __restrict__ lp, Fractional* buffer,
                                         std::uint32_t* in_buffer, std::uint32_t* out_buffer, std::uint8_t* converged,
                                         uint2* inout)
    {
        const auto n = lp->n;  // 項の数
        const auto m = lp->m;  // 制約の数
        for (std::uint32_t block = blockIdx.x; block < n; block += gridDim.x) {
            // ブロックごとに、1つの非基底変数を探索する
            if (threadIdx.x == 0) {
                buffer[block * m] = make_int2(-1, 0);  // 収束したかどうか判定するために仕込む
                in_buffer[block] = block;              // 初期化
            }
            const auto obj_coef = lp->obj[block];
            if (negative(obj_coef) || zero(obj_coef)) continue;

            for (std::uint32_t thread = threadIdx.x; thread < m; thread += blockDim.x) {
                // スレッドごとに、1つの基底変数を探索する
                // 最もタイトな制約を選んだ場合にどの程度目的関数が大きくなるのか計算する
                const auto offset = block * m + thread;
                const auto mat_offset = thread * n + block;  // TODO: 行列のアクセスが良くない
                const auto con_coef = lp->con[thread];
                const auto mat_coef = lp->mat[mat_offset];
                out_buffer[offset] = thread;
                if (zero(mat_coef) || negative(mat_coef))
                    buffer[offset] = make_int2(1, 0);  // INFINITY
                else
                    buffer[offset] = div(con_coef, mat_coef);
            }

            __syncthreads();

            // reduction
            // TODO: 最後までreductionするのはパフォーマンス的に良くなさそう
            std::uint32_t tmp_m = m;
            while (tmp_m >= 2) {
                const auto half = (tmp_m + 1) / 2;
                for (std::uint32_t thread = threadIdx.x; thread < half; thread += blockDim.x) {
                    if (thread + half >= tmp_m) continue;
                    const auto offset = block * m + thread;
                    if (gt(buffer[offset], buffer[offset + half])) {
                        // printf("[%d, %d] %d/%d > %d/%d\n", block, thread, buffer[offset].x, buffer[offset].y,
                        //        buffer[offset + half].x, buffer[offset + half].y);
                        buffer[offset] = buffer[offset + half];
                        out_buffer[offset] = out_buffer[offset + half];
                    }
                }
                tmp_m = half;
                __syncthreads();
            }
            // buffer[block*m]に非基底変数blockの、制約を満たす範囲での最大値が格納されている
            // obj_coefを乗じることで、目的関数の上昇度が計算できる
            if (threadIdx.x == 0) buffer[block * m] = mul(buffer[block * m], obj_coef);
        }
        return;
    }

    // TODO: 実質、1ブロックしか機能していない
    __global__ void findExchangeIndexAft(const LinearProblem* __restrict__ lp, Fractional* buffer,
                                         std::uint32_t* in_buffer, std::uint32_t* out_buffer, std::uint8_t* converged,
                                         uint2* inout)
    {
        // シンプレックス法が収束したかチェック
        __shared__ std::uint32_t flag;
        if (threadIdx.x == 0) flag = 0;
        __syncthreads();
        for (std::uint32_t thread = threadIdx.x; thread < lp->n; thread += blockDim.x) {
            if (buffer[thread * lp->m].x == -1 && buffer[thread * lp->m].y == 0) atomicAdd(&flag, 1);
        }
        __syncthreads();
        if (flag == lp->n) {
            if (threadIdx.x == 0 && blockIdx.x == 0) *converged = 1;
            return;
        }

        const auto n = lp->n;  // 項の数
        const auto m = lp->m;  // 制約の数
        // reduction
        if (blockIdx.x == 0) {
            /*
            if (threadIdx.x == 0)
            {
                for (std::uint32_t i = 0; i < n; ++i) printf("%d/%d, ", buffer[i * m].x, buffer[i * m].y);
                printf("\n");

                for (std::uint32_t i = 0; i < n; ++i) printf("%d, ", out_buffer[i * m]);
                printf("\n");
            } */
            __syncthreads();
            std::uint32_t tmp_n = n;
            while (tmp_n >= 2) {
                const auto half = (tmp_n + 1) / 2;
                for (std::uint32_t thread = threadIdx.x; thread < half; thread += blockDim.x) {
                    if (thread + half >= tmp_n) continue;
                    const auto offset = thread * m;
                    const auto step = half * m;
                    // const auto a = buffer[offset];
                    // const auto b = buffer[offset + step];
                    // printf("[%d, %d] %d/%d v.s. %d/%d\n", blockIdx.x, thread, a.x, a.y, b.x, b.y);
                    if (lt(buffer[offset], buffer[offset + step])) {
                        buffer[offset] = buffer[offset + step];
                        out_buffer[offset] = out_buffer[offset + step];
                        in_buffer[thread] = in_buffer[thread + half];
                    }
                }
                tmp_n = half;
                __syncthreads();
            }
        }

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // TODO: 解が有界か判定
            *inout = make_uint2(in_buffer[0], out_buffer[0]);
            // printf("in: %d, out: %d\n", inout->x, inout->y);
        }
        return;
    }

    __global__ void pivot(const LinearProblem* __restrict__ lp_old, LinearProblem* lp_new, const uint2* inout)
    {
        const auto n = lp_old->n;   // 項の数
        const auto m = lp_old->m;   // 制約の数
        const auto in = inout->x;   // 非基底変数  →   基底変数
        const auto out = inout->y;  // 基底変数    → 非基底変数

        __shared__ Fractional zero;
        __shared__ Fractional one;
        __shared__ LinearProblem lp;
        __shared__ LinearProblem to;
        __shared__ Fractional ale;
        __shared__ Fractional ce;
        if (threadIdx.x == 0) {
            zero = make_int2(0, 1);
            one = make_int2(1, 1);
            lp = *lp_old;
            to = *lp_new;
            ale = lp.mat[out * n + in];
            ce = lp.obj[in];
        }
        __syncthreads();

        // 基底変数、非基底変数をコピーする
        if (blockIdx.x == 0) {
            kernelMemcpy(to.basis, lp.basis, m);
            kernelMemcpy(to.non_basis, lp.non_basis, n);
            __syncthreads();
            if (threadIdx.x == 0) {
                to.basis[out] = lp.non_basis[in];
                to.non_basis[in] = lp.basis[out];
            }
        }

        for (std::uint32_t block = blockIdx.x; block < m; block += gridDim.x) {
            // blockが基底変数(==制約)をなめる
            // threadが非基底変数(==項)をなめる
            // see アルゴリズムイントロダクション
            // out    -> l
            // in     -> e
            // block  -> i
            // thread -> j
            if (block == out) {
                for (std::uint32_t thread = threadIdx.x; thread < n; thread += blockDim.x) {
                    const auto alj = lp.mat[out * n + thread];
                    const auto cj = lp.obj[thread];
                    if (eq(alj, zero)) {
                        to.mat[out * n + thread] = zero;
                        to.obj[thread] = cj;
                        continue;
                    }
                    const auto aej_hat = div(alj, ale);
                    const auto cj_hat = sub(cj, mul(ce, aej_hat));
                    to.mat[out * n + thread] = aej_hat;
                    to.obj[thread] = cj_hat;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    const auto be_hat = div(lp.con[out], ale);
                    const auto ael_hat = div(one, ale);
                    to.con[out] = be_hat;
                    to.mat[out * n + in] = ael_hat;
                    lp_new->v = add(lp.v, mul(ce, be_hat));
                    to.obj[in] = negate(mul(ce, ael_hat));
                }
            } else {
                const auto aie = lp.mat[block * n + in];
                for (std::uint32_t thread = threadIdx.x; thread < n; thread += blockDim.x) {
                    const auto aij = lp.mat[block * n + thread];
                    if (eq(aie, zero)) {
                        to.mat[block * n + thread] = aij;
                        continue;
                    }
                    const auto alj = lp.mat[out * n + thread];
                    if (eq(alj, zero)) {
                        to.mat[block * n + thread] = aij;
                        continue;
                    }
                    const auto aej_hat = div(alj, ale);
                    const auto aij_hat = sub(aij, mul(aie, aej_hat));
                    to.mat[block * n + thread] = aij_hat;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    const auto be_hat = div(lp.con[out], ale);
                    const auto ael_hat = div(make_int2(1, 1), ale);
                    to.con[block] = sub(lp.con[block], mul(aie, be_hat));
                    to.mat[block * n + in] = negate(mul(aie, ael_hat));
                }
            }
        }
    }

    std::tuple<float, int, int> maxOccupancyConfiguration()
    {
        int min_grid_size = 0;
        int block_size = 0;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, pivot, 0, 0);
        // calculate theoretical occupancy
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, pivot, block_size, 0);
        int device;
        cudaDeviceProp props;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        float occupancy = static_cast<float>(max_active_blocks * block_size / props.warpSize) /
                          static_cast<float>(props.maxThreadsPerMultiProcessor / props.warpSize);

        return {occupancy, min_grid_size, block_size};
    }

    void LPSolver::solve(std::chrono::milliseconds timeout)
    {
        const auto [occupancy, grid, block] = maxOccupancyConfiguration();

        const auto start = std::chrono::high_resolution_clock::now();
        std::uint8_t* d_converged;
        cudaMalloc(&d_converged, sizeof(std::uint8_t));
        cudaMemset(d_converged, 0, sizeof(std::uint8_t));
        uint2* d_inout;
        cudaMalloc(&d_inout, sizeof(uint2));
        Fractional* d_buffer;
        cudaMalloc(&d_buffer, sizeof(Fractional) * n_ * m_);
        std::uint32_t* d_in_buffer;
        cudaMalloc(&d_in_buffer, sizeof(std::uint32_t) * n_);
        std::uint32_t* d_out_buffer;
        cudaMalloc(&d_out_buffer, sizeof(std::uint32_t) * n_ * m_);

        using Time = std::chrono::duration<float, std::milli>;
        auto acc_find = Time::zero();
        auto acc_pivot = Time::zero();
        std::uint32_t loop = 0;
        while (true) {
            loop++;
            const auto tmp_start = std::chrono::high_resolution_clock::now();
            const auto s_find = std::chrono::high_resolution_clock::now();
            findExchangeIndexPre<<<grid, block>>>(d_lp1_, d_buffer, d_in_buffer, d_out_buffer, d_converged, d_inout);
            findExchangeIndexAft<<<grid, block>>>(d_lp1_, d_buffer, d_in_buffer, d_out_buffer, d_converged, d_inout);
            std::uint8_t converged;
            cudaMemcpy(&converged, d_converged, sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
            const auto e_find = std::chrono::high_resolution_clock::now();
            acc_find += std::chrono::duration_cast<Time>(e_find - s_find);

            if (converged != 0) {
                optimal_ = true;
                LinearProblem lp;
                cudaMemcpy(&lp, d_lp1_, sizeof(LinearProblem), cudaMemcpyDeviceToHost);
                value_ = lp.v;
                decodeSolution(lp, answer_.data());
                break;
            }

            const auto s_pivot = std::chrono::high_resolution_clock::now();
            pivot<<<grid, block>>>(d_lp1_, d_lp2_, d_inout);
            cudaDeviceSynchronize();
            const auto e_pivot = std::chrono::high_resolution_clock::now();
            acc_pivot += std::chrono::duration_cast<Time>(e_pivot - s_pivot);

            std::swap(d_lp1_, d_lp2_);

            const auto tmp_stop = std::chrono::high_resolution_clock::now();
            const auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(tmp_stop - start);
            const auto step_time = std::chrono::duration_cast<std::chrono::milliseconds>(tmp_stop - tmp_start);
            if (current_time + step_time > timeout) break;
        }

        std::cout << "loop     : " << loop << std::endl;
        std::cout << "find     : " << acc_find.count() << " [msec]" << std::endl;
        std::cout << "pivot    : " << acc_pivot.count() << " [msec]" << std::endl;

        cudaFree(d_converged);
        cudaFree(d_inout);
        cudaFree(d_buffer);
        cudaFree(d_in_buffer);
        cudaFree(d_out_buffer);
    }
}  // namespace solver
