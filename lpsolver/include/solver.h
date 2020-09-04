#pragma once

#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace solver
{
    using Fractional = int2;

    __forceinline__ __host__ __device__ std::int32_t gcd(std::int32_t lhs, std::int32_t rhs)
    {
        lhs = lhs < 0 ? -lhs : lhs;
        rhs = rhs < 0 ? -rhs : rhs;
        // TODO: rhs or lhs == 0の時の挙動を考える
        while (lhs % rhs != 0) {
            const auto tmp = lhs % rhs;
            lhs = rhs;
            rhs = tmp;
        }
        return rhs;
    }
    __forceinline__ __host__ __device__ Fractional reduce(const Fractional& q)
    {
        if (q.x == 0) return {0, 1};  // ZERO
        std::int32_t g = gcd(q.x, q.y);
        if (q.y < 0)
            return make_int2(-q.x / g, -q.y / g);
        else
            return make_int2(q.x / g, q.y / g);
    }
    __forceinline__ __host__ __device__ Fractional add(const Fractional& lhs, const Fractional& rhs)
    {
        const auto ret = make_int2(lhs.x * rhs.y + lhs.y * rhs.x, lhs.y * rhs.y);
        return reduce(ret);
    }
    __forceinline__ __host__ __device__ Fractional sub(const Fractional& lhs, const Fractional& rhs)
    {
        const auto ret = make_int2(lhs.x * rhs.y - lhs.y * rhs.x, lhs.y * rhs.y);
        return reduce(ret);
    }
    __forceinline__ __host__ __device__ Fractional mul(const Fractional& lhs, const Fractional& rhs)
    {
        const auto ret = make_int2(lhs.x * rhs.x, lhs.y * rhs.y);
        return reduce(ret);
    }
    __forceinline__ __host__ __device__ Fractional div(const Fractional& lhs, const Fractional& rhs)
    {
        const auto ret = make_int2(lhs.x * rhs.y, lhs.y * rhs.x);
        return reduce(ret);
    }
    __forceinline__ __host__ __device__ Fractional negate(const Fractional& q) { return make_int2(-q.x, q.y); }
    __forceinline__ __host__ __device__ bool zero(const Fractional& q) { return q.x == 0; }
    __forceinline__ __host__ __device__ bool positive(const Fractional& q) { return q.x > 0; }
    __forceinline__ __host__ __device__ bool negative(const Fractional& q) { return q.x < 0; }
    __forceinline__ __host__ __device__ bool lt(const Fractional& lhs, const Fractional& rhs)
    {
        return lhs.x * rhs.y < rhs.x * lhs.y;
    }
    __forceinline__ __host__ __device__ bool gt(const Fractional& lhs, const Fractional& rhs)
    {
        return lhs.x * rhs.y > rhs.x * lhs.y;
    }
    __forceinline__ __host__ __device__ bool eq(const Fractional& lhs, const Fractional& rhs)
    {
        return lhs.x * rhs.y == rhs.x * lhs.y;
    }

    struct LinearProblem {
        std::uint32_t n;
        std::uint32_t m;
        Fractional* mat;
        Fractional* obj;
        Fractional* con;
        Fractional v;
        std::uint32_t* basis;
        std::uint32_t* non_basis;
    };

    // TODO: 収束判定を仕込む
    // TODO: 非有界の判定を仕込む
    __global__ void findExchangeIndexPre(const LinearProblem* __restrict__ lp, Fractional* buffer,
                                         std::uint32_t* in_buffer, std::uint32_t* out_buffer, std::uint8_t* converged,
                                         uint2* inout);
    __global__ void findExchangeIndexAft(const LinearProblem* __restrict__ lp, Fractional* buffer,
                                         std::uint32_t* in_buffer, std::uint32_t* out_buffer, std::uint8_t* converged,
                                         uint2* inout);
    __global__ void pivot(const LinearProblem* __restrict__ lp_old, LinearProblem* lp_new, const uint2* inout);

    class LPSolver
    {
    private:
        std::uint32_t n_;
        std::uint32_t m_;
        Fractional* d_mat1_;
        Fractional* d_mat2_;
        Fractional* d_obj1_;
        Fractional* d_obj2_;
        Fractional* d_con1_;
        Fractional* d_con2_;
        std::uint32_t* d_basis1_;
        std::uint32_t* d_basis2_;
        std::uint32_t* d_non_basis1_;
        std::uint32_t* d_non_basis2_;
        LinearProblem* d_lp1_;
        LinearProblem* d_lp2_;

        // output
        bool optimal_;
        Fractional value_;
        std::vector<Fractional> answer_;

    public:
        // 入力は標準系を想定している。(最大化問題)
        // n  : 項の数
        // m  : 制約の数
        // mat: m行n列の行列。要素は行志向に格納されている。
        // obj: 要素nの配列。目的関数。
        // con: 要素mの配列。制約部分の定数。
        // v  : 目的関数の定数部分。
        // maximize v + obj * x
        // subject to mat * x <= con AND x >= 0
        LPSolver(std::uint32_t n, std::uint32_t m, const Fractional* mat, const Fractional* obj, const Fractional* con,
                 Fractional v);
        ~LPSolver()
        {
            cudaFree(d_mat1_);
            cudaFree(d_mat2_);
            cudaFree(d_obj1_);
            cudaFree(d_obj2_);
            cudaFree(d_con1_);
            cudaFree(d_con2_);
            cudaFree(d_basis1_);
            cudaFree(d_basis2_);
            cudaFree(d_non_basis1_);
            cudaFree(d_non_basis2_);
            cudaFree(d_lp1_);
            cudaFree(d_lp2_);
        }
        void solve(std::chrono::milliseconds timeout);
        bool optimal() const { return optimal_; }
        Fractional value() const { return value_; };
        Fractional upper_bound() const;
        const Fractional* solution() const { return answer_.data(); }
    };
}  // namespace solver
