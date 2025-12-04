#include "libspu/kernel/hal/brent_kung_network.h"

#include <vector>
#include <cmath>

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

// NoteFunc with input g2
static std::pair<Value, Value> NoteFunc(SPUContext* ctx, 
                                 const Value& p1, const Value& p2, 
                                 const Value& g1, const Value& g2) {
    // g3 = g1 * g2
    auto g3 = hal::mul(ctx, g1, g2);

    // p3 = p1 + (p2 - p1) * g1
    auto diff = hal::sub(ctx, p2, p1);
    
    Value g1_broadcasted = g1;
    if (g1.shape() != diff.shape()) {
        g1_broadcasted = hal::broadcast_to(ctx, g1, diff.shape());
    }

    auto term = hal::mul(ctx, diff, g1_broadcasted);
    auto p3 = hal::add(ctx, p1, term);

    return std::make_pair(p3, g3);
}

// NoteFunc without input g2
static Value NoteFunc(SPUContext* ctx, 
               const Value& p1, const Value& p2, 
               const Value& g1) {
    auto diff = hal::sub(ctx, p2, p1);

    Value g1_broadcasted = g1;
    if (g1.shape() != diff.shape()) {
        g1_broadcasted = hal::broadcast_to(ctx, g1, diff.shape());
    }
    
    auto term = hal::mul(ctx, diff, g1_broadcasted);
    auto p3 = hal::add(ctx, p1, term);
    return p3;
}

Value AggregateBrentKung(SPUContext* ctx, const Value& x_full, const Value& g_full) {
    const int64_t n = x_full.shape()[0];
    const int64_t block_size = x_full.shape()[1];
    const int64_t logn = std::floor(std::log2(n));

    std::vector<Value> x_rows(n);
    std::vector<Value> g_rows(n);
    
    for(int i=0; i<n; ++i) {
        x_rows[i] = hal::slice(ctx, x_full, {i, 0}, {i+1, block_size}, {});
        g_rows[i] = hal::slice(ctx, g_full, {i, 0}, {i+1, 1}, {});
        
        x_rows[i] = hal::reshape(ctx, x_rows[i], {block_size});
        g_rows[i] = hal::reshape(ctx, g_rows[i], {1});
    }

    std::vector<std::vector<Value>> p_tree(n, std::vector<Value>(logn));
    std::vector<std::vector<Value>> g_tree(n, std::vector<Value>(logn));
    std::vector<Value> res(n);

    // --- Up-Sweep ---
    for (int i = 1; i < n; i += 2) {
        auto result = NoteFunc(ctx, x_rows[i], x_rows[i-1], g_rows[i], g_rows[i-1]);
        p_tree[i][0] = result.first;
        g_tree[i][0] = result.second;
    }

    for (int j = 1; j < logn; ++j) {
        int step = 1 << (j + 1);
        for (int i = step - 1; i < n; i += step) {
            int prev_idx = i - (1 << j);
            auto result = NoteFunc(ctx, 
                                   p_tree[i][j-1], p_tree[prev_idx][j-1], 
                                   g_tree[i][j-1], g_tree[prev_idx][j-1]);
            p_tree[i][j] = result.first;
            g_tree[i][j] = result.second;
        }
    }

    // --- Down-Sweep ---
    res[0] = x_rows[0];
    
    for (int j = 0; j < logn; ++j) {
        int idx = (1 << (j + 1)) - 1;
        if (idx < n) {
            res[idx] = p_tree[idx][j];
        }
    }

    for (int j = logn - 3; j >= 0; --j) {
        int step = 1 << (j + 2);
        int half_step = 1 << (j + 1);
        
        for (int k = 1; k < n / step + 1; ++k) {
            int idx_curr = n - half_step * (2 * k - 1) - 1;
            int idx_prev = n - half_step * 2 * k - 1;

            if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
                res[idx_curr] = NoteFunc(ctx, 
                                         p_tree[idx_curr][j], 
                                         res[idx_prev], 
                                         g_tree[idx_curr][j]);
            }
        }
    }

    for (int k = 1; k < n / 2 + 1; ++k) {
        int idx_curr = n - 2 * k;
        int idx_prev = n - 2 * k - 1;
        
        if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
             res[idx_curr] = NoteFunc(ctx, 
                                      x_rows[idx_curr], 
                                      res[idx_prev], 
                                      g_rows[idx_curr]);
        }
    }

    // --- Reshape & Concatenate ---
    std::vector<Value> res_reshaped;
    for(auto& v : res) {
        res_reshaped.push_back(hal::reshape(ctx, v, {1, block_size}));
    }
    
    return hal::concatenate(ctx, res_reshaped, 0);
}

} // namespace spu::kernel::hal