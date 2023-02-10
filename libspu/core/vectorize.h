// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "absl/types/span.h"

#include "libspu/core/prelude.h"

namespace spu {
namespace detail {

// Ref: https://en.cppreference.com/w/cpp/types/void_t
template <typename, typename = void>
constexpr bool is_container_like_v = false;

template <typename T>
constexpr bool
    is_container_like_v<T, std::void_t<decltype(std::declval<T>().begin()),
                                       decltype(std::declval<T>().end()),
                                       decltype(std::declval<T>().max_size()),
                                       decltype(std::declval<T>().size())>> =
        true;

}  // namespace detail

/// Simd trait.
//
// Simd means that we could operates a list of data together with one
// operation. Intuition is if we can pack a list of data together and unpack it
// later, then we can operates on the packed data, which reduce the number of
// ops.
//
// Formal definition, given:
//   a :: [T]
//   b :: [T]
//   f :: T -> T -> T
//
// calculate the following function with one f call.
//   simd_f :: [T] -> [T] -> (T -> T -> T) -> [T]
//
// so we can automatic do the simd if the data is packable/unpackable.
//   simd f = lambda x, y -> unpack(f(pack(x), pack(y)))
//
// This trait defines two methods, pack & unpack, if a type implements this
// trait, any operations on it could be automatically vectorized.
//
template <typename T, typename Enable = void>
struct SimdTrait {};

// Specification for container type.
// TODO(jint) NOT all container type could be packed and unpacked.
template <typename C>
struct SimdTrait<
    C, typename std::enable_if<detail::is_container_like_v<C>>::type> {
  using PackInfo = std::vector<size_t>;

  template <typename InputIt>
  static C pack(InputIt first, InputIt last, PackInfo& pi) {
    size_t total_size = 0;
    for (auto itr = first; itr != last; ++itr) {
      total_size += itr->size();
    }
    C result(total_size);
    for (auto itr = result.begin(); first != last; ++first) {
      itr = std::copy(first->begin(), first->end(), itr);
      pi.push_back(first->size());
    }
    return result;
  }

  template <typename OutputIt>
  static OutputIt unpack(const C& v, OutputIt result, const PackInfo& pi) {
    const size_t total_num =
        std::accumulate(pi.begin(), pi.end(), 0U, std::plus<>());

    SPU_ENFORCE(v.size() == total_num, "split number mismatch {} != {}",
                v.size(), total_num);

    size_t offset = 0;
    for (const auto& sz : pi) {
      C sub{};
      std::copy(v.begin() + offset, v.begin() + offset + sz,
                std::back_inserter(sub));
      offset += sz;
      *result++ = std::move(sub);
    }

    return result;
  }
};

// Apply https://en.cppreference.com/w/cpp/types/void_t trick to detect if a
// class has SimdTrait info.
template <class, class = void>
struct HasSimdTrait : std::false_type {};

template <class T>
struct HasSimdTrait<T, std::void_t<typename SimdTrait<T>::PackInfo>>
    : std::true_type {};

template <typename InputIt, typename OutputIt, typename UnaryOp>
OutputIt vectorize(InputIt first, InputIt last, OutputIt result, UnaryOp&& op) {
  using T = typename std::iterator_traits<InputIt>::value_type;
  using PackInfo = typename SimdTrait<T>::PackInfo;

  PackInfo pi;
  const T& joined = SimdTrait<T>::pack(first, last, pi);

  return SimdTrait<T>::unpack(op(joined), result, pi);
}

// another form of unary op vectorize
template <typename T, typename UnaryOp>
std::vector<T> vectorize(std::initializer_list<T> a, UnaryOp&& op) {
  std::vector<T> result;
  vectorize(a.begin(), a.end(), std::back_inserter(result),
            std::forward<UnaryOp>(op));
  return result;
}

template <typename InputIt, typename OutputIt, typename BinaryOp>
OutputIt vectorize(InputIt a_first, InputIt a_last, InputIt b_first,
                   InputIt b_last, OutputIt result, BinaryOp&& op) {
  using T = typename std::iterator_traits<InputIt>::value_type;
  using PackInfo = typename SimdTrait<T>::PackInfo;

  PackInfo a_pi{};
  PackInfo b_pi{};
  const T& a_joined = SimdTrait<T>::pack(a_first, a_last, a_pi);
  const T& b_joined = SimdTrait<T>::pack(b_first, b_last, b_pi);

  return SimdTrait<T>::unpack(op(a_joined, b_joined), result, a_pi);
}

// another form of binary op vectorize
template <typename T, typename BinaryOp>
std::vector<T> vectorize(std::initializer_list<T> a, std::initializer_list<T> b,
                         BinaryOp&& op) {
  std::vector<T> result;
  vectorize(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result),
            std::forward<BinaryOp>(op));
  return result;
}

// Given T as a type which could be processed using vectorization.
//
// This function reduce a list of objects with size n to one object with log(n)
// operations. As a comparison, normal ring reduce or non-vectorized tree reduce
// use n-1 operations.
//
// For example, a simple tree-reduce works like this:
//
//  a1 a2 a3 a4 a5 a6 a7 a8
//   \_|   \_|   \_|   \_|
//      \____|      \____|
//            \__________|
//
// For n(=8) element, we have n-1 operations and log(n) rounds.
//
//
// When the element has SimdTrait, elements could be vectorized
//
//  a1 a2 a3 a4 a5 a6 a7 a8
//   \_|   \_|   \_|   \_|     ; these 4 ops could be packed into one.
//      \____|      \____|     ; these 2 ops could be packed into one.
//            \__________|
//
// For n(=8) element, we have log(n) operations and log(n) rounds.
//
template <typename InputIt, typename BinaryFn,
          typename T = typename std::iterator_traits<InputIt>::value_type>
T vectorizedReduce(InputIt first, InputIt last, BinaryFn&& op) {
  size_t len = std::distance(first, last);

  std::vector<T> cur_level;
  while (len > 1) {
    const size_t half = len / 2;

    std::vector<T> next_level;
    vectorize(first, first + half,             // lhs
              first + half, first + 2 * half,  // rhs
              std::back_inserter(next_level), op);

    if (len % 2 == 1) {
      next_level.push_back(*(last - 1));
    }

    cur_level = std::move(next_level);

    first = cur_level.begin();
    last = cur_level.end();

    len = std::distance(first, last);
  }

  return *first;
}

}  // namespace spu
