// The libsnark library is developed by SCIPR Lab (http://scipr-lab.org)
// and contributors.

// Copyright (c) 2012-2014 SCIPR Lab and contributors (see AUTHORS file).

// All files, with the exceptions below, are released under the MIT License:

//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the "Software"),
//   to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:

//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.

//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.

#pragma once

#include "libspu/core/shape.h"

namespace spu::mpc {
//
// Some refs we refer to:
//  - Paper: https://inria.hal.science/inria-00072871/document
//  - Implementation:
//  https://github.com/scipr-lab/libsnark/blob/master/libsnark/common/routing_algorithms/as_waksman_routing_algorithm.hpp
//
//
// The following helper functions and comments are mainly borrowed from
// libsnark library with some modifications for both optimization in building
// routing setting and fitting in the SPU.

using PermEleType = int64_t;

class IntegerPermutation {
 private:
  Index contents_; /* offset by min_element_ */
  PermEleType min_element_;
  PermEleType max_element_;

 public:
  // Constructors
  explicit IntegerPermutation(size_t size = 0)
      : contents_(size),
        min_element_(0),
        max_element_(size > 0 ? size - 1 : 0) {
    std::iota(contents_.begin(), contents_.end(), 0);
  }

  IntegerPermutation(PermEleType min_element, PermEleType max_element)
      : contents_(max_element - min_element + 1),
        min_element_(min_element),
        max_element_(max_element) {
    std::iota(contents_.begin(), contents_.end(), min_element);
  }

  explicit IntegerPermutation(Index&& other_contents)
      : contents_(std::move(other_contents)),
        min_element_(0),
        max_element_(contents_.size() > 0 ? contents_.size() - 1 : 0) {}

  explicit IntegerPermutation(const Index& other_contents)
      : contents_(other_contents),
        min_element_(0),
        max_element_(other_contents.size() > 0 ? other_contents.size() - 1
                                               : 0) {}

  size_t size() const { return contents_.size(); }

  // Overload operator[]
  PermEleType operator[](size_t position) const {
    return contents_[position - min_element_];  // Read-only access
  }

  PermEleType& operator[](size_t position) {
    return contents_[position - min_element_];  // Read and write access
  }

  // Overload ==
  bool operator==(const IntegerPermutation& other) const {
    return (this->min_element_ == other.min_element_ &&
            this->max_element_ == other.max_element_ &&
            this->contents_ == other.contents_);
  }

  // slice method
  IntegerPermutation slice(PermEleType slice_min_element,
                           PermEleType slice_max_element) const {
    SPU_ENFORCE(slice_min_element >= min_element_ &&
                    slice_max_element <= max_element_ &&
                    slice_min_element <= slice_max_element,
                "invalid slice.");

    IntegerPermutation sub(slice_min_element, slice_max_element);
    std::copy(contents_.begin() + (slice_min_element - min_element_),
              contents_.begin() + (slice_max_element - min_element_ + 1),
              sub.contents_.begin());
    return sub;
  }

  // get inv perm
  IntegerPermutation inverse() const {
    IntegerPermutation inv(min_element_, max_element_);
    for (size_t i = 0; i < size(); ++i) {
      inv.contents_[contents_[i] - min_element_] = i + min_element_;
    }
    return inv;
  }

  ///
  /// debug only functions
  /// will just be used in unittest
  ///

  // test whether the permutation is valid
  bool is_valid() const {
    std::vector<bool> seen(size(), false);
    for (const auto& val : contents_) {
      if (val < min_element_ || val > max_element_ ||
          seen[val - min_element_]) {
        return false;
      }
      seen[val - min_element_] = true;
    }
    return true;
  }

  // return false if it is the last permutation
  bool next_permutation() {
    return std::next_permutation(contents_.begin(), contents_.end());
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  const IntegerPermutation& perm) {
    out << fmt::format("{}", fmt::join(perm.contents_, ","));
    return out;
  }
};

/**
 * When laid out on num_packets \times num_columns grid, each switch
 * occupies two positions: its top input and output ports are at
 * position (column_idx, row_idx) and the bottom input and output
 * ports are at position (column_idx, row_idx+1).
 *
 * We call the position assigned to the top ports of a switch its
 * "canonical" position.
 */

/**
 * A data structure that stores the topology of an AS-Waksman network.
 *
 * For a given column index column_idx and packet index packet_idx,
 * AsWaksmanTopology[column_idx][packet_idx] specifies the two
 * possible destinations at column_idx+1-th column where the
 * packet_idx-th packet in the column_idx-th column could be routed
 * after passing the switch, which has (column_idx, packet_idx) as one
 * of its occupied positions.
 *
 * This information is stored as a pair of indices, where:
 * - the first index denotes the destination when the switch is
 *   operated in "straight" setting, and
 * - the second index denotes the destination when the switch is
 *   operated in "cross" setting.
 *
 * If no switch occupies a position (column_idx, packet_idx),
 * i.e. there is just a wire passing through that position, then the
 * two indices are set to be equal and the packet is always routed to
 * the specified destination at the column_idx+1-th column.
 */
using AsWaksmanTopology =
    std::vector<std::vector<std::pair<PermEleType, PermEleType>>>;

/**
 * A routing assigns a bit to each switch in the AS-Waksman routing network.
 *
 * More precisely:
 *
 * - AsWaksmanRouting[column_idx][packet_idx]=false, if switch with
 *   canonical position of (column_idx,packet_idx) is set to
 *   "straight" setting, and
 *
 * - AsWaksmanRouting[column_idx][packet_idx]=true, if switch with
 *   canonical position of (column_idx,packet_idx) is set to "cross"
 *   setting.
 *
 * Note that AsWaksmanRouting[column_idx][packet_idx] does contain
 * entries for the positions associated with the bottom ports of the
 * switches, i.e. only canonical positions are present.
 */
// TODO: use bitset to cut down the memory usage?
using AsWaksmanRouting = std::vector<std::unordered_map<PermEleType, bool>>;

/**
 * Return the topology of an AS-Waksman network for a given number of packets.
 *
 * See AsWaksmanTopology (above) for details.
 */
AsWaksmanTopology generate_as_waksman_topology(size_t num_packets);

/**
 * Route the given permutation on an AS-Waksman network of suitable size.
 */
AsWaksmanRouting get_as_waksman_routing(const IntegerPermutation& permutation);

AsWaksmanRouting get_as_waksman_routing(const Index& permutation);

}  // namespace spu::mpc
