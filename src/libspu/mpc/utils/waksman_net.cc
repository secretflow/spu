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

#include "libspu/mpc/utils/waksman_net.h"

#include "libspu/core/bit_utils.h"

namespace spu::mpc {
namespace internal {
/**
 * Return the number of (switch) columns in a AS-Waksman network for a given
 * number of packets.
 *
 * For example:
 * - as_waksman_num_columns(2) = 1,
 * - as_waksman_num_columns(3) = 3,
 * - as_waksman_num_columns(4) = 3,
 * and so on.
 */
inline size_t as_waksman_num_columns(size_t num_packets) {
  return (num_packets > 1 ? 2 * Log2Ceil(num_packets) - 1 : 0);
}

/**
 * Return the height of the AS-Waksman network's top sub-network.
 */
inline size_t as_waksman_top_height(size_t num_packets) {
  return num_packets / 2;
}

/**
 * Return the input wire of a left-hand side switch of an AS-Waksman network for
 * a given number of packets.
 *
 * A switch is specified by a row index row_idx, relative to a "row_offset" that
 * records the level of recursion. (The corresponding column index column_idx
 * can be inferred from row_offset and num_packets, and it is easier to reason
 * about implicitly.)
 *
 * If top = true, return the top wire, otherwise return bottom wire.
 */
inline size_t as_waksman_switch_output(size_t num_packets, size_t row_offset,
                                       size_t row_idx, bool use_top) {
  size_t relpos = row_idx - row_offset;
  SPU_ENFORCE(relpos % 2 == 0 && relpos + 1 < num_packets);
  return row_offset + (relpos / 2) +
         (use_top ? 0 : as_waksman_top_height(num_packets));
}

/**
 * Return the input wire of a right-hand side switch of an AS-Waksman network
 * for a given number of packets.
 *
 * This function is analogous to as_waksman_switch_output above.
 *
 * If top = true, return the top wire, otherwise return bottom wire.
 */
inline size_t as_waksman_switch_input(size_t num_packets, size_t row_offset,
                                      size_t row_idx, bool use_top) {
  /* Due to symmetry, this function equals as_waksman_switch_output. */
  return as_waksman_switch_output(num_packets, row_offset, row_idx, use_top);
}

/**
 * Given either a position occupied either by its top or bottom ports,
 * return the row index of its canonical position.
 *
 * This function is agnostic to column_idx, given row_offset, so we omit
 * column_idx.
 */
inline size_t as_waksman_get_canonical_row_idx(size_t row_offset,
                                               size_t row_idx) {
  /* translate back relative to row_offset, clear LSB, and then translate
   * forward */
  return ((row_idx - row_offset) & ~1) + row_offset;
}

/**
 *
 * Return a switch value that makes switch row_idx =
 * as_waksman_switch_position_from_wire_position(row_offset, packet_idx) to
 * route the wire packet_idx via the top (if top = true), resp.,
 * bottom (if top = false) subnetwork.
 *
 * NOTE: pos is assumed to be
 * - the input position for the LHS switches, and
 * - the output position for the RHS switches.
 */
inline bool as_waksman_get_switch_setting_from_top_bottom_decision(
    size_t row_offset, size_t packet_idx, bool use_top) {
  const size_t row_idx =
      as_waksman_get_canonical_row_idx(row_offset, packet_idx);
  return (packet_idx == row_idx) ^ use_top;
}

/**
 * Return true if the switch with input port at (column_idx, row_idx)
 * when set to "straight" (if top = true), resp., "cross" (if top =
 * false), routes the packet at (column_idx, row_idx) via the top
 * subnetwork.
 *
 * NOTE: packet_idx is assumed to be
 * - the input position for the RHS switches, and
 * - the output position for the LHS switches.
 */
inline bool as_waksman_get_top_bottom_decision_from_switch_setting(
    size_t row_offset, size_t packet_idx, bool switch_setting) {
  const size_t row_idx =
      as_waksman_get_canonical_row_idx(row_offset, packet_idx);
  return (row_idx == packet_idx) ^ switch_setting;
}

/**
 * Given an output wire of a RHS switch, compute and return the output
 * position of the other wire also connected to this switch.
 */
inline size_t as_waksman_other_output_position(size_t row_offset,
                                               size_t packet_idx) {
  const size_t row_idx =
      as_waksman_get_canonical_row_idx(row_offset, packet_idx);
  return (1 - (packet_idx - row_idx)) + row_idx;
}

/**
 * Given an input wire of a LHS switch, compute and return the input
 * position of the other wire also connected to this switch.
 */
inline size_t as_waksman_other_input_position(size_t row_offset,
                                              size_t packet_idx) {
  /* Due to symmetry, this function equals as_waksman_other_output_position. */
  return as_waksman_other_output_position(row_offset, packet_idx);
}

// TODO: use bfs may be efficient for memory, same as graph construction
/**
 * Compute AS-Waksman switch settings for the subnetwork occupying switch
 * columns [left,left+1,...,right] that will route
 * - from left-hand side inputs [lo,lo+1,...,hi]
 * - to right-hand side destinations pi[lo],pi[lo+1],...,pi[hi].
 *
 * The permutation
 * - pi maps [lo, lo+1, ... hi] to itself, offset by lo, and
 * - piinv is the inverse of pi.
 *
 * NOTE: due to offsets, neither pi or piinv are instances of
 * IntegerPermutation.
 */
void as_waksman_route_inner(size_t left,
                            size_t right,  // for column
                            PermEleType lo,
                            PermEleType hi,  // for packets
                            const IntegerPermutation& permutation,
                            const IntegerPermutation& permutation_inv,
                            AsWaksmanRouting& routing) {
  if (left > right) {
    return;
  }

  const size_t subnetwork_size = (hi - lo + 1);
  const size_t subnetwork_width = as_waksman_num_columns(subnetwork_size);
  const auto to_build_width = right - left + 1;

  SPU_ENFORCE(to_build_width >= subnetwork_width);

  if (to_build_width > subnetwork_width) {
    /**
     * If there is more space for the routing network than required,
     * then the topology for this subnetwork includes straight edges
     * along its sides and no switches, so it suffices to recurse.
     */
    as_waksman_route_inner(left + 1, right - 1, lo, hi, permutation,
                           permutation_inv, routing);
  } else if (subnetwork_size == 2) {
    /**
     * Non-trivial base case: switch settings for a 2-element permutation
     */
    SPU_ENFORCE(permutation[lo] == lo || permutation[lo] == lo + 1);
    SPU_ENFORCE(permutation[lo + 1] == lo || permutation[lo + 1] == lo + 1);
    SPU_ENFORCE(permutation[lo] != permutation[lo + 1]);

    routing[left][lo] = (permutation[lo] != lo);

  } else {
    /**
     * The algorithm first assigns a setting to a LHS switch,
     * route its target to RHS, which will enforce a RHS switch setting.
     * Then, it back-routes the RHS value back to LHS.
     * If this enforces a LHS switch setting, then forward-route that;
     * otherwise we will select the next value from LHS to route.
     */
    IntegerPermutation new_permutation(lo, hi);
    IntegerPermutation new_permutation_inv(lo, hi);
    std::vector<bool> lhs_routed(
        subnetwork_size, false); /* offset by lo, i.e. lhs_routed[packet_idx-lo]
                                    is set if packet packet_idx is routed */

    PermEleType to_route;      // next ele to route
    PermEleType max_unrouted;  // the maximum un-routed ele
    bool route_left;

    if (subnetwork_size % 2 == 1) {
      /**
       * ODD CASE: we first deal with the bottom-most straight wire,
       * which is not connected to any of the switches at this level
       * of recursion and just passed into the lower subnetwork.
       */
      if (permutation[hi] == hi) {
        /**
         * Easy sub-case: it is routed directly to the bottom-most
         * wire on RHS, so no switches need to be touched.
         */
        new_permutation[hi] = hi;
        new_permutation_inv[hi] = hi;
        to_route = hi - 1;
        route_left = true;
      } else {
        /**
         * Other sub-case: the straight wire is routed to a switch
         * on RHS, so route the other value from that switch
         * using the lower subnetwork.
         */
        const size_t rhs_switch =
            as_waksman_get_canonical_row_idx(lo, permutation[hi]);
        const bool rhs_switch_setting =
            as_waksman_get_switch_setting_from_top_bottom_decision(
                lo, permutation[hi], false);
        routing[right][rhs_switch] = rhs_switch_setting;

        size_t tprime =
            as_waksman_switch_input(subnetwork_size, lo, rhs_switch, false);

        new_permutation[hi] = tprime;
        new_permutation_inv[tprime] = hi;
        to_route = as_waksman_other_output_position(lo, permutation[hi]);
        route_left = false;
      }
      lhs_routed[hi - lo] = true;
      max_unrouted = hi - 1;
    } else {
      /**
       * EVEN CASE: the bottom-most switch is fixed to a constant
       * straight setting. So we route wire hi accordingly.
       *
       * Note: initialize only, route in other case
       */
      routing[left][hi - 1] = false;
      to_route = hi;
      route_left = true;
      max_unrouted = hi;
    }

    while (true) {
      /**
       * INVARIANT: the wire `to_route' on LHS (if route_left = true),
       * resp., RHS (if route_left = false) can be routed.
       */
      if (route_left) {
        /* If switch value has not been assigned, assign it arbitrarily. */
        const size_t lhs_switch =
            as_waksman_get_canonical_row_idx(lo, to_route);
        if (routing[left].find(lhs_switch) == routing[left].end()) {
          routing[left][lhs_switch] = false;
        }
        const bool lhs_switch_setting = routing[left][lhs_switch];
        const bool use_top =
            as_waksman_get_top_bottom_decision_from_switch_setting(
                lo, to_route, lhs_switch_setting);
        const size_t t =
            as_waksman_switch_output(subnetwork_size, lo, lhs_switch, use_top);
        if (permutation[to_route] == hi) {
          /**
           * We have routed to the straight wire for the odd case,
           * so now we back-route from it.
           */
          new_permutation[t] = hi;
          new_permutation_inv[hi] = t;
          lhs_routed[to_route - lo] = true;
          to_route = max_unrouted;
          route_left = true;
        } else {
          const size_t rhs_switch =
              as_waksman_get_canonical_row_idx(lo, permutation[to_route]);
          /**
           * We know that the corresponding switch on the right-hand side
           * cannot be set, so we set it according to the incoming wire.
           */
          assert(routing[right].find(rhs_switch) == routing[right].end());
          routing[right][rhs_switch] =
              as_waksman_get_switch_setting_from_top_bottom_decision(
                  lo, permutation[to_route], use_top);
          const size_t tprime =
              as_waksman_switch_input(subnetwork_size, lo, rhs_switch, use_top);
          new_permutation[t] = tprime;
          new_permutation_inv[tprime] = t;

          lhs_routed[to_route - lo] = true;
          to_route =
              as_waksman_other_output_position(lo, permutation[to_route]);
          route_left = false;
        }
      } else {
        /**
         * We have arrived on the right-hand side, so the switch setting is
         * fixed. Next, we back route from here.
         */
        const size_t rhs_switch =
            as_waksman_get_canonical_row_idx(lo, to_route);
        const size_t lhs_switch =
            as_waksman_get_canonical_row_idx(lo, permutation_inv[to_route]);
        assert(routing[right].find(rhs_switch) != routing[right].end());
        const bool rhs_switch_setting = routing[right][rhs_switch];
        const bool use_top =
            as_waksman_get_top_bottom_decision_from_switch_setting(
                lo, to_route, rhs_switch_setting);
        const bool lhs_switch_setting =
            as_waksman_get_switch_setting_from_top_bottom_decision(
                lo, permutation_inv[to_route], use_top);

        routing[left][lhs_switch] = lhs_switch_setting;

        const size_t t =
            as_waksman_switch_input(subnetwork_size, lo, rhs_switch, use_top);
        const size_t tprime =
            as_waksman_switch_output(subnetwork_size, lo, lhs_switch, use_top);
        new_permutation[tprime] = t;
        new_permutation_inv[t] = tprime;

        lhs_routed[permutation_inv[to_route] - lo] = true;
        to_route =
            as_waksman_other_input_position(lo, permutation_inv[to_route]);
        route_left = true;
      }

      /* If the next packet to be routed hasn't been routed before, then try
       * routing it. */
      if (!route_left || !lhs_routed[to_route - lo]) {
        continue;
      }

      /* Otherwise just find the next unrouted packet. */
      while (max_unrouted > lo && lhs_routed[max_unrouted - lo]) {
        --max_unrouted;
      }

      if (max_unrouted < lo ||
          (max_unrouted == lo &&
           lhs_routed[0])) /* lhs_routed[0] = corresponds to lo shifted by lo */
      {
        /* All routed! */
        break;
      } else {
        to_route = max_unrouted;
        route_left = true;
      }
    }

    if (subnetwork_size % 2 == 0) {
      /* Remove the AS-Waksman switch with the fixed value. */
      routing[left].erase(hi - 1);
    }

    const size_t d = as_waksman_top_height(subnetwork_size);
    const IntegerPermutation new_permutation_upper =
        new_permutation.slice(lo, lo + d - 1);
    const IntegerPermutation new_permutation_lower =
        new_permutation.slice(lo + d, hi);

    const IntegerPermutation new_permutation_inv_upper =
        new_permutation_inv.slice(lo, lo + d - 1);
    const IntegerPermutation new_permutation_inv_lower =
        new_permutation_inv.slice(lo + d, hi);

    as_waksman_route_inner(left + 1, right - 1, lo, lo + d - 1,
                           new_permutation_upper, new_permutation_inv_upper,
                           routing);
    as_waksman_route_inner(left + 1, right - 1, lo + d, hi,
                           new_permutation_lower, new_permutation_inv_lower,
                           routing);
  }
}

/**
 * Construct AS-Waksman subnetwork occupying switch columns
 *           [left,left+1, ..., right]
 * that will route
 * - from left-hand side inputs [lo,lo+1,...,hi]
 * - to right-hand side destinations
 * rhs_dests[0],rhs_dests[1],...,rhs_dests[hi-lo+1]. That is, rhs_dests are
 * 0-indexed w.r.t. row_offset of lo.
 *
 * Note that rhs_dests is *not* a permutation of [lo, lo+1, ... hi].
 *
 * This function fills out neighbors[left] and neighbors[right-1].
 */
void construct_as_waksman_inner(size_t left, size_t right, PermEleType lo,
                                PermEleType hi,
                                const std::vector<PermEleType>& rhs_dests,
                                AsWaksmanTopology& neighbors) {
  if (left > right) {
    return;
  }

  const size_t subnetwork_size = (hi - lo + 1);
  SPU_ENFORCE(rhs_dests.size() == subnetwork_size);
  const size_t subnetwork_width = as_waksman_num_columns(subnetwork_size);
  SPU_ENFORCE(right - left + 1 >= subnetwork_width);

  if (right - left + 1 > subnetwork_width) {
    /**
     * If there is more space for the routing network than needed,
     * just add straight edges. This also handles the size-1 base case.
     */
    for (PermEleType packet_idx = lo; packet_idx <= hi; ++packet_idx) {
      neighbors[left][packet_idx].first = neighbors[left][packet_idx].second =
          packet_idx;
      neighbors[right][packet_idx].first = neighbors[right][packet_idx].second =
          rhs_dests[packet_idx - lo];
    }

    std::vector<PermEleType> new_rhs_dests(subnetwork_size, -1);
    for (PermEleType packet_idx = lo; packet_idx <= hi; ++packet_idx) {
      new_rhs_dests[packet_idx - lo] = packet_idx;
    }

    construct_as_waksman_inner(left + 1, right - 1, lo, hi, new_rhs_dests,
                               neighbors);
  } else if (subnetwork_size == 2) {
    /* Non-trivial base case: routing a 2-element permutation. */
    neighbors[left][lo].first = neighbors[left][hi].second = rhs_dests[0];
    neighbors[left][lo].second = neighbors[left][hi].first = rhs_dests[1];
  } else {
    /**
     * Networks of size sz > 2 are handled by adding two columns of
     * switches alongside the network and recursing.
     */
    std::vector<PermEleType> new_rhs_dests(subnetwork_size, -1);

    /**
     * This adds floor(sz/2) switches alongside the network.
     *
     * As per the AS-Waksman construction, one of the switches in the
     * even case can be eliminated (i.e., set to a constant). We handle
     * this later.
     */
    for (PermEleType row_idx = lo;
         row_idx < (subnetwork_size % 2 == 1 ? hi : hi + 1); row_idx += 2) {
      neighbors[left][row_idx].first = neighbors[left][row_idx + 1].second =
          as_waksman_switch_output(subnetwork_size, lo, row_idx, true);
      neighbors[left][row_idx].second = neighbors[left][row_idx + 1].first =
          as_waksman_switch_output(subnetwork_size, lo, row_idx, false);

      new_rhs_dests[as_waksman_switch_input(subnetwork_size, lo, row_idx,
                                            true) -
                    lo] = row_idx;
      new_rhs_dests[as_waksman_switch_input(subnetwork_size, lo, row_idx,
                                            false) -
                    lo] = row_idx + 1;

      neighbors[right][row_idx].first = neighbors[right][row_idx + 1].second =
          rhs_dests[row_idx - lo];
      neighbors[right][row_idx].second = neighbors[right][row_idx + 1].first =
          rhs_dests[row_idx + 1 - lo];
    }

    if (subnetwork_size % 2 == 1) {
      /**
       * Odd special case:
       * the last wire is not connected to any switch,
       * and the wire is merely routed "straight".
       */
      neighbors[left][hi].first = neighbors[left][hi].second = hi;
      neighbors[right][hi].first = neighbors[right][hi].second =
          rhs_dests[hi - lo];
      new_rhs_dests[hi - lo] = hi;
    } else {
      /**
       * Even special case:
       * fix the bottom-most left-hand-side switch
       * to a constant "straight" setting.
       */
      neighbors[left][hi - 1].second = neighbors[left][hi - 1].first;
      neighbors[left][hi].second = neighbors[left][hi].first;
    }

    const size_t d = as_waksman_top_height(subnetwork_size);
    const std::vector<PermEleType> new_rhs_dests_top(new_rhs_dests.begin(),
                                                     new_rhs_dests.begin() + d);
    const std::vector<PermEleType> new_rhs_dests_bottom(
        new_rhs_dests.begin() + d, new_rhs_dests.end());

    construct_as_waksman_inner(left + 1, right - 1, lo, lo + d - 1,
                               new_rhs_dests_top, neighbors);
    construct_as_waksman_inner(left + 1, right - 1, lo + d, hi,
                               new_rhs_dests_bottom, neighbors);
  }
}

}  // namespace internal

AsWaksmanTopology generate_as_waksman_topology(size_t num_packets) {
  SPU_ENFORCE(num_packets > 1);
  const size_t width = internal::as_waksman_num_columns(num_packets);

  AsWaksmanTopology neighbors(
      width,
      std::vector<std::pair<PermEleType, PermEleType>>(
          num_packets, std::make_pair<PermEleType, PermEleType>(-1, -1)));

  // can use iota
  std::vector<PermEleType> rhs_dests(num_packets);
  std::iota(rhs_dests.begin(), rhs_dests.end(), 0);

  internal::construct_as_waksman_inner(0, width - 1,        // column
                                       0, num_packets - 1,  // row
                                       rhs_dests, neighbors);

  return neighbors;
}

AsWaksmanRouting get_as_waksman_routing(const IntegerPermutation& permutation) {
  const auto num_packets = permutation.size();
  const auto width = internal::as_waksman_num_columns(num_packets);

  AsWaksmanRouting routing(width);

  internal::as_waksman_route_inner(0, width - 1,        // column
                                   0, num_packets - 1,  // row
                                   permutation, permutation.inverse(),  // perm
                                   routing);
  return routing;
}

AsWaksmanRouting get_as_waksman_routing(const Index& permutation) {
  // copy the permutation here
  auto perm = IntegerPermutation(permutation);

  return get_as_waksman_routing(perm);
}

}  // namespace spu::mpc
