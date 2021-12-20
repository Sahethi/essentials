/**
 * @file launch_box.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-16
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/cuda/sm.hxx>

namespace gunrock {
namespace cuda {

namespace launch_box {
namespace detail {

/**
 * @brief Abstract base class for launch parameters.
 *
 * @tparam sm_flags_ Bitwise flags indicating SM versions (`sm_flag_t` enum).
 */
template <sm_flag_t sm_flags_>
struct launch_params_base_t {
  enum : unsigned { sm_flags = sm_flags_ };

 protected:
  launch_params_base_t() {}
};

/**
 * @brief False value dependent on template param so compiler can't optimize.
 *
 * @tparam T Arbitrary type.
 */
template <typename T>
struct always_false {
  enum { value = false };
};

/**
 * @brief Raises static assert when template is instantiated.
 *
 * @tparam T Arbitrary type.
 */
template <typename T>
struct raise_not_found_error_t {
  static_assert(always_false<T>::value,
                "Launch box could not find valid launch parameters");
};

// Macro for the flag of the current device's SM version
#define SM_TARGET_FLAG _SM_FLAG_WRAPPER(SM_TARGET)
// "ver" will be expanded before the call to _SM_FLAG
#define _SM_FLAG_WRAPPER(ver) _SM_FLAG(ver)
#define _SM_FLAG(ver) sm_##ver

/**
 * @brief Subsets a pack of launch parameters (children of
 * `launch_params_base_t`), selecting the ones that match the architecture being
 * compiled for, stored in a tuple type.
 *
 * @par Overview
 * This template alias is a tuple type of all the launch parameter types that
 * have `sm_flags` matching the current SM architecture being compiled for. It
 * uses the `tuple_cat()` funtion to concatenate tuples that are empty or
 * contain the launch parameter type if the SM version matches. The `lp_v` pack
 * is placed inside a `conditional_t`, which checks for a match and is then
 * expanded into the arguments of `tuple_cat()` using the `...` operator. This
 * was inspired by this Stack Overflow solution:
 * https://stackoverflow.com/a/67155114/13232647.
 *
 * @tparam lp_v Pack of `launch_params_t` types for each desired arch.
 */
template <typename... lp_v>
using match_launch_params_t = decltype(std::tuple_cat(
    std::declval<std::conditional_t<(bool)(lp_v::sm_flags& SM_TARGET_FLAG),
                                    std::tuple<lp_v>,
                                    std::tuple<>>>()...));

}  // namespace detail
}  // namespace launch_box

}  // namespace cuda
}  // namespace gunrock