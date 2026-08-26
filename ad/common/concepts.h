#include <concepts>
#include <type_traits>

#ifndef CONCEPTS_H
#define CONCEPTS_H

/* Value concept */
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

#endif