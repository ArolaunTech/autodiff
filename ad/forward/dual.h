#include <array>
#include <limits>
#include <cmath>

#include "../common/factorial.h"

#ifndef DUAL_H
#define DUAL_H

namespace ad {

/* Fwd */
template <typename T, std::size_t N>
struct Fwd {
	static_assert(
		N != std::numeric_limits<std::size_t>::max(), 
		"Decrease the number of derivatives being taken."
	);

	std::array<T, N + 1> grads;

	/* constructors */
	Fwd() {
		grads.fill(0);
	}

	Fwd(T val) {
		grads.fill(0);
		grads[0] = val;
	}

	Fwd(const std::array<T, N + 1> & ngrads): grads(ngrads) {}

	/* funcs */
	T derivative(std::size_t order) {
		return grads[order] * factorial(order);
	}
};

/* Fwd funcs */

template <typename T, std::size_t N>
Fwd<T, N> reciprocal(const Fwd<T, N> & x) {
	Fwd<T, N> out;

	out.grads[0] = 1 / x.grads[0];

	for (std::size_t i = 1; i <= N; i++) {
		for (std::size_t j = 1; j <= i; j++) {
			out.grads[i] -= x.grads[j] * out.grads[i - j];
		}
		out.grads[i] /= x.grads[0];
	}

	return out;
}

template <typename T, std::size_t N>
Fwd<T, N> sqrt(const Fwd<T, N> & x) {
	Fwd<T, N> out;

	out.grads[0] = std::sqrt(x.grads[0]);

	for (std::size_t i = 1; i <= N; i++) {
		out.grads[i] = x.grads[i];

		for (std::size_t j = 1; j < i; j++) {
			out.grads[i] -= out.grads[j] * out.grads[i - j];
		}

		out.grads[i] /= 2 * out.grads[0];
	}

	return out;
}

template <typename T, std::size_t N>
Fwd<T, N> exp(const Fwd<T, N> & x) {
	Fwd<T, N> out;

	out.grads[0] = std::exp(x.grads[0]);

	for (std::size_t i = 1; i <= N; i++) {
		for (std::size_t j = 1; j <= i; j++) {
			out.grads[i] += j * x.grads[j] * out.grads[i - j];
		}
		out.grads[i] /= i;
	}

	return out;
}

/* Fwd operators */

// Addition
template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator+(const Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	Fwd<T1, N> out;

	for (std::size_t i = 0; i <= N; i++) {
		out.grads[i] = lhs.grads[i] + rhs.grads[i];
	}

	return out;
}

template <typename T1, typename T2, std::size_t N>
Fwd<T2, N> operator+(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return rhs + lhs;
}

template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator+(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs + Fwd<T2, N>(rhs);
}

// Unary plus
template <typename T, std::size_t N>
Fwd<T, N> & operator+(Fwd<T, N> & val) {
	return val;
}

// Addition assignment
template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator+=(Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	lhs = lhs + rhs;
	return lhs;
}

template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator+=(Fwd<T1, N> & lhs, const T2 & rhs) {
	lhs = lhs + rhs;
	return lhs;
}

// Prefix increment
template <typename T, std::size_t N>
Fwd<T, N> & operator++(Fwd<T, N> & val) {
	val += 1;
	return val;
}

// Postfix increment
template <typename T, std::size_t N>
Fwd<T, N> operator++(Fwd<T, N> & val, int) {
	Fwd<T, N> copy = val;
	++val;
	return copy;
}

// Subtraction
template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator-(const Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	return lhs + (-rhs);
}

template <typename T1, typename T2, std::size_t N>
Fwd<T2, N> operator-(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs + (-rhs);
}

template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator-(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs + (-rhs);
}

// Unary negation
template <typename T, std::size_t N>
Fwd<T, N> operator-(const Fwd<T, N> & val) {
	Fwd<T, N> copy = val;

	for (std::size_t i = 0; i <= N; i++) {
		copy.grads[i] = -copy.grads[i];
	}
	return copy;
}

// Subtraction assignment
template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator-=(Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	lhs = lhs - rhs;
	return lhs;
}

template <typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator-=(Fwd<T1, N> & lhs, const T2 & rhs) {
	lhs = lhs - rhs;
	return lhs;
}

// Prefix decrement
template <typename T, std::size_t N>
Fwd<T, N> & operator--(Fwd<T, N> & val) {
	val-= 1;
	return val;
}

// Postfix decrement
template <typename T, std::size_t N>
Fwd<T, N> operator--(Fwd<T, N> & val, int) {
	Fwd<T, N> copy = val;
	--val;
	return copy;
}

// Multiplication
template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator*(const Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	Fwd<T1, N> out;

	for (std::size_t i = 0; i <= N; i++) {
		for (std::size_t j = 0; i + j <= N; j++) {
			out.grads[i + j] += lhs.grads[i] * rhs.grads[j];
		}
	}

	return out;
}

template<typename T1, typename T2, std::size_t N>
Fwd<T2, N> operator*(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return rhs * lhs;
}

template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator*(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs * Fwd<T2, N>(rhs);
}

// Multiplication assignment
template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator*=(Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	lhs = lhs * rhs;
	return lhs;
}

template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator*=(Fwd<T1, N> & lhs, const T2 & rhs) {
	lhs = lhs * rhs;
	return lhs;
}

// Division
template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator/(const Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	return lhs * reciprocal(rhs);
}

template<typename T1, typename T2, std::size_t N>
Fwd<T2, N> operator/(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs * reciprocal(rhs);
}

template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> operator/(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs * (1 / rhs);
}

// Division assignment
template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator/=(Fwd<T1, N> & lhs, const Fwd<T2, N> & rhs) {
	lhs = lhs / rhs;
	return lhs;
}

template<typename T1, typename T2, std::size_t N>
Fwd<T1, N> & operator/=(Fwd<T1, N> & lhs, const T2 & rhs) {
	lhs = lhs / rhs;
	return lhs;
}

// Equality
template<typename T1, typename T2, std::size_t N>
bool operator==(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs.grads[0] == rhs;
}

template<typename T1, typename T2, std::size_t N>
bool operator==(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs == rhs.grads[0];
}

// Inequality
template<typename T1, typename T2, std::size_t N>
bool operator!=(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return !(lhs == rhs);
}

template<typename T1, typename T2, std::size_t N>
bool operator!=(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return !(lhs == rhs);
}

// Less than
template<typename T1, typename T2, std::size_t N>
bool operator<(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs.grads[0] < rhs;
}

template<typename T1, typename T2, std::size_t N>
bool operator<(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs < rhs.grads[0];
}

// Less than or equal
template<typename T1, typename T2, std::size_t N>
bool operator<=(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs.grads[0] <= rhs;
}

template<typename T1, typename T2, std::size_t N>
bool operator<=(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs <= rhs.grads[0];
}

// Greater than
template<typename T1, typename T2, std::size_t N>
bool operator>(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs.grads[0] > rhs;
}

template<typename T1, typename T2, std::size_t N>
bool operator>(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs > rhs.grads[0];
}

// Greater than or equal
template<typename T1, typename T2, std::size_t N>
bool operator>=(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return lhs.grads[0] >= rhs;
}

template<typename T1, typename T2, std::size_t N>
bool operator>=(const T1 & lhs, const Fwd<T2, N> & rhs) {
	return lhs >= rhs.grads[0];
}

// Logical and
template<typename T1, typename T2, std::size_t N>
bool operator&&(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return !((!lhs) || (!rhs));
}

template<typename T1, typename T2, std::size_t N>
bool operator&&(const T2 & lhs, const Fwd<T1, N> & rhs) {
	return !((!lhs) || (!rhs));
}

// Logical or
template<typename T1, typename T2, std::size_t N>
bool operator||(const Fwd<T1, N> & lhs, const T2 & rhs) {
	return !((!lhs) && (!rhs));
}

template<typename T1, typename T2, std::size_t N>
bool operator||(const T2 & lhs, const Fwd<T1, N> & rhs) {
	return !((!lhs) && (!rhs));
}

// Logical not
template<typename T, std::size_t N>
bool operator!(const Fwd<T, N> & val) {
	return !(val.grads[0]);
}

// Stream insertion


/*
free function -> std::ostream & operator<<( std::ostream & os, T const & value )
*/

/* Default dual type */
using fwd = Fwd<double, 1>;

} // ad

#endif