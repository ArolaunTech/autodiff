#include <array>
#include <limits>
#include <cmath>
#include <numbers>
#include <concepts>
#include <type_traits>

#include "../common/factorial.h"

#ifndef DUAL_H
#define DUAL_H

namespace ad {

/* cmath */
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::exp;
using std::log10;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;
using std::cosh;
using std::sinh;
using std::tanh;
using std::erf;
using std::hypot;

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

	template <typename T2>
	Fwd(T2 val) {
		grads.fill(0);
		grads[0] = static_cast<T>(val);
	}

	Fwd(const std::array<T, N + 1> & ngrads): grads(ngrads) {}

	/* funcs */
	T derivative(std::size_t order) {
		return grads[order] * factorial(order);
	}
};

/* Forward concept */
template <typename T>
struct Fwdcheck : std::false_type {};

template <typename T, std::size_t N>
struct Fwdcheck<Fwd<T, N> > : std::true_type {};

template <typename T>
concept isFwd = Fwdcheck<T>::value;

/* Forward declarations */

template <typename T, std::size_t N>
Fwd<T, N> derivative(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> integral(const T & x0, const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> reciprocal(const Fwd<T, N> & x);

template <typename A, typename B, typename T>
requires isFwd<A> || isFwd<B> || isFwd<T>
auto lerp(const A & a, const B & b, const T & t);

template <typename T, std::size_t N>
Fwd<T, N> exp(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> exp2(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> expm1(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> log(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> log10(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> log2(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> log1p(const Fwd<T, N> & x);

template <typename A, typename B>
requires isFwd<A> || isFwd<B>
auto pow(const A & base, const B & exponent);

template <typename T, std::size_t N>
Fwd<T, N> sqrt(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> cbrt(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> hypot(const Fwd<T, N> & x, const Fwd<T, N> & y);
template <typename T, std::size_t N>
Fwd<T, N> hypot(const Fwd<T, N> & x, const Fwd<T, N> & y, const Fwd<T, N> & z);

template <typename T, std::size_t N>
std::array<Fwd<T, N>, 2> sincos(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> sin(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> cos(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> tan(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> asin(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> acos(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> atan(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> atan2(const Fwd<T, N> & y, const Fwd<T, N> & x);
template <typename T, std::size_t N>
Fwd<T, N> atan2(const T & y, const Fwd<T, N> & x);
template <typename T, std::size_t N>
Fwd<T, N> atan2(const Fwd<T, N> & y, const T & x);

template <typename T, std::size_t N>
Fwd<T, N> sinh(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> cosh(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> tanh(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> asinh(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> acosh(const Fwd<T, N> & x);

template <typename T, std::size_t N>
Fwd<T, N> atanh(const Fwd<T, N> & x);

/* Fwd funcs */

template <typename T, std::size_t N>
Fwd<T, N> derivative(const Fwd<T, N> & x) {
	Fwd<T, N> out;

	for (std::size_t i = 1; i <= N; i++) {
		out.grads[i - 1] = i * x.grads[i];
	}

	return out;
}

template <typename T, std::size_t N>
Fwd<T, N> integral(const T & x0, const Fwd<T, N> & x) {
	Fwd<T, N> out;

	out.grads[0] = x0;

	for (std::size_t i = 1; i <= N; i++) {
		out.grads[i] = x.grads[i - 1] / i;
	}

	return out;
}

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

template <typename A, typename B, typename T>
requires isFwd<A> || isFwd<B> || isFwd<T>
auto lerp(const A & a, const B & b, const T & t) {
	return a + (b - a) * t;
}

template <typename T, std::size_t N>
Fwd<T, N> exp(const Fwd<T, N> & x) {
	Fwd<T, N> out;

	out.grads[0] = exp(x.grads[0]);

	for (std::size_t i = 1; i <= N; i++) {
		for (std::size_t j = 1; j <= i; j++) {
			out.grads[i] += j * x.grads[j] * out.grads[i - j];
		}
		out.grads[i] /= i;
	}

	return out;
}

template <typename T, std::size_t N>
Fwd<T, N> exp2(const Fwd<T, N> & x) {
	return pow(T(2), x);
}

template <typename T, std::size_t N>
Fwd<T, N> expm1(const Fwd<T, N> & x) {
	return exp(x) - T(1);
}

template <typename T, std::size_t N>
Fwd<T, N> log(const Fwd<T, N> & x) {
	return integral(log(x.grads[0]), derivative(x) * reciprocal(x));
}

template <typename T, std::size_t N>
Fwd<T, N> log10(const Fwd<T, N> & x) {
	const T invlog10 = 1 / log(10);

	return invlog10 * log(x);
}

template <typename T, std::size_t N>
Fwd<T, N> log2(const Fwd<T, N> & x) {
	const T invlog2 = 1 / log(2);

	return invlog2 * log(x);
}

template <typename T, std::size_t N>
Fwd<T, N> log1p(const Fwd<T, N> & x) {
	return log(T(1) + x);
}

template <typename A, typename B>
requires isFwd<A> || isFwd<B>
auto pow(const A & base, const B & exponent) {
	return exp(log(base) * exponent);
}

template <typename T, std::size_t N>
Fwd<T, N> sqrt(const Fwd<T, N> & x) {
	return pow(x, T(0.5));
}

template <typename T, std::size_t N>
Fwd<T, N> cbrt(const Fwd<T, N> & x) {
	return pow(x, T(1.0 / 3.0));
}

template <typename T, std::size_t N>
Fwd<T, N> hypot(const Fwd<T, N> & x, const Fwd<T, N> & y) {
	return sqrt(x * x + y * y);
}

template <typename T, std::size_t N>
Fwd<T, N> hypot(const Fwd<T, N> & x, const Fwd<T, N> & y, const Fwd<T, N> & z) {
	return sqrt(x * x + y * y + z * z);
}

template <typename T, std::size_t N>
std::array<Fwd<T, N>, 2> sincos(const Fwd<T, N> & x) {
	std::array<Fwd<T, N>, 2> out;

	// out[0] == sin(x), out[1] == cos(x)

	out[0].grads[0] = sin(x.grads[0]);
	out[1].grads[0] = cos(x.grads[0]);

	for (std::size_t i = 1; i <= N; i++) {
		for (std::size_t j = 1; j <= i; j++) {
			out[0].grads[i] += j * x.grads[j] * out[1].grads[i - j];
			out[1].grads[i] -= j * x.grads[j] * out[0].grads[i - j];
		}
		out[0].grads[i] /= i;
		out[1].grads[i] /= i;
	}

	return out;
}

template <typename T, std::size_t N>
Fwd<T, N> sin(const Fwd<T, N> & x) {
	return sincos(x)[0];
}

template <typename T, std::size_t N>
Fwd<T, N> cos(const Fwd<T, N> & x) {
	return sincos(x)[1];
}

template <typename T, std::size_t N>
Fwd<T, N> tan(const Fwd<T, N> & x) {
	std::array<Fwd<T, N>, 2> sc = sincos(x);

	return sc[0] / sc[1];
}

template <typename T, std::size_t N>
Fwd<T, N> asin(const Fwd<T, N> & x) {
	return integral(asin(x.grads[0]), derivative(x) * reciprocal(sqrt(T(1) - x * x)));
}

template <typename T, std::size_t N>
Fwd<T, N> acos(const Fwd<T, N> & x) {
	const T HALF_PI = T(0.5 * std::numbers::pi);

	return HALF_PI - asin(x);
}

template <typename T, std::size_t N>
Fwd<T, N> atan(const Fwd<T, N> & x) {
	return integral(atan(x.grads[0]), derivative(x) * reciprocal(T(1) + x * x));
}

template <typename T, std::size_t N>
Fwd<T, N> atan2(const Fwd<T, N> & y, const Fwd<T, N> & x) {
	return integral(
		atan2(y.grads[0], x.grads[0]), 
		(x * derivative(y) - y * derivative(x)) / (x * x + y * y)
	);
}

template <typename T, std::size_t N>
Fwd<T, N> atan2(const T & y, const Fwd<T, N> & x) {
	return atan2(Fwd<T, N>(y), x);
}

template <typename T, std::size_t N>
Fwd<T, N> atan2(const Fwd<T, N> & y, const T & x) {
	return atan2(y, Fwd<T, N>(x));
}

template <typename T, std::size_t N>
Fwd<T, N> sinh(const Fwd<T, N> & x) {
	return T(0.5) * (exp(x) - exp(-x));
}

template <typename T, std::size_t N>
Fwd<T, N> cosh(const Fwd<T, N> & x) {
	return T(0.5) * (exp(x) + exp(-x));
}

template <typename T, std::size_t N>
Fwd<T, N> tanh(const Fwd<T, N> & x) {
	return sinh(x) / cosh(x);
}

template <typename T, std::size_t N>
Fwd<T, N> asinh(const Fwd<T, N> & x) {
	return log(x + sqrt(x * x + T(1)));
}

template <typename T, std::size_t N>
Fwd<T, N> acosh(const Fwd<T, N> & x) {
	return log(x + sqrt(x * x - T(1)));
}

template <typename T, std::size_t N>
Fwd<T, N> atanh(const Fwd<T, N> & x) {
	return T(0.5) * log((T(1) + x) / (T(1) - x));
}

/* Fwd operators */

// Addition
template <typename T, std::size_t N>
Fwd<T, N> operator+(const Fwd<T, N> & lhs, const Fwd<T, N> & rhs) {
	Fwd<T, N> out;

	for (std::size_t i = 0; i <= N; i++) {
		out.grads[i] = lhs.grads[i] + rhs.grads[i];
	}

	return out;
}
template <typename T, std::size_t N>
Fwd<T, N> operator+(const T & lhs, const Fwd<T, N> & rhs) {
	return rhs + lhs;
}

template <typename T, std::size_t N>
Fwd<T, N> operator+(const Fwd<T, N> & lhs, const T & rhs) {
	return lhs + Fwd<T, N>(rhs);
}

// Unary plus
template <typename T, std::size_t N>
Fwd<T, N> & operator+(Fwd<T, N> & val) {
	return val;
}

// Addition assignment
template <typename T, typename T2, std::size_t N>
Fwd<T, N> & operator+=(Fwd<T, N> & lhs, const T2 & rhs) {
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
template <typename T, std::size_t N>
Fwd<T, N> operator-(const Fwd<T, N> & lhs, const Fwd<T, N> & rhs) {
	return lhs + (-rhs);
}

template <typename T, std::size_t N>
Fwd<T, N> operator-(const T & lhs, const Fwd<T, N> & rhs) {
	return lhs + (-rhs);
}

template <typename T, std::size_t N>
Fwd<T, N> operator-(const Fwd<T, N> & lhs, const T & rhs) {
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
template <typename T, typename T2, std::size_t N>
Fwd<T, N> & operator-=(Fwd<T, N> & lhs, const T2 & rhs) {
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
template <typename T, std::size_t N>
Fwd<T, N> operator*(const Fwd<T, N> & lhs, const Fwd<T, N> & rhs) {
	Fwd<T, N> out;

	for (std::size_t i = 0; i <= N; i++) {
		for (std::size_t j = 0; i + j <= N; j++) {
			out.grads[i + j] += lhs.grads[i] * rhs.grads[j];
		}
	}

	return out;
}

template <typename T, std::size_t N>
Fwd<T, N> operator*(const T & lhs, const Fwd<T, N> & rhs) {
	return rhs * lhs;
}

template <typename T, std::size_t N>
Fwd<T, N> operator*(const Fwd<T, N> & lhs, const T & rhs) {
	return lhs * Fwd<T, N>(rhs);
}

// Multiplication assignment
template<typename T, typename T2, std::size_t N>
Fwd<T, N> & operator*=(Fwd<T, N> & lhs, const T2 & rhs) {
	lhs = lhs * rhs;
	return lhs;
}

// Division
template <typename T, std::size_t N>
Fwd<T, N> operator/(const Fwd<T, N> & lhs, const Fwd<T, N> & rhs) {
	return lhs * reciprocal(rhs);
}

template <typename T, std::size_t N>
Fwd<T, N> operator/(const T & lhs, const Fwd<T, N> & rhs) {
	return lhs * reciprocal(rhs);
}

template <typename T, std::size_t N>
Fwd<T, N> operator/(const Fwd<T, N> & lhs, const T & rhs) {
	return lhs * (1 / rhs);
}

// Division assignment
template<typename T, typename T2, std::size_t N>
Fwd<T, N> & operator/=(Fwd<T, N> & lhs, const T2 & rhs) {
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

/* Default dual type */
using fwd = Fwd<double, 1>;

} // ad

#endif