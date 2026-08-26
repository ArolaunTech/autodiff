#include <vector>
#include <stdexcept>
#include <ostream>
#include <numbers>

#include "../common/concepts.h"

#ifndef TAPE_H
#define TAPE_H

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

/* Forward declarations */

template <typename T> struct TapeEntry;
template <typename T> struct Tape;
template <typename T> struct Var;

/* Structs / enums */

enum Op {
	OP_NONE, // Used for input variables
	OP_ADD, // (Binary) Addition to a variable or constant
	OP_NEG, // (Unary) Negation
	OP_MUL, // (Binary) Multiplication of two variables
	OP_CMUL, // (Unary) Multiplication with a constant
	OP_DIV, // (Binary) Division of two variables
	OP_CDIV, // (Unary) Constant divided by a variable
	OP_EXP, // (Unary) e^x
	OP_LOG, // (Unary) Natural logarithm
	OP_SIN, // (Unary) Sine
	OP_COS, // (Unary) Cosine
	OP_ASIN, // (Unary) Arcsine
	OP_ATAN // (Unary) Arctangent
};

template <typename T>
struct TapeEntry {
	T value;
	T grad;

	std::vector<std::size_t> refs;
	Op operation;

	TapeEntry() : value(0), grad(0), operation(OP_NONE) {
		refs.clear();
	}
};

template <typename T>
struct Tape {
	std::vector<TapeEntry<T> > tape;

	std::size_t size() {
		return tape.size();
	}

	Var<T> get_var() {
		Var<T> out;
		out.tape = this;
		out.index = tape.size();

		tape.push_back(TapeEntry<T>());

		return out;
	}

	TapeEntry<T> & operator[](std::size_t pos) {
		return tape[pos];
	}
};

template <typename T>
struct Var {
	Tape<T>* tape;
	std::size_t index;

	T & value() const {
		return tape->tape[index].value;
	}

	T & grad() const {
		return tape->tape[index].grad;
	}

	Var<T> & operator=(const T & val) {
		value() = val;
		grad() = 0;

		tape->tape[index].refs.clear();
		tape->tape[index].operation = OP_NONE;

		return *this;
	}
};

/* Var concept */

template <typename T>
struct Varcheck : std::false_type {};

template <typename T>
struct Varcheck<Var<T> > : std::true_type {};

template <typename T>
concept isVar = Varcheck<T>::value;

/* cmath forward declarations */

template <typename T, typename T2, typename T3>
requires isVar<T> || isVar<T2> || isVar<T3>
auto lerp(const T & a, const T2 & b, const T3 & t);

template <typename T>
Var<T> exp(const Var<T> & x);

template <typename T>
Var<T> exp2(const Var<T> & x);

template <typename T>
Var<T> expm1(const Var<T> & x);

template <typename T>
Var<T> log(const Var<T> & x);

template <typename T>
Var<T> log10(const Var<T> & x);

template <typename T>
Var<T> log2(const Var<T> & x);

template <typename T, typename T2>
requires isVar<T> || isVar<T2>
auto pow(const T & x, const T2 & y);

template <typename T>
Var<T> sqrt(const Var<T> & x);

template <typename T>
Var<T> cbrt(const Var<T> & x);

template <typename T, typename T2>
requires isVar<T> || isVar<T2>
auto hypot(const T & x, const T2 & y);

template <typename T, typename T2, typename T3>
requires isVar<T> || isVar<T2> || isVar<T3>
auto hypot(const T & x, const T2 & y, const T3 & z);

template <typename T>
Var<T> sin(const Var<T> & x);

template <typename T>
Var<T> cos(const Var<T> & x);

template <typename T>
Var<T> tan(const Var<T> & x);

template <typename T>
Var<T> asin(const Var<T> & x);

template <typename T>
Var<T> acos(const Var<T> & x);

template <typename T>
Var<T> atan(const Var<T> & x);

/* cmath functions */

template <typename T, typename T2, typename T3>
requires isVar<T> || isVar<T2> || isVar<T3>
auto lerp(const T & a, const T2 & b, const T3 & t) {
	return a + t * (b - a);
}

template <typename T>
Var<T> exp(const Var<T> & x) {
	Var<T> out = x.tape->get_var();

	out.value() = exp(x.value());
	out.tape->tape[out.index].refs.push_back(x.index);
	out.tape->tape[out.index].operation = OP_EXP;

	return out;
}

template <typename T>
Var<T> exp2(const Var<T> & x) {
	return pow(2.0, x);
}

template <typename T>
Var<T> expm1(const Var<T> & x) {
	return exp(x) - 1.0;
}

template <typename T>
Var<T> log(const Var<T> & x) {
	Var<T> out = x.tape->get_var();

	out.value() = log(x.value());
	out.tape->tape[out.index].refs.push_back(x.index);
	out.tape->tape[out.index].operation = OP_LOG;

	return out;
}

template <typename T>
Var<T> log10(const Var<T> & x) {
	return log(x) / log(10.0);
}

template <typename T>
Var<T> log2(const Var<T> & x) {
	return log(x) / log(2.0);
}

template <typename T>
Var<T> log1p(const Var<T> & x) {
	return log(1.0 + x);
}

template <typename T, typename T2>
requires isVar<T> || isVar<T2>
auto pow(const T & x, const T2 & y) {
	return exp(log(x) * y);
}

template <typename T>
Var<T> sqrt(const Var<T> & x) {
	return pow(x, 0.5);
}

template <typename T>
Var<T> cbrt(const Var<T> & x) {
	return pow(x, 1.0/3.0);
}

template <typename T, typename T2>
requires isVar<T> || isVar<T2>
auto hypot(const T & x, const T2 & y) {
	return sqrt(x * x + y * y);
}

template <typename T, typename T2, typename T3>
requires isVar<T> || isVar<T2> || isVar<T3>
auto hypot(const T & x, const T2 & y, const T3 & z) {
	return sqrt(x * x + y * y + z * z);
}

template <typename T>
Var<T> sin(const Var<T> & x) {
	Var<T> out = x.tape->get_var();

	out.value() = sin(x.value());
	out.tape->tape[out.index].refs.push_back(x.index);
	out.tape->tape[out.index].operation = OP_SIN;

	return out;
}

template <typename T>
Var<T> cos(const Var<T> & x) {
	Var<T> out = x.tape->get_var();

	out.value() = cos(x.value());
	out.tape->tape[out.index].refs.push_back(x.index);
	out.tape->tape[out.index].operation = OP_COS;

	return out;
}

template <typename T>
Var<T> tan(const Var<T> & x) {
	return sin(x) / cos(x);
}

template <typename T>
Var<T> asin(const Var<T> & x) {
	Var<T> out = x.tape->get_var();

	out.value() = asin(x.value());
	out.tape->tape[out.index].refs.push_back(x.index);
	out.tape->tape[out.index].operation = OP_ASIN;

	return out;
}

template <typename T>
Var<T> acos(const Var<T> & x) {
	return 0.5 * std::numbers::pi - asin(x);
}

template <typename T>
Var<T> atan(const Var<T> & x) {
	Var<T> out = x.tape->get_var();

	out.value() = atan(x.value());
	out.tape->tape[out.index].refs.push_back(x.index);
	out.tape->tape[out.index].operation = OP_ATAN;

	return out;
}

/* Operators */

// Addition
template <typename T>
Var<T> operator+(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape != rhs.tape) {
		throw std::runtime_error("Cannot add/subtract variables from different tapes!");
	}

	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() + rhs.value();

	out.tape->tape[out.index].refs.push_back(lhs.index);
	out.tape->tape[out.index].refs.push_back(rhs.index);

	out.tape->tape[out.index].operation = OP_ADD;

	return out;
}

template <typename T, Numeric T2>
Var<T> operator+(const Var<T> & lhs, const T2 & rhs) {
	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() + rhs;

	out.tape->tape[out.index].refs.push_back(lhs.index);

	out.tape->tape[out.index].operation = OP_ADD;

	return out;
}

template <typename T, Numeric T2>
Var<T> operator+(const T2 & lhs, const Var<T> & rhs) {
	return rhs + lhs;
}

// Unary plus
template <typename T>
Var<T> & operator+(Var<T> & val) {
	return val;
}

// Addition assignment
template <typename T, typename T2>
Var<T> & operator+=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs + rhs;
	return lhs;
}

// Prefix increment
template <typename T>
Var<T> & operator++(Var<T> & val) {
	val += 1;
	return val;
}

// Postfix increment
template <typename T>
Var<T> operator++(Var<T> & val, int) {
	Var<T> copy = val;
	++val;
	return copy;
}

// Subtraction
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
auto operator-(const T & lhs, const T2 & rhs) {
	return lhs + (-rhs);
}

// Unary negation
template <typename T>
Var<T> operator-(const Var<T> & val) {
	Var<T> out = val.tape->get_var();

	out.value() = -val.value();
	out.tape->tape[out.index].refs.push_back(val.index);
	out.tape->tape[out.index].operation = OP_NEG;

	return out;
}

// Subtraction assignment
template <typename T, typename T2>
Var<T> & operator-=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs - rhs;
	return lhs;
}

// Prefix decrement
template <typename T>
Var<T> & operator--(Var<T> & val) {
	val -= 1;
	return val;
}

// Postfix decrement
template <typename T>
Var<T> operator--(Var<T> & val, int) {
	Var<T> copy = val;
	--val;
	return copy;
}

// Multiplication
template <typename T>
Var<T> operator*(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape != rhs.tape) {
		throw std::runtime_error("Cannot multiply variables from different tapes!");
	}

	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() * rhs.value();

	out.tape->tape[out.index].refs.push_back(lhs.index);
	out.tape->tape[out.index].refs.push_back(rhs.index);

	out.tape->tape[out.index].operation = OP_MUL;

	return out;
}

template <typename T, Numeric T2>
Var<T> operator*(const Var<T> & lhs, const T2 & rhs) {
	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() * rhs;

	out.tape->tape[out.index].refs.push_back(lhs.index);

	out.tape->tape[out.index].operation = OP_CMUL;

	return out;
}

template <typename T, Numeric T2>
Var<T> operator*(const T2 & lhs, const Var<T> & rhs) {
	return rhs * lhs;
}

// Multiplication assignment
template <typename T, typename T2>
Var<T> & operator*=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs * rhs;
	return lhs;
}

// Division
template <typename T>
Var<T> operator/(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape != rhs.tape) {
		throw std::runtime_error("Cannot multiply variables from different tapes!");
	}

	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() / rhs.value();

	out.tape->tape[out.index].refs.push_back(lhs.index);
	out.tape->tape[out.index].refs.push_back(rhs.index);

	out.tape->tape[out.index].operation = OP_DIV;

	return out;
}

template <typename T, Numeric T2>
Var<T> operator/(const Var<T> & lhs, const T2 & rhs) {
	return lhs * (1.0 / rhs);
}

template <typename T, Numeric T2>
Var<T> operator/(const T2 & lhs, const Var<T> & rhs) {
	Var<T> out = lhs.tape->get_var();
	out.value() = lhs / rhs.value();

	out.tape->tape[out.index].refs.push_back(rhs.index);

	out.tape->tape[out.index].operation = OP_CDIV;

	return out;
}

// Division assignment
template <typename T, typename T2>
Var<T> & operator/=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs / rhs;
	return lhs;
}

// Equality
template <typename T, typename T2>
bool operator==(const Var<T> & lhs, const T2 & rhs) {
	return lhs.value() == rhs;
}

template <typename T, typename T2>
bool operator==(const T2 & lhs, const Var<T> & rhs) {
	return lhs == rhs.value();
}

// Inequality
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator!=(const T & lhs, const T2 & rhs) {
	return !(lhs == rhs);
}

// Less than
template <typename T, typename T2>
bool operator<(const Var<T> & lhs, const T2 & rhs) {
	return lhs.value() < rhs;
}

template <typename T, typename T2>
bool operator<(const T2 & lhs, const Var<T> & rhs) {
	return lhs < rhs.value();
}

// Less than or equal to
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator<=(const T & lhs, const T2 & rhs) {
	return !(lhs > rhs);
}

// Greater than
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator>(const T & lhs, const T2 & rhs) {
	return rhs < lhs;
}

// Greater than or equal to
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator>=(const T & lhs, const T2 & rhs) {
	return !(lhs < rhs);
}

// Logical and
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator&&(const T & lhs, const T2 & rhs) {
	return !((!lhs) || (!rhs));
}

// Logical or
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator||(const T & lhs, const T2 & rhs) {
	return !((!lhs) && (!rhs));
}

// Logical not
template <typename T>
bool operator!(const Var<T> & val) {
	return !(val.value());
}

// Stream insertion
template <typename T>
std::ostream & operator<<(std::ostream & os, const Var<T> & val) {
	os << val.value();
	return os;
}

/* Default types */
using tape = Tape<double>;
using var = Var<double>;

}

#endif