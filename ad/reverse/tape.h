#include <vector>
#include <stdexcept>
#include <ostream>
#include <numbers>

#include <iostream>

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

/* Var concept */

template <typename T>
struct Varcheck : std::false_type {};

template <typename T>
struct Varcheck<Var<T> > : std::true_type {};

template <typename T>
concept isVar = Varcheck<T>::value;

/* Structs / enums */

enum Op {
	OP_NONE, // Used for input variables
	OP_ADD, // (Binary/Unary) Addition to a variable or constant
	OP_NEG, // (Unary) Negation
	OP_MUL, // (Binary) Multiplication of two variables
	OP_CMUL, // (Unary) Multiplication with a constant
	OP_DIV, // (Binary) Divsion of two variables
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

	TapeEntry() : operation(OP_NONE) {
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

	template <typename T2>
	Var<T> get_var(const T2 & x) {
		Var<T> out = get_var();

		out = x;

		return out;
	}

	void clear_derivatives() {
		for (std::size_t i = 0; i < size(); i++) {
			// Problem: if tape[i].grad is a Var, we have not initialized it earlier.
			// Its tape will be nullptr which results in a segfault.
			if constexpr (isVar<T>) {
				tape[i].grad = tape[i].value.tape->get_var();
			}
			tape[i].grad = 0;
		}
	}

	void compute_derivatives() {
		for (int i = static_cast<int>(size() - 1); i >= 0; i--) {
			switch (tape[i].operation) {
			case OP_NONE:
				break;
			case OP_ADD:
				for (std::size_t j = 0; j < tape[i].refs.size(); j++) {
					tape[tape[i].refs[j]].grad += tape[i].grad;
				}
				break;
			case OP_NEG:
				tape[tape[i].refs[0]].grad -= tape[i].grad;
				break;
			case OP_MUL:
				tape[tape[i].refs[0]].grad += tape[i].grad * tape[tape[i].refs[1]].value;
				tape[tape[i].refs[1]].grad += tape[i].grad * tape[tape[i].refs[0]].value;
				break;
			case OP_CMUL:
				tape[tape[i].refs[0]].grad += tape[i].grad * tape[i].value / tape[tape[i].refs[0]].value;
				break;
			case OP_DIV:
				tape[tape[i].refs[0]].grad += tape[i].grad / tape[tape[i].refs[1]].value;
				tape[tape[i].refs[1]].grad -= tape[i].grad * tape[tape[i].refs[0]].value / (tape[tape[i].refs[1]].value * tape[tape[i].refs[1]].value);
				break;
			case OP_CDIV:
				tape[tape[i].refs[0]].grad -= tape[i].grad * tape[i].value / tape[tape[i].refs[0]].value;
				break;
			case OP_EXP:
				tape[tape[i].refs[0]].grad += tape[i].value * tape[i].grad;
				break;
			case OP_LOG:
				tape[tape[i].refs[0]].grad += tape[i].grad / tape[tape[i].refs[0]].value;
				break;
			case OP_SIN:
				tape[tape[i].refs[0]].grad += tape[i].grad * cos(tape[tape[i].refs[0]].value);
				break;
			case OP_COS:
				tape[tape[i].refs[0]].grad -= tape[i].grad * sin(tape[tape[i].refs[0]].value);
				break;
			case OP_ASIN:
				tape[tape[i].refs[0]].grad += tape[i].grad / sqrt(1 - tape[tape[i].refs[0]].value * tape[tape[i].refs[0]].value);
				break;
			case OP_ATAN:
				tape[tape[i].refs[0]].grad += tape[i].grad / (1 + tape[tape[i].refs[0]].value * tape[tape[i].refs[0]].value);
				break;
			default:
				std::cout << i << " " << tape[i].operation << "\n";
			}
		}
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

	template <typename T2>
	void seed(const T2 & val) {
		grad() = val;
	}

	Var<T> & operator=(const Var<T> & val) {
		tape = val.tape;
		index = val.index;

		return *this;
	}

	template <typename T2>
	Var<T> & operator=(const T2 & val) {
		value() = val;

		tape->tape[index].refs.clear();
		tape->tape[index].operation = OP_NONE;

		return *this;
	}
};

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

template <typename T>
Var<T> sinh(const Var<T> & x);

template <typename T>
Var<T> cosh(const Var<T> & x);

template <typename T>
Var<T> tanh(const Var<T> & x);

template <typename T>
Var<T> asinh(const Var<T> & x);

template <typename T>
Var<T> acosh(const Var<T> & x);

template <typename T>
Var<T> atanh(const Var<T> & x);

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

template <typename T>
Var<T> sinh(const Var<T> & x) {
	return 0.5 * (exp(x) - exp(-x));
}

template <typename T>
Var<T> cosh(const Var<T> & x) {
	return 0.5 * (exp(x) + exp(-x));
}

template <typename T>
Var<T> tanh(const Var<T> & x) {
	return sinh(x) / cosh(x);
}

template <typename T>
Var<T> asinh(const Var<T> & x) {
	return log(x + sqrt(x * x + 1));
}

template <typename T>
Var<T> acosh(const Var<T> & x) {
	return log(x + sqrt(x * x - 1));
}

template <typename T>
Var<T> atanh(const Var<T> & x) {
	return 0.5 * log((1 + x) / (1 - x));
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
		throw std::runtime_error("Cannot divide variables from different tapes!");
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