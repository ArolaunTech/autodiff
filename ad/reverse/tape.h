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
template <typename T> struct Tape;
template <typename T> struct Var;

/* Forward concept */
template <typename T>
struct Varcheck : std::false_type {};

template <typename T>
struct Varcheck<Var<T> > : std::true_type {};

template <typename T>
concept isVar = Varcheck<T>::value;

/* Structs and enums */
enum Operation {
	OP_NONE,
	OP_ADD,
	OP_NEG,
	OP_MUL,
	OP_DIV,
	OP_EXP,
	OP_LOG,
	OP_SIN,
	OP_COS,
	OP_ASIN,
	OP_ATAN
};

template <typename T>
struct Tape {
	std::vector<Var<T> > tape;
	std::vector<T> grads;

	std::size_t size() const {
		return tape.size();
	}

	void register_var(Var<T> & x) {
		x.tape = this;
		x.index = size();

		x.op = OP_NONE;
		x.refs.clear();

		tape.push_back(x);
	}

	// Derivative calculation
	void clear_derivatives() {
		grads.clear();

		for (std::size_t i = 0; i < size(); i++) {
			grads.push_back(static_cast<T>(0));
		}
	}

	void compute_derivatives() {
		for (int i = size() - 1; i >= 0; i--) {
			switch (tape[i].op) {
			case OP_NONE:
				break;
			case OP_ADD:
				grads[tape[i].refs[0]] += grads[i];
				grads[tape[i].refs[1]] += grads[i];
				break;
			case OP_NEG:
				grads[tape[i].refs[0]] -= grads[i];
				break;
			case OP_MUL:
				grads[tape[i].refs[0]] += grads[i] * tape[tape[i].refs[1]].value;
				grads[tape[i].refs[1]] += grads[i] * tape[tape[i].refs[0]].value;
				break;
			case OP_DIV:
				grads[tape[i].refs[0]] += grads[i] / tape[tape[i].refs[1]].value;
				grads[tape[i].refs[1]] -= grads[i] * tape[i].value / tape[tape[i].refs[1]].value;
				break;
			case OP_EXP:
				grads[tape[i].refs[0]] += grads[i] * tape[i].value;
				break;
			case OP_LOG:
				grads[tape[i].refs[0]] += grads[i] / tape[tape[i].refs[0]].value;
				break;
			case OP_SIN:
				grads[tape[i].refs[0]] += grads[i] * cos(tape[tape[i].refs[0]].value);
				break;
			case OP_COS:
				grads[tape[i].refs[0]] -= grads[i] * sin(tape[tape[i].refs[0]].value);
				break;
			case OP_ASIN:
				grads[tape[i].refs[0]] += grads[i] / sqrt(1 - tape[tape[i].refs[0]].value * tape[tape[i].refs[0]].value);
				break;
			case OP_ATAN:
				grads[tape[i].refs[0]] += grads[i] / (1 + tape[tape[i].refs[0]].value * tape[tape[i].refs[0]].value);
				break;
			}
		}
	}
};

template <typename T>
struct Var {
	// Fields used for all variables.
	T value;

	// Only used for variables with a tape.
	Operation op;
	std::vector<std::size_t> refs;

	Tape<T> * tape;
	std::size_t index;

	// Constructors
	Var() : value(0), op(OP_NONE), tape(nullptr), index(0) {
		refs.clear();
	}

	template <typename T2>
	Var(const T2 & x) : op(OP_NONE), tape(nullptr), index(0) {
		value = static_cast<T>(x);
		refs.clear();
	}

	// Functions
	T & grad() const {
		return tape->grads[index];
	}

	// Derivative calculations
	template <typename T2>
	void seed(const T2 & x) const {
		grad() = x;
	}
};

/* cmath forward declations */
template <isVar T>
T abs(const T & x);

template <typename T1, typename T2, typename T3>
requires isVar<T1> || isVar<T2> || isVar<T3>
auto lerp(const T1 & a, const T2 & b, const T3 & t);

template <isVar T>
T exp(const T & x);

template <isVar T>
T exp2(const T & x);

template <isVar T>
T expm1(const T & x);

template <isVar T>
T log(const T & x);

template <isVar T>
T log10(const T & x);

template <isVar T>
T log2(const T & x);

template <isVar T>
T log1p(const T & x);

template <typename T1, typename T2>
requires isVar<T1> || isVar<T2>
auto pow(const T1 & x, const T2 & y);

template <isVar T>
T sqrt(const T & x);

template <isVar T>
T cbrt(const T & x);

template <typename T1, typename T2>
requires isVar<T1> || isVar<T2>
auto hypot(const T1 & x, const T2 & y);

template <typename T1, typename T2, typename T3>
requires isVar<T1> || isVar<T2> || isVar<T3>
auto hypot(const T1 & x, const T2 & y, const T3 & z);

template <isVar T>
T sin(const T & x);

template <isVar T>
T cos(const T & x);

template <isVar T>
T tan(const T & x);

template <isVar T>
T asin(const T & x);

template <isVar T>
T acos(const T & x);

template <isVar T>
T atan(const T & x);

template <isVar T>
T sinh(const T & x);

template <isVar T>
T cosh(const T & x);

template <isVar T>
T tanh(const T & x);

template <isVar T>
T asinh(const T & x);

template <isVar T>
T acosh(const T & x);

template <isVar T>
T atanh(const T & x);

/* cmath functions */
template <isVar T>
T abs(const T & x) {
	if (x > 0) return x;
	return -x;
}

template <typename T1, typename T2, typename T3>
requires isVar<T1> || isVar<T2> || isVar<T3>
auto lerp(const T1 & a, const T2 & b, const T3 & t) {
	return a + t * (b - a);
}

template <isVar T>
T exp(const T & x) {
	T out = exp(x.value);

	if (x.tape != nullptr) {
		out.op = OP_EXP;
		out.refs.push_back(x.index);
		out.tape = x.tape;
		out.index = x.tape->size();

		x.tape->tape.push_back(out);
	}

	return out;
}

template <isVar T>
T exp2(const T & x) {
	return pow(2.0, x);
}

template <isVar T>
T expm1(const T & x) {
	return exp(x) - 1;
}

template <isVar T>
T log(const T & x) {
	T out = log(x.value);

	if (x.tape != nullptr) {
		out.op = OP_LOG;
		out.refs.push_back(x.index);
		out.tape = x.tape;
		out.index = x.tape->size();

		x.tape->tape.push_back(out);
	}

	return out;
}

template <isVar T>
T log10(const T & x) {
	return log(x) / log(10.0);
}

template <isVar T>
T log2(const T & x) {
	return log(x) / log(2.0);
}

template <isVar T>
T log1p(const T & x) {
	return log(1 + x);
}

template <typename T1, typename T2>
requires isVar<T1> || isVar<T2>
auto pow(const T1 & x, const T2 & y) {
	return exp(log(x) * y);
}

template <isVar T>
T sqrt(const T & x) {
	return pow(x, 0.5);
}

template <isVar T>
T cbrt(const T & x) {
	return pow(x, 1.0 / 3.0);
}

template <typename T1, typename T2>
requires isVar<T1> || isVar<T2>
auto hypot(const T1 & x, const T2 & y) {
	return sqrt(x * x + y * y);
}

template <typename T1, typename T2, typename T3>
requires isVar<T1> || isVar<T2> || isVar<T3>
auto hypot(const T1 & x, const T2 & y, const T3 & z) {
	return sqrt(x * x + y * y + z * z);
}

template <isVar T>
T sin(const T & x) {
	T out = sin(x.value);

	if (x.tape != nullptr) {
		out.op = OP_SIN;
		out.refs.push_back(x.index);
		out.tape = x.tape;
		out.index = x.tape->size();

		x.tape->tape.push_back(out);
	}

	return out;
}

template <isVar T>
T cos(const T & x) {
	T out = cos(x.value);

	if (x.tape != nullptr) {
		out.op = OP_COS;
		out.refs.push_back(x.index);
		out.tape = x.tape;
		out.index = x.tape->size();

		x.tape->tape.push_back(out);
	}

	return out;
}

template <isVar T>
T tan(const T & x) {
	return sin(x) / cos(x);
}

template <isVar T>
T asin(const T & x) {
	T out = asin(x.value);

	if (x.tape != nullptr) {
		out.op = OP_ASIN;
		out.refs.push_back(x.index);
		out.tape = x.tape;
		out.index = x.tape->size();

		x.tape->tape.push_back(out);
	}

	return out;
}

template <isVar T>
T acos(const T & x) {
	return 0.5 * std::numbers::pi - asin(x);
}

template <isVar T>
T atan(const T & x) {
	T out = atan(x.value);

	if (x.tape != nullptr) {
		out.op = OP_ATAN;
		out.refs.push_back(x.index);
		out.tape = x.tape;
		out.index = x.tape->size();

		x.tape->tape.push_back(out);
	}

	return out;
}

template <isVar T>
T sinh(const T & x) {
	return 0.5 * (exp(x) - exp(-x));
}

template <isVar T>
T cosh(const T & x) {
	return 0.5 * (exp(x) + exp(-x));
}

template <isVar T>
T tanh(const T & x) {
	return sinh(x) / cosh(x);
}

template <isVar T>
T asinh(const T & x) {
	return log(x + sqrt(x * x + 1));
}

template <isVar T>
T acosh(const T & x) {
	return log(x + sqrt(x * x - 1));
}

template <isVar T>
T atanh(const T & x) {
	return 0.5 * log((1 + x) / (1 - x));
}

/* Var operators */

// Addition
template <typename T>
Var<T> operator+(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape == nullptr && rhs.tape == nullptr) {
		Var<T> out;
		out.value = lhs.value + rhs.value;
		return out;
	}
	if (lhs.tape == nullptr) return rhs + lhs;
	if (rhs.tape != nullptr && rhs.tape != lhs.tape) {
		throw std::runtime_error("Cannot add/subtract variables from different tapes!");
	}

	Var<T> out;
	out.value = lhs.value + rhs.value;
	out.op = OP_ADD;
	out.tape = lhs.tape;
	out.refs.push_back(lhs.index);

	if (rhs.tape == nullptr) {
		lhs.tape->tape.push_back(rhs);
		out.refs.push_back(lhs.tape->size() - 1);
	} else {
		out.refs.push_back(rhs.index);
	}

	out.index = lhs.tape->size();

	lhs.tape->tape.push_back(out);
	return out;
}

template <typename T, Numeric T2>
Var<T> operator+(const Var<T> & lhs, const T2 & rhs) {
	Var<T> rhsvar = rhs;
	return lhs + rhsvar;
}

template <typename T, Numeric T2>
Var<T> operator+(const T2 & lhs, const Var<T> & rhs) {
	Var<T> lhsvar = lhs;
	return lhsvar + rhs;
}

// Unary plus
template <isVar T>
T & operator+(T & val) {
	return val;
}

// Addition assignment
template <isVar T, typename T2>
T & operator+=(T & lhs, const T2 & rhs) {
	lhs = lhs + rhs;
	return lhs;
}

// Prefix increment
template <isVar T>
T & operator++(T & val) {
	val += 1;
	return val;
}

// Postfix increment
template <isVar T>
T operator++(T & val, int) {
	T copy = val;
	val += 1;
	return copy;
}

// Subtraction
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
auto operator-(const T & lhs, const T2 & rhs) {
	return lhs + (-rhs);
}

// Unary negation
template <isVar T>
T operator-(const T & val) {
	T out = -val.value;

	if (val.tape != nullptr) {
		out.op = OP_NEG;
		out.refs.push_back(val.index);
		out.tape = val.tape;
		out.index = val.tape->size();

		val.tape->tape.push_back(out);
	}

	return out;
}

// Subtraction assignment
template <isVar T, typename T2>
T & operator-=(T & lhs, const T2 & rhs) {
	lhs = lhs - rhs;
	return lhs;
}

// Prefix decrement
template <isVar T>
T & operator--(T & val) {
	val -= 1;
	return val;
}

// Postfix decrement
template <isVar T>
T operator--(T & val, int) {
	T copy = val;
	val -= 1;
	return copy;
}

// Multiplication
template <typename T>
Var<T> operator*(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape == nullptr && rhs.tape == nullptr) {
		Var<T> out;
		out.value = lhs.value * rhs.value;
		return out;
	}
	if (lhs.tape == nullptr) return rhs * lhs;
	if (rhs.tape != nullptr && rhs.tape != lhs.tape) {
		throw std::runtime_error("Cannot multiply variables from different tapes!");
	}

	Var<T> out;
	out.value = lhs.value * rhs.value;
	out.op = OP_MUL;
	out.tape = lhs.tape;
	out.refs.push_back(lhs.index);

	if (rhs.tape == nullptr) {
		lhs.tape->tape.push_back(rhs);
		out.refs.push_back(lhs.tape->size() - 1);
	} else {
		out.refs.push_back(rhs.index);
	}

	out.index = lhs.tape->size();

	lhs.tape->tape.push_back(out);
	return out;
}

template <typename T, Numeric T2>
Var<T> operator*(const Var<T> & lhs, const T2 & rhs) {
	Var<T> rhsvar = rhs;
	return lhs * rhsvar;
}

template <typename T, Numeric T2>
Var<T> operator*(const T2 & lhs, const Var<T> & rhs) {
	Var<T> lhsvar = lhs;
	return lhsvar * rhs;
}

// Multiplication assignment
template <isVar T, typename T2>
T & operator*=(T & lhs, const T2 & rhs) {
	lhs = lhs * rhs;
	return lhs;
}

// Division
template <typename T>
Var<T> operator/(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape == nullptr && rhs.tape == nullptr) {
		Var<T> out;
		out.value = lhs.value / rhs.value;
		return out;
	}
	if (rhs.tape == nullptr) return lhs * (1 / rhs);
	if (lhs.tape != nullptr && lhs.tape != rhs.tape) {
		throw std::runtime_error("Cannot divide variables from different tapes!");
	}

	Var<T> out;
	out.value = lhs.value / rhs.value;
	out.op = OP_DIV;
	out.tape = rhs.tape;

	if (lhs.tape == nullptr) {
		rhs.tape->tape.push_back(lhs);
		out.refs.push_back(rhs.tape->size() - 1);
	} else {
		out.refs.push_back(lhs.index);
	}

	out.refs.push_back(rhs.index);
	out.index = rhs.tape->size();

	rhs.tape->tape.push_back(out);

	return out;
}

template <typename T, Numeric T2>
Var<T> operator/(const Var<T> & lhs, const T2 & rhs) {
	Var<T> rhsvar = rhs;
	return lhs / rhsvar;
}

template <typename T, Numeric T2>
Var<T> operator/(const T2 & lhs, const Var<T> & rhs) {
	Var<T> lhsvar = lhs;
	return lhsvar / rhs;
}

// Division assignment
template <isVar T, typename T2>
T & operator/=(T & lhs, const T2 & rhs) {
	lhs = lhs / rhs;
	return lhs;
}

// Equality
template <isVar T, typename T2>
bool operator==(const T & lhs, const T2 & rhs) {
	return lhs.value == rhs;
}

template <typename T, isVar T2>
bool operator==(const T & lhs, const T2 & rhs) {
	return lhs == rhs.value;
}

// Inequality
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator!=(const T & lhs, const T2 & rhs) {
	return !(lhs == rhs);
}

// Less than
template <isVar T, typename T2>
bool operator<(const T & lhs, const T2 & rhs) {
	return lhs.value < rhs;
}

template <typename T, isVar T2>
bool operator<(const T & lhs, const T2 & rhs) {
	return lhs < rhs.value;
}

// Less than or equal
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator<=(const T & lhs, const T2 & rhs) {
	return (lhs < rhs) || (lhs == rhs);
}

// Greater than
template <typename T, typename T2>
requires isVar<T> || isVar<T2>
bool operator>(const T & lhs, const T2 & rhs) {
	return !(lhs <= rhs);
}

// Greater than or equal
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
template <isVar T>
bool operator!(const T & x) {
	return !(x.value);
}

// Stream insertion
template <isVar T>
std::ostream & operator<<(std::ostream & os, const T & x) {
	os << x.value;
	return os;
}

/* Default types */
using tape = Tape<double>;
using var = Var<double>;

}

#endif