#include <vector>
#include <stdexcept>

#ifndef TAPE_H
#define TAPE_H

namespace ad {

/* Forward declarations */

template <typename T> struct TapeEntry;
template <typename T> struct Tape;
template <typename T> struct Var;

/* Structs / enums */

enum Op {
	OP_NONE, // Used for input variables
	OP_ADD,
	OP_NEG,
	OP_MUL,
	OP_RECIPROCAL
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

	Var<T> get_var() {
		Var<T> out;
		out.tape = this;
		out.index = tape.size();

		tape.push_back(TapeEntry<T>());

		return out;
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

/* Operators */

template <typename T>
Var<T> operator+(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape != rhs.tape) {
		throw std::runtime_error("Cannot add variables from different tapes!");
	}

	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() + rhs.value();

	out.tape->tape[out.index].refs.push_back(lhs.index);
	out.tape->tape[out.index].refs.push_back(rhs.index);

	out.tape->tape[out.index].operation = OP_ADD;

	return out;
}

template <typename T>
Var<T> operator+(const Var<T> & lhs, const T & rhs) {
	Var<T> x = lhs.tape->get_var();
	x = rhs;

	return lhs + x;
}

template <typename T>
Var<T> operator+(const T & lhs, const Var<T> & rhs) {
	return rhs + lhs;
}

template <typename T>
Var<T> & operator+(Var<T> & val) {
	return val;
}

template <typename T, typename T2>
Var<T> & operator+=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs + rhs;
	return lhs;
}

template <typename T>
Var<T> & operator++(Var<T> & val) {
	val += T(1);
	return val;
}

template <typename T>
Var<T> operator++(Var<T> & val, int) {
	Var<T> copy = val;
	++val;
	return copy;
}

template <typename T, typename T2>
Var<T> operator-(const Var<T> & lhs, const T2 & rhs) {
	return lhs + (-rhs);
}

template <typename T, typename T2>
Var<T> operator-(const T2 & lhs, const Var<T> & rhs) {
	return lhs + (-rhs);
}

template <typename T>
Var<T> operator-(const Var<T> & val) {
	Var<T> out = val.tape->get_var();

	out.value() = -val.value();
	out.tape->tape[out.index].refs.push_back(val.index);
	out.tape->tape[out.index].operation = OP_NEG;

	return out;
}

template <typename T, typename T2>
Var<T> & operator-=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs - rhs;
	return lhs;
}

template <typename T>
Var<T> & operator--(Var<T> & val) {
	val -= T(1);
	return val;
}

template <typename T>
Var<T> operator--(Var<T> & val, int) {
	Var<T> copy = val;
	--val;
	return copy;
}

template <typename T>
Var<T> operator*(const Var<T> & lhs, const Var<T> & rhs) {
	if (lhs.tape != rhs.tape) {
		throw std::runtime_error("Cannot add variables from different tapes!");
	}

	Var<T> out = lhs.tape->get_var();
	out.value() = lhs.value() * rhs.value();

	out.tape->tape[out.index].refs.push_back(lhs.index);
	out.tape->tape[out.index].refs.push_back(rhs.index);

	out.tape->tape[out.index].operation = OP_MUL;

	return out;
}

template <typename T>
Var<T> operator*(const Var<T> & lhs, const T & rhs) {
	Var<T> x = lhs.tape->get_var();
	x = rhs;

	return lhs * x;
}

template <typename T>
Var<T> operator*(const T & lhs, const Var<T> & rhs) {
	return rhs * lhs;
}

template <typename T, typename T2>
Var<T> & operator*=(Var<T> & lhs, const T2 & rhs) {
	lhs = lhs * rhs;
	return lhs;
}



/*

operator/ division
free function -> T operator/( T const & lhs, T const & rhs )

operator/= division assignment
free function ->  T & operator/=( T & lhs, T const & rhs )

operator== equality
free function -> bool operator==( T const & lhs, T const & rhs )

operator!= or operator not_eq inequality
free function -> bool operator!=( T const & lhs, T const & rhs )

operator< less than
free function -> bool operator<( T const & lhs, T const & rhs )

operator<= less than or equal
free function -> bool operator<=( T const & lhs, T const & rhs )

operator> greater than
free function -> bool operator>( T const & lhs, T const & rhs )

operator>= greater than or equal
free function -> bool operator>=( T const & lhs, T const & rhs )

operator&& or operator and logical and
free function -> bool operator&&( T const & lhs, T const & rhs )

operator|| or operator or logical or
free function -> bool operator||( T const & lhs, T const & rhs )

operator! or operator not logical not
free function -> bool operator!( T const & value ) const

operator( ) cast
member function -> From a type T -> operator U( ) const

operator<< stream insertion
free function -> std::ostream & operator<<( std::ostream & os, T const & value )

*/

/* Default types */
using tape = Tape<double>;
using var = Var<double>;

}

#endif