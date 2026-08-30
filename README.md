# [insert name here]

I don't have a good name for this yet. 

An autodifferentiation library in C++. This is designed to be super simple to include in your project and also be fairly low-level. That does lead to some monstrosities like this:

```cpp
template <typename T>
T f(T x) {
	return x / 2 + 5 * exp(x) / sin(x);
}

int main() {
	// Two reverse autodiffs

	ad::tape tape2;
	ad::var xin = 5;
	tape2.register_var(xin);

	ad::Tape<ad::var> tape;
	ad::Var<ad::var> x = xin;
	tape.register_var(x);

	ad::Var<ad::var> y = f(x);

	tape.clear_derivatives();
	y.seed(1);
	tape.compute_derivatives();

	tape2.clear_derivatives();
	x.grad().seed(1);
	tape2.compute_derivatives();

	std::cout << y << "\n";
	std::cout << x.grad() << "\n";
	std::cout << xin.grad() << "\n";

	// Two forward autodiffs

	ad::Fwd<double, 2> xfwd = 5;
	xfwd.grads[1] = 1;

	ad::Fwd<double, 2> yfwd = f(xfwd);

	std::cout << yfwd << "\n";
	std::cout << yfwd.derivative(1) << "\n";
	std::cout << yfwd.derivative(2) << "\n";
}
```

(getting the second derivative of a function) but it also lets you handle computing things like Hessians yourself.

 - **Requires C++20**

## Installation

You can install this like a normal header-only library (no instructions provided) or you can just include the `ad` folder in your project and include the `ad/ad.h` file when you want to do autodifferentiation.