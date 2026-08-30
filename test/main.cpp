#include <iostream>

#include <ad/ad.h>

ad::var f(ad::var x) {
	ad::var out = 1;

	out = tan(x);

	return out;
}

int main() {
	ad::tape tape;

	ad::var x = 5;
	tape.register_var(x);

	ad::var y = f(x);

	tape.clear_derivatives();
	y.seed(1);
	tape.compute_derivatives();

	std::cout << y.tape << "\n";
	std::cout << y << "\n";

	std::cout << x.grad() << "\n";
}