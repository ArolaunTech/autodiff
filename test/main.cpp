#include <iostream>

#include <ad/ad.h>

ad::Fwd<double, 4> f(ad::Fwd<double, 4> x) {
	return ad::exp(2 * x);
}

int main() {
	ad::Fwd<double, 4> x = 0.5;
	x.grads[1] = 1;

	ad::Fwd<double, 4> y = f(x);

	std::cout << y.derivative(0) << "\n";
	std::cout << y.derivative(1) << "\n";
	std::cout << y.derivative(2) << "\n";
	std::cout << y.derivative(3) << "\n";
	std::cout << y.derivative(4) << "\n";
}