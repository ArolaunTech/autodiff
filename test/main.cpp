#include <iostream>

#include <ad/ad.h>

const int derindex = 4;

ad::Fwd<double, derindex> f(ad::Fwd<double, derindex> x) {
	ad::Fwd<double, derindex> y = x, z = 2;
	y /= z;

	return ad::exp2(ad::sqrt(y));
}

int main() {
	ad::Fwd<double, derindex> x = 0.5;
	x.grads[1] = 1;

	ad::Fwd<double, derindex> y = f(x);

	for (int i = 0; i <= derindex; i++) {
		std::cout << y.derivative(i) << "\n";
	}
}