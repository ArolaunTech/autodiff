#include <iostream>

#include <ad/ad.h>

const int derindex = 4;

ad::Fwd<double, derindex> f(ad::Fwd<double, derindex> x) {
	return ad::tan(x);
}

int main() {
	ad::Fwd<double, derindex> x = 0;
	x.grads[1] = 1;

	ad::Fwd<double, derindex> y = f(x);

	for (int i = 0; i <= derindex; i++) {
		std::cout << y.derivative(i) << "\n";
	}
}