#include <iostream>

#include <ad/ad.h>

ad::Var<ad::var> f(ad::Var<ad::var> x) {
	return x / 2.0 + 5.0 * exp(x) / sin(x);
}

int main() {
	double xinin = 5;

	ad::tape tape2;

	ad::var xin = tape2.get_var(xinin);

	ad::Tape<ad::var> tape;

	ad::Var<ad::var> x = tape.get_var(xin);

	ad::Var<ad::var> y = f(x);

	tape.clear_derivatives();
	y.seed(1);
	tape.compute_derivatives();

	ad::var yder = x.grad();

	tape2.clear_derivatives();
	yder.seed(1);
	tape2.compute_derivatives();

	double ddy = xin.grad();

	std::cout << y.value().value() << " " << yder.value() << " " << ddy << "\n";
}