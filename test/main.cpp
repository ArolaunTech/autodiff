#include <iostream>

#include <ad/ad.h>

const int derindex = 25;

ad::Fwd<double, derindex> f(ad::Fwd<double, derindex> x) {
	return ad::erf(x + 1.0);
}

int main() {
	ad::tape tape;

	ad::var x = tape.get_var();

	x = 5;

	std::cout << x.tape << "\n";
	std::cout << x.index << "\n";
	std::cout << x.value() << "\n";
	std::cout << x.grad() << "\n"; 
}