#include <iostream>

#include <ad/ad.h>

ad::var f(ad::var x) {
	return x + x;
}

int main() {
	ad::tape tape;

	ad::var x = tape.get_var();

	x = 5;

	std::cout << x.tape << "\n";
	std::cout << x.index << "\n";

	ad::var y = f(x);

	std::cout << y.tape << "\n";
	std::cout << y.index << "\n";

	std::cout << "\n";

	for (std::size_t i = 0; i < tape.size(); i++) {
		std::cout << tape[i].value << "\n";
		std::cout << tape[i].grad << "\n";
		std::cout << "{\n";
		for (std::size_t j = 0; j < tape[i].refs.size(); j++) {
			std::cout << "\t" << tape[i].refs[j] << ",\n";
		}
		std::cout << "}\n";
		std::cout << tape[i].operation << "\n\n";
	}
}