#ifndef FACTORIAL_H
#define FACTORIAL_H

unsigned int factorial(unsigned int N) {
	if (N == 0) return 1;

	return N * factorial(N - 1);
}

#endif