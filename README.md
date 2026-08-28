# [insert name here]

I don't have a good name for this yet. 

An autodifferentiation library in C++. This is designed to be super simple to include in your project and also be fairly low-level. That does lead to some monstrosities like this:

```cpp
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
```

(getting the second derivative of a function via reverse autodifferentiation) but it also lets you handle computing things like Hessians yourself.

 - **Requires C++20**

## Installation

You can install this like a normal header-only library (no instructions provided) or you can just include the `ad` folder in your project and include the `ad/ad.h` file when you want to do autodifferentiation.