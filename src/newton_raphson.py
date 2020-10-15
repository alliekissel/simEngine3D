#!/usr/bin/env python3

def newton_raphson(f, df, x0, eps, max_iter):
    # Return the approximate solution to f(x)=0 via Newton-Raphson Method

    # Initialize and begin Newton-Raphson method
    xk = x0
    for i in range(0, max_iter):
        if abs(f(xk)) < eps:
            return xk
        if df(xk) == 0:
            print('Derivative is zero.')
            return None

        h = f(xk)/df(xk)
        xkk = xk - h

        # Update xn for next iteration
        xk = xkk

    print('Exceeded maximum iterations.')
    return None
