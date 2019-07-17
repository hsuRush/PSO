# -*- coding: UTF-8 -*-

def rosenbrock(x,y):
    return (1-x)**2 + 100* ((y-x**2))**2

def rotated_rosenbrock(x, y):
    return (1-y)**2 + 100* ((x-y**2))**2

def rastrigin(x,y):
    return ( x**2 - 10 * np.cos(2*np.pi*x) + 10 ) + (y**2 - 10 * np.cos(2 * np.pi * y) + 10 )

def griewank(x, y):
    return 1 + (x**2 + y**2) / 4000 - cos(x) * cos((1/2) * y * 2**.5)

if __name__ == "__main__":
    print(rosenbrock(1,1))
