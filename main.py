import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle
from inspect import isfunction

from typing import Union, List, Tuple


def fun(x):
    return np.exp(-2 * x) + x ** 2 - 1


def dfun(x):
    return -2 * np.exp(-2 * x) + 2 * x


def ddfun(x):
    return 4 * np.exp(-2 * x) + 2


def bisection(a: Union[int, float], b: Union[int, float], f: typing.Callable[[float], float], epsilon: float,
              iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and isfunction(f) and isinstance(epsilon,
                                                                                                         float) and isinstance(
            iteration, int)):
        return None
    if f(a) * f(b) > 0:
        return None

    iterations = 0
    while iterations < iteration:
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < epsilon or (b - a) / 2 < epsilon:
            return c, iterations

        if np.sign(f(a)) == np.sign(fc):
            a = c
        else:
            b = c

        iterations += 1

    return c, iterations


def difference_quotient(f: typing.Callable[[float], float], x: Union[int, float], h: Union[int, float]):
    '''Funkcja obliczająca iloaz różnicowy zadanej funkcji
    Parametry:

    f - funkcja dla której jest poszukiwane rozwiązanie
    x - argument funkcji la której jest
    h - krok różnicy wykorzystywanej do wyliczenia ilorazu różnicowego

    return:
    diff - wartość ilorazu różnicowego

    '''
    if not (isfunction(f) and isinstance(x, (int, float)) and isinstance(h, (int, float)) and h != 0):
        return None
    return (f(x + h) - f(x)) / h


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float],
           ddf: typing.Callable[[float], float], a: Union[int, float], b: Union[int, float], epsilon: float,
           iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry:
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''

    if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) and isinstance(
            iteration, int) and callable(f) and callable(df) and callable(ddf)):
        return None
    if f(a) * f(b) > 0 or ddf(a) * ddf(b) < 0 or df(a) * df(b) < 0:
        return None

    counter = 0
    x0 = b
    while abs(f(x0)) > epsilon and counter < iteration:
        counter += 1
        x0 = x0 - (f(x0) / df(x0))
    return x0, counter



