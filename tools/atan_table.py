"""The constants of include/numerical/elementary.hpp's atan, from exact decimal arithmetic, printed as the header holds
them: π/2 as the double nearest it and the double nearest what that one leaves out, and atan(k/64) for k = 0…64,
each the double nearest it and the double nearest what that one leaves out."""
from decimal import Decimal, getcontext

getcontext().prec = 80


def arctangent(x):
    """atan(`x`) for 0 ≤ x ≤ 1: three halvings, atan x = 2 atan(x / (1 + sqrt(1 + x²))), then its Taylor series."""
    for _ in range(3):
        x = x / (1 + (1 + x * x).sqrt())
    total, power, n, square = Decimal(0), x, 1, x * x
    while True:
        term = power / n
        if abs(term) < Decimal(10) ** -75:
            break
        total += term if n % 4 == 1 else -term
        power *= square
        n += 2
    return 8 * total


def pair(value):
    """The double nearest `value` and the double nearest what that one leaves out."""
    head = float(value)
    return head, float(value - Decimal(head))


def main():
    half_pi = 2 * arctangent(Decimal(1))
    head, tail = pair(half_pi)
    print("atan_half_pi_high", head.hex())
    print("atan_half_pi_low ", tail.hex())
    for k in range(65):
        head, tail = pair(arctangent(Decimal(k) / 64))
        print(f"{head.hex()}, {tail.hex()},")


if __name__ == "__main__":
    main()
