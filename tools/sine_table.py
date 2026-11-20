"""The constants of include/numerical/elementary.hpp's sin and cos of a double, from exact decimal arithmetic, printed as
the header holds them: 32/π to the nearest double, π/32 in three parts (the double nearest it, the double nearest what
that one leaves out, then the double nearest what both leave out), and for j = 0…63 sin(jπ/32) and cos(jπ/32), each
the double nearest it and the double nearest what that one leaves out."""
from decimal import Decimal, getcontext

getcontext().prec = 120


def arctangent(x):
    """atan(`x`) for 0 ≤ x ≤ 1: three halvings, atan x = 2 atan(x / (1 + sqrt(1 + x²))), then its Taylor series."""
    for _ in range(3):
        x = x / (1 + (1 + x * x).sqrt())
    total, power, n, square = Decimal(0), x, 1, x * x
    while True:
        term = power / n
        if abs(term) < Decimal(10) ** -115:
            break
        total += term if n % 4 == 1 else -term
        power *= square
        n += 2
    return 8 * total


PI = 4 * arctangent(Decimal(1))


def sine(x):
    """sin(`x`) for |x| ≤ 2π by its Taylor series."""
    total, term, n = Decimal(0), x, 1
    while abs(term) > Decimal(10) ** -115:
        total += term
        term = -term * x * x / ((n + 1) * (n + 2))
        n += 2
    return total


def pair(value):
    """The double nearest `value` and the double nearest what that one leaves out, a series' last digits (below
    10⁻¹⁰⁰, where the value is a double exactly: 0 and 1) dropped."""
    value = Decimal(0) if abs(value) < Decimal(10) ** -100 else value
    head = float(value)
    rest = value - Decimal(head)
    return head, float(Decimal(0) if abs(rest) < Decimal(10) ** -100 else rest)


def main():
    step = PI / 32
    first = float(step)
    second = float(step - Decimal(first))
    third = float(step - Decimal(first) - Decimal(second))
    print("sine_inverse_step", float(32 / PI).hex())
    print("sine_step_parts", first.hex(), second.hex(), third.hex())
    for j in range(64):
        s_head, s_tail = pair(sine(j * step))
        c_head, c_tail = pair(sine(j * step + PI / 2))
        print(f"{s_head.hex()}, {s_tail.hex()}, {c_head.hex()}, {c_tail.hex()},")


if __name__ == "__main__":
    main()
