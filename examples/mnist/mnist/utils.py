
def metadata_label(digit_int) -> int:
    return digit_int

def metadata_even_odd(digit_int) -> str:
    if digit_int % 2 == 0:
        return "even"
    else:
        return "odd"

def metadata_circle(digit_int) -> str:
    if digit_int in [0, 6, 8, 9]:
        return 'yes'
    else:
        return 'no'
