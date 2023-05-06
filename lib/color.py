from typing import Annotated

def rgba_to_hex(rgba: Annotated[tuple[float], 4]) -> str:
    r, g, b, a = rgba
    r_hex = hex(int(r * 255))[2:].zfill(2)
    g_hex = hex(int(g * 255))[2:].zfill(2)
    b_hex = hex(int(b * 255))[2:].zfill(2)
    a_hex = hex(int(a * 255))[2:].zfill(2)
    return f'#{r_hex}{g_hex}{b_hex}{a_hex}'
