num_char_names = [
    'num_chars_0K_12K',
    'num_chars_12K_48K',
    'num_chars_48K_192K',
    'num_chars_192K_768K',
    'num_chars_768K_1536K',
    'num_chars_1536K_3072K',
]

num_char_ranges = [
    (0, 12000, 'num_chars_0K_12K'),
    (12000, 48000, 'num_chars_12K_48K'),
    (48000, 192000, 'num_chars_48K_192K'),
    (192000, 768000, 'num_chars_192K_768K'),
    (768000, 1536000, 'num_chars_768K_1536K'),
    (1536000, 3072000, 'num_chars_1536K_3072K'),
]

assert all(r[-1] in num_char_names for r in num_char_ranges), f'num_char names and ranges are different'
# TODO: add more consistency checks

def get_num_char_name(num_chars: int) -> str | None:
    for lower, upper, name in num_char_ranges:
        if lower <= num_chars < upper:
            return name
    return None
