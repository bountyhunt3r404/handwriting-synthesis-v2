class Background:
    STANDARD_SIZES = {
        "A0": (841, 1189),
        "A1": (594, 841),
        "A2": (420, 594),
        "A3": (297, 420),
        "A4": (210, 297),
        "A5": (148, 210),
        "A6": (105, 148),
        "Letter": (216, 279),
        "Legal": (216, 356),
        "Tabloid": (279, 432)
    }

    def __init__(self, size="A4", ruled=False):
        if isinstance(size, str):
            self.size = self.STANDARD_SIZES[size]
        elif isinstance(size, tuple) and len(size) == 2:
            self.size = (size[0], size[1])
        else:
            raise ValueError("Invalid size format. Use a string from STANDARD_SIZES or a tuple (width, height).")

        self.horizontal_ruled_lines = 32 if ruled else 0
        self.vertical_ruled_lines = 1 if ruled else 0
        self.offset_horizontal = 34
        self.offset_vertical = 33
        self.line_space = 8

    def set_size(self, size_specification) -> None:
        if isinstance(size_specification, str):
            if size_specification not in self.STANDARD_SIZES:
                raise ValueError("Invalid paper size")
            self.size = self.STANDARD_SIZES[size_specification]
        elif isinstance(size_specification, tuple) and len(size_specification) == 2:
            width, height = size_specification
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive integers")
            self.size = (width, height)
        else:
            raise ValueError("Invalid size format. Use a string from STANDARD_SIZES or a tuple (width, height).")

    def get_size(self) -> tuple[int, int]:
        return self.size

    def set_offsets(self, offset_horizontal, offset_vertical) -> None:
        if not isinstance(offset_horizontal, int) or not isinstance(offset_vertical, int):
            raise TypeError("Offsets must be integers")
        if offset_horizontal < 0 or offset_vertical < 0:
            raise ValueError("Offsets must be non-negative")
        self.offset_horizontal = offset_horizontal
        self.offset_vertical = offset_vertical

    def set_ruled_line_parameters(self, horizontal_lines, vertical_lines, line_space) -> None:
        if not isinstance(horizontal_lines, int) or not isinstance(vertical_lines, int) or not isinstance(line_space, int):
            raise TypeError("Line parameters must be integers")
        if horizontal_lines < 0 or vertical_lines < 0 or line_space < 0:
            raise ValueError("Line parameters must be non-negative")
        self.horizontal_ruled_lines = horizontal_lines
        self.vertical_ruled_lines = vertical_lines
        self.line_space = line_space
