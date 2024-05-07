class Paper:
    def __init__(self, size="A4"):
        self.sizes = {
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

        self.size_name = size
        self.width, self.height = self.sizes[size]
        self.horizontal_ruled_lines = 34
        self.vertical_ruled_lines = 1
        self.offset_horizontal = 34
        self.offset_vertical = 33
        self.line_space = 8

    def get_size(self):
        return self.sizes.get(self.size_name)

    def set_size(self, size: tuple):
        self.width, self.height = size

    def set_paper(self, name="A4"):
        size = self.sizes.get(name)
        if size:
            self.width, self.height = size
        else:
            raise ValueError("Invalid paper size")

    def set_offsets(self, offset_horizontal: int, offset_vertical: int):
        if not isinstance(offset_horizontal, int) or not isinstance(offset_vertical, int):
            raise TypeError("Offsets must be integers")
        if offset_horizontal < 0 or offset_vertical < 0:
            raise ValueError("Offsets must be non-negative")

        self.offset_horizontal, self.offset_vertical = offset_horizontal, offset_vertical

    def set_ruled_line_parameters(self, horizontal_lines: int, vertical_lines: int, line_space: int):
        self.horizontal_ruled_lines, self.vertical_ruled_lines, self.line_space = horizontal_lines, vertical_lines, line_space