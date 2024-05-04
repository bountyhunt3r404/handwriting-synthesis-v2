import svgwrite


def mm_to_px(length, dpi=96, unit='mm'):
    """
    Convert millimeters or other units to pixels.

    Parameters:
        length (float): Length to be converted to pixels.
        dpi (int, optional): Dots per inch, default is 96 DPI (standard for most screens).
        unit (str, optional): Unit of the length. Possible values are 'mm', 'cm', 'in'. Default is 'mm'.

    Returns:
        int: Length in pixels.
    """
    if unit == 'mm':
        return round(length * dpi / 25.4)
    elif unit == 'cm':
        return round(length * dpi / 2.54)
    elif unit == 'in':
        return round(length * dpi)
    else:
        raise ValueError("Invalid unit. Supported units are 'mm', 'cm', and 'in'.")


def create_svg_with_ruled_lines(filename, size=(210, 297), offset_horizontal=0, offset_vertical=0, line_gap=8,
                                line_color="black", line_width=0.5, unit='mm'):
    """
    Create an SVG file with ruled lines.

    Parameters:
        filename (str): The filename of the SVG file to be saved.
        size (tuple, optional): Paper size in millimeters. Default is A4 size (210mm x 297mm).
        offset_horizontal (float, optional): Horizontal offset for the ruled lines. Default is 0.
        offset_vertical (float, optional): Vertical offset for the ruled lines. Default is 0.
        line_gap (float, optional): Gap between each ruled line. Default is 8.
        line_color (str, optional): Color of the ruled lines. Default is "black".
        line_width (float, optional): Width of the ruled lines. Default is 0.5.
        unit (str, optional): Unit of the dimensions and offsets. Possible values are 'mm', 'cm', and 'in'. Default is 'mm'.

    Returns:
        dwg (svgwrite.Drawing): SVG drawing object.
    """
    # Unpack the size tuple
    width, height = size

    # Create a new SVG drawing with default units set to millimeters
    dwg = svgwrite.Drawing(filename=filename, size=(f"{width}{unit}", f"{height}{unit}"), profile='tiny')

    # Set the viewbox of the SVG drawing
    dwg.viewbox(width=mm_to_px(width, unit=unit), height=mm_to_px(height, unit=unit))

    # Define starting position
    x_start = mm_to_px(offset_horizontal, unit=unit)
    y_start = mm_to_px(offset_vertical, unit=unit)

    # Draw ruled lines horizontally
    while y_start <= mm_to_px(height, unit=unit) - mm_to_px(15,
                                                            unit=unit):  # Adjusted to leave space for the border line
        horizontal_line = dwg.line((f"{mm_to_px(0, unit=unit)}", f"{y_start}"),
                                   (f"{mm_to_px(width, unit=unit)}", f"{y_start}"))
        horizontal_line.stroke(width=mm_to_px(line_width, unit=unit), color=line_color)
        dwg.add(horizontal_line)

        y_start += mm_to_px(line_gap, unit=unit)

    # Draw border vertical Line
    vertical_line = dwg.line((f"{x_start}", f"{mm_to_px(0, unit=unit)}"),
                             (f"{x_start}", f"{mm_to_px(height, unit=unit)}"))
    vertical_line.stroke(width=mm_to_px(line_width, unit=unit), color=line_color)
    dwg.add(vertical_line)

    return dwg
