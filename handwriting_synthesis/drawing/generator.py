import svgwrite

from .background import Background


def mm_to_px(length, dpi=96, unit='mm'):
    """
    Convert millimeters or other units to pixels.

    Parameters:
        length (float): Length to be converted to pixels.
        dpi (int, optional): Dots per inch, default is 96 DPI (standard for most screens).
        unit (str, optional): Unit of the length. Possible values are 'mm', 'cm', and 'in'. Default is 'mm'.

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


def create_blank_svg(filename, background: Background, save_file=False):
    """
    Create a blank SVG file based on the Background instance.

    Parameters:
        filename (str): The filename of the SVG file to be saved (only used if save_file=True).
        background (Background): An instance of the Background class defining paper size.
        save_file (bool, optional): Whether to save the SVG file. Default is False (return the SVG object).

    Returns:
        svgwrite.Drawing or None: If save_file=False, returns the SVG drawing object. Otherwise, saves the SVG file and returns None.
    """
    # Unpack size from Background instance
    width, height = background.get_size()

    # Create a new SVG drawing with default units set to millimeters
    dwg = svgwrite.Drawing(filename=filename, size=(f"{width}mm", f"{height}mm"), profile='tiny')

    # Adding a white background rectangle
    dwg.add(dwg.rect(insert=(0, 0), size=(f"{width}mm", f"{height}mm"), fill='white'))

    # Save SVG file if specified
    if save_file:
        dwg.save()
        return None
    else:
        return dwg


def create_svg_with_lines(filename, background: Background, save_file=False):
    """
    Create an SVG file with ruled lines based on the Background instance.

    Parameters:
        filename (str): The filename of the SVG file to be saved (only used if save_file=True).
        background (Background): An instance of the Background class defining paper size and ruled line parameters.
        save_file (bool, optional): Whether to save the SVG file. Default is False (return the SVG object).

    Returns:
        svgwrite.Drawing or None: If save_file=False, returns the SVG drawing object. Otherwise, saves the SVG file and returns None.
    """
    # Unpack size from Background instance
    width, height = background.get_size()

    # Create a new SVG drawing with default units set to millimeters
    dwg = svgwrite.Drawing(filename=filename, size=(f"{width}mm", f"{height}mm"), profile='tiny')

    # Set the viewbox of the SVG drawing (in pixels)
    dwg.viewbox(0, 0, mm_to_px(width), mm_to_px(height))

    # Adding a white background rectangle
    dwg.add(dwg.rect(insert=(0, 0), size=(f"{width}mm", f"{height}mm"), fill='white'))

    # Calculate starting position in pixels based on offsets
    x_start = mm_to_px(background.offset_horizontal)
    y_start = mm_to_px(background.offset_vertical)

    # Draw horizontal ruled lines
    for i in range(background.horizontal_ruled_lines):
        y = y_start + i * mm_to_px(background.line_space)
        horizontal_line = dwg.line((0, y), (mm_to_px(width), y))
        horizontal_line.stroke(width=mm_to_px(0.5), color='black')
        dwg.add(horizontal_line)

    # Draw vertical borderline
    if background.vertical_ruled_lines != 0:
        vertical_line = dwg.line((x_start, 0), (x_start, mm_to_px(height)))
        vertical_line.stroke(width=mm_to_px(0.5), color='black')
        dwg.add(vertical_line)

    # Save SVG file if specified
    if save_file:
        dwg.save()
        return None
    else:
        return dwg
