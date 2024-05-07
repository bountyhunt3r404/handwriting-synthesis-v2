import numpy as np
import svgwrite

from handwriting_synthesis import drawing
from handwriting_synthesis.drawing.paper import Paper
from random import randint


def draw(strokes, lines, filename, paper: Paper,
         scale_factor=1.0,
         stroke_colors=None,
         stroke_widths=None
         ):
    """
    Draw strokes and lines onto an SVG file with a ruled background.

    Parameters:
        strokes (list): A list of stroke data.
            Each element in the list represents the coordinates of a stroke.
        lines (list): A list of lines.
            Each line represents a separate element to be drawn.
        filename (str): The filename of the SVG file to be saved.
            The SVG file will contain the rendered strokes and lines.
        paper (Paper): An instance of the Paper class representing the drawing paper.
        scale_factor (float, optional): The scaling factor for the strokes. Defaults to 1.0.
            This parameter scales the size of the strokes. Values greater than 1.0 increase the size,
            while values less than 1.0 decrease the size.
        stroke_colors (list, optional): List of stroke colors for each line. Defaults to None.
            If provided, each line will be drawn with the corresponding color in the list.
            If not provided, all lines will be drawn in black.
        stroke_widths (list, optional): List of stroke widths for each line. Defaults to None.
            If provided, each line will be drawn with the corresponding width in the list.
            If not provided, all lines will be drawn with a width of 2.

    Returns:
        None

    Notes:
        This function generates an SVG file with the specified filename, containing the rendered strokes and lines.
        The strokes and lines are drawn relative to the provided paper size, offsets, and scaling factor.
        If stroke_colors and stroke_widths are not provided, default values are used.

    Recommendations:
        1. Ensure the strokes and lines data are correctly formatted before passing them to this function.
        2. Use meaningful filenames for the SVG output to easily identify the content.
        3. Experiment with different values for offset_horizontal, offset_vertical, and scale_factor
           to achieve the desired positioning and scaling of the strokes.
        4. If stroke_colors and stroke_widths are provided, ensure they have the same length as the lines list.
        5. Consider providing stroke_colors and stroke_widths to customize the appearance of the rendered lines.
    """

    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    line_height = drawing.generator.mm_to_px(paper.line_space)
    view_width, view_height = paper.get_size()  # Extracting width and height from the paper size tuple

    # Create a new SVG drawing
    dwg = drawing.generator.create_svg_with_ruled_lines(filename,
                                                        size=paper.get_size(),
                                                        offset_horizontal=paper.offset_horizontal,
                                                        offset_vertical=paper.offset_vertical,
                                                        line_gap=paper.line_space)

    # Initial position for drawing strokes
    initial_coord = np.array([0, -line_height / 4])

    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

        if not line:
            initial_coord[1] -= line_height
            continue

        # Scaling the stroke coordinates
        offsets[:, :2] *= scale_factor
        strokes = drawing.offsets_to_coords(offsets)
        strokes = drawing.denoise(strokes)
        strokes[:, :2] = drawing.align(strokes[:, :2])

        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord
        # strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2
        strokes[:, 0] += drawing.generator.mm_to_px(paper.offset_horizontal) - drawing.generator.mm_to_px(randint(0, 6))
        strokes[:, 1] += drawing.generator.mm_to_px(paper.offset_vertical)

        prev_eos = 1.0
        p = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos

        # Create a path element for the strokes
        path = svgwrite.path.Path(p)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)

        initial_coord[1] -= line_height

    # Save the SVG drawing to a file
    dwg.save()
