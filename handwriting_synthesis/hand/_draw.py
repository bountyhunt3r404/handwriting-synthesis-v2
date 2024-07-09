from random import randint

import numpy as np
import svgwrite

import handwriting_synthesis.drawing.generator as generator
from handwriting_synthesis.drawing import Background
from handwriting_synthesis.drawing import offsets_to_coords, denoise, align


def _draw(strokes,
          lines: list[str],
          filename, background: Background,
          alignment: str,
          scale_factor: float = 1.0,
          stroke_colors=None,
          stroke_widths=None) -> None:
    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    # Convert line height from millimeters to pixels
    line_height = generator.mm_to_px(background.line_space)

    # Convert view size from millimeters to pixels
    view_width_px = generator.mm_to_px(background.get_size()[0])
    view_height_px = generator.mm_to_px(background.get_size()[1])

    # Create a new SVG drawing with background
    dwg = generator.create_svg_with_lines(filename, background, save_file=False)

    # Calculate initial position based on alignment
    # if alignment == 'left':
    #     initial_x = 0
    # elif alignment == 'center':
    #     initial_x = (view_width_px - background.width) / 2 + generator.mm_to_px(background.offset_horizontal)
    # elif alignment == 'right':
    #     initial_x = view_width_px - background.width + generator.mm_to_px(background.offset_horizontal)
    # else:
    #     raise ValueError(f"Invalid alignment value: {alignment}. Must be 'left', 'center', or 'right'.")

    initial_x = 0
    initial_y = - line_height / 4
    padding = generator.mm_to_px(5)

    # Iterate through strokes and lines
    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
        if not line:
            initial_y -= line_height
            continue

        # Scaling the stroke coordinates
        offsets[:, :2] *= scale_factor
        strokes = offsets_to_coords(offsets)
        strokes = denoise(strokes)
        strokes[:, :2] = align(strokes[:, :2])

        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min(axis=0) + np.array([initial_x, initial_y])

        if alignment == 'left':
            strokes[:, 0] += (generator.mm_to_px(background.offset_horizontal) +
                              generator.mm_to_px(randint(0, 6)) +
                              padding)
        elif alignment == 'right':
            # Adjust the x-coordinate to align to the right
            strokes[:, 0] += view_width_px - strokes[:, 0].max() - padding
        else:  # Center alignment handled in previous block
            strokes[:, 0] += (view_width_px - strokes[:, 0].max()) / 2

        strokes[:, 1] += generator.mm_to_px(background.offset_vertical)

        # Convert strokes into SVG path string
        prev_eos = 1.0
        path_str = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            path_str += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos

        # Create SVG path element for the strokes
        path = svgwrite.path.Path(path_str)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)

        initial_y -= line_height

    # Save the SVG drawing to a file
    dwg.save()