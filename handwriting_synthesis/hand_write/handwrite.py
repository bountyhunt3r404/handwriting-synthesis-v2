import os
import pathlib
import textwrap

from handwriting_synthesis.drawing import Background
from handwriting_synthesis.hand import Hand

# Initialize Hand object statically
hand = Hand()


def handwrite(filename: pathlib.Path, text: any, style: int = None, ruled: bool = False, color: str = 'black',
              align: str = 'center',
              width: int = 70, scale: float = 1.0, legibility: float = 1.0, stroke_widths: float = 1.0) -> None:
    """
    Generate handwriting in SVG format based on input parameters.

    Parameters:
        text (str): The text to be written in handwriting.
        filename (str): The name of the SVG file to generate.
        style (str, optional): The style of handwriting (e.g., 'italic'). Default is None.
        ruled (bool, optional): Whether to use ruled background. Default is False.
        color (str, optional): The stroke color. Default is 'black'.
        align (str, optional): Text alignment ('left', 'center', 'right'). Default is 'center'.
        width (int, optional): Maximum width of each line. Default is 70.
        scale (float, optional): Scaling factor for the handwriting. Default is 1.0.
        legibility (float or List[float], optional): legibility for each line. Default is 1.0.
        stroke_widths (float or List[float], optional): Stroke widths for each line. Default is 1.0.

    Raises:
        ValueError: If an invalid alignment value is provided.
        FileExistsError: If the specified filename already exists.
    """
    # Validate alignment
    if align not in ['left', 'center', 'right']:
        raise ValueError("Invalid alignment value. Allowed values are 'left', 'center', 'right'.")

    # Check if filename already exists
    # if os.path.exists(filename):
    #     raise FileExistsError(f"File '{filename}' already exists. Please choose a different filename.")

    # Wrap text into lines
    lines = textwrap.wrap(text=text, width=width)

    # Prepare legibility, stroke widths, and styles
    if isinstance(legibility, float):
        legibility = [legibility] * len(lines)
    if isinstance(stroke_widths, float):
        stroke_widths = [stroke_widths] * len(lines)
    styles = [style] * len(lines) if style else None

    # Generate SVG using pre-initialized Hand object
    hand.write(
        filename=filename,
        lines=lines,
        background=Background(ruled=ruled),
        biases=legibility,
        styles=styles,
        stroke_colors=[color] * len(lines),
        stroke_widths=stroke_widths,
        scale_factor=scale,
        alignment=align
    )
