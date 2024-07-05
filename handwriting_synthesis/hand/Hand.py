import os
from random import randint

import numpy as np
import svgwrite

import handwriting_synthesis.drawing.generator as generator
from handwriting_synthesis import drawing
from handwriting_synthesis.config import prediction_path, checkpoint_path, style_path
from handwriting_synthesis.drawing import offsets_to_coords, denoise, align
from handwriting_synthesis.drawing.background import Background
from handwriting_synthesis.rnn import RNN


class Hand(object):
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = RNN(
            log_dir='logs',
            checkpoint_dir=checkpoint_path,
            prediction_dir=prediction_path,
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            # logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename: str, lines: list[str], background: Background = None, biases=None, styles=None,
              stroke_colors=None, stroke_widths=None, scale_factor: float = 1.0, alignment: str = "center"):
        """
        Writes text to SVG file with customizable styles, biases, and alignment.

        Parameters:
            filename (str): The name of the file to write the text to.
            lines (List[str]): List of strings representing each line of text.
            background (Background, optional): A Background object specifying background size and type (ruled/unruled). Default is None (uses A4 unruled).
            biases (List[float], optional): List of bias values for each line. Default is None.
            styles (List[str], optional): List of style names for each line. Default is None.
            stroke_colors (List[str], optional): List of stroke colors for each line. Default is None.
            stroke_widths (List[float], optional): List of stroke widths for each line. Default is None.
            scale_factor (float, optional): Scaling factor. Default is 1.0.
            alignment (str, optional): Text alignment ('left', 'center', 'right'). Default is 'center'.

        Returns:
            None

        Raises:
            ValueError: If any of the lines exceed the maximum character length or contain invalid characters.
            ValueError: If an invalid alignment value is provided.
        """
        # Validate alignment
        if alignment not in ['left', 'center', 'right']:
            raise ValueError("Invalid alignment value. Allowed values are 'left', 'center', 'right'.")

        # Validate and initialize background
        if background is None:
            background = Background()  # Default to A4 unruled background

        # Validate lines
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > drawing.MAX_CHAR_LEN:
                raise ValueError(
                    (
                        f"Each line must be at most {drawing.MAX_CHAR_LEN} characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        # Generate strokes based on biases and styles
        strokes = self._sample(lines, biases=biases, styles=styles)

        # Draw strokes onto SVG file
        self._draw(strokes, lines, filename, background, alignment, scale_factor,
                   stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(f"{style_path}/style-{style}-strokes.npy")
                c_p = np.load(f"{style_path}/style-{style}-chars.npy").tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _draw(self, strokes, lines: list[str], filename, background: Background, alignment: str, scale_factor: float=1.0, stroke_colors=None,
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
