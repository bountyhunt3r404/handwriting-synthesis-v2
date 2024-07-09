import logging
import os

import numpy as np

from handwriting_synthesis import drawing
from handwriting_synthesis.config import prediction_path, checkpoint_path, style_path
from handwriting_synthesis.drawing.background import Background
from handwriting_synthesis.hand._draw import _draw
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
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename: str, lines: list[str], background: Background = None, biases: list[float] = None,
              styles=None, stroke_colors: list[str] = None, stroke_widths: list[float] = None,
              scale_factor: float = 1.0, alignment: str = "center") -> None:
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
        _draw(strokes, lines, filename, background, alignment, scale_factor,
              stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def strokes(self, lines: list[str], background: Background = None, biases: list[float] = None,
                styles=None, alignment: str = "center") -> np.array:

        # Validate alignment
        if alignment not in ['left', 'center', 'right']:
            raise ValueError("Invalid alignment value. Allowed values are 'left', 'center', 'right'.")

        # Validate and initialize background
        if background is None:
            background = Background()  # Default to A4 unruffled background

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

        return strokes

    def _sample(self, lines, biases=None, styles=None):
        """
        Prepares input data for the neural network and generates handwriting strokes.

        Parameters:
            lines (List[str]): List of strings representing each line of text.
            biases (List[float], optional): List of bias values for each line. Default is None.
            styles (List[str], optional): List of style names for each line. Default is None.

        Returns:
            List[np.array]: List of stroke sequences generated by the neural network.
        """
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
