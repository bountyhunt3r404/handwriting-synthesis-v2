import logging
import os

import numpy as np

from handwriting_synthesis import drawing
from handwriting_synthesis.config import prediction_path, checkpoint_path, style_path
from handwriting_synthesis.drawing.paper import Paper
from handwriting_synthesis.hand._draw_enhanced import draw
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

    def write(self, filename: str, paper: Paper, words: list[str], biases=None, styles=None,
              stroke_colors=None, stroke_widths=None, scale_factor=1.0):
        """
        Writes text to SVG file with customizable styles and biases.

        Parameters:
            filename (str): The name of the file to write the text to.
            paper (Paper): A Paper object specifying paper size and offsets.
            words (List[str]): List of strings representing each word.
            biases (List[float], optional): List of bias values for each word. Default is None.
            styles (List[str], optional): List of style names for each word. Default is None.
            stroke_colors (List[str], optional): List of stroke colors for each word. Default is None.
            stroke_widths (List[float], optional): List of stroke widths for each word. Default is None.
            scale_factor (float, optional): Scaling factor. Default is 1.0.

        Returns:
            None

        Raises:
            ValueError: If any of the words exceed the maximum character length or contain invalid characters.
        """
        valid_char_set = set(drawing.alphabet)
        for word_num, word in enumerate(words):
            if len(word) > drawing.MAX_CHAR_LEN:
                raise ValueError(
                    (
                        f"Each word must be at most {drawing.MAX_CHAR_LEN} characters. "
                        "Word {} contains {}"
                    ).format(word_num, len(word))
                )

            for char in word:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in word {}. "
                            "Valid character set is {}"
                        ).format(char, word_num, valid_char_set)
                    )

        strokes = self._sample(words, biases=biases, styles=styles)
        draw(strokes, words, filename, paper, scale_factor,
             stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, words, biases=None, styles=None):
        num_samples = len(words) # no. of words
        max_tsteps = 40 * max([len(i) for i in words])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3]) # a 1200 x 3 grid for each word
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120]) # a 120 length array for each word
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(words, styles)):
                # cs = current word
                # style = current style
                """
                This is how "priming" works. The model is "shown" the text "The quick brown fox" and at the same time shown the actual strokes (x_p) for it. 
                Then, when it gets to the "Hello World" part of the string (for which it has no stroke data), it tries to continue drawing in the same style it just learned.
                """
                x_p = np.load(f"{style_path}/style-{style}-strokes.npy") # It loads the priming stroke data (the .npy file containing the [x, y, end_flag] data) for that style. This is the visual part of the style.
                c_p = np.load(f"{style_path}/style-{style}-chars.npy").tostring().decode('utf-8') # It loads the text that corresponds to those strokes (e.g., "The quick brown fox"). This is the text part of the style.

                c_p = str(c_p) + " " + cs # It creates a new, combined string. For example: "The quick brown fox Hello World".
                c_p = drawing.encode_ascii(c_p) # It then encodes this entire combined string (c_p) and puts it into chars[i].
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p # It puts the loaded style strokes (x_p) into x_prime[i].
                x_prime_len[i] = len(x_p) # It records the actual lengths of c_p and x_p into chars_len[i] and x_prime_len[i].
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(words[i])
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
        """
        x_prime_len is the Timer: The x_prime_len variable acts as the timer for this switch.
        As soon as step t = x_prime_len is reached, the model's switch flips.
        It stops consuming strokes from x_prime.
        It starts generating its own strokes.
        The attention mechanism, which had "finished" looking at the "sample text," now moves on to the 
        "hello world" part of the chars array to guide this new generation.
        """

        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        """
        This is a list comprehension that does the following for each sample (i.e., for each line's generated output):
        np.all(sample == 0.0, axis=1): Finds all rows that are [0.0, 0.0, 0.0]. This returns a boolean array like [False, False, ..., True, True].
        ~: This is a "NOT" operator. It inverts the boolean array to [True, True, ..., False, False].
        sample[...]: This is "fancy indexing". It selects only the rows that are True.
        In short, it trims all the trailing zero-padding from the end of the generated data.
        The function then returns samples, which is now a clean list, where each element is a NumPy array of shape (actual_steps_taken, 3) representing the strokes for one line.
        """
        return samples
