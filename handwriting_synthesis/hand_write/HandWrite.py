from handwriting_synthesis.hand import Hand
from handwriting_synthesis.drawing import Background
import textwrap


class HandWrite(Hand, Background):
    def __init__(self,
                 text: str,
                 **kwargs):
        super().__init__()
        pass