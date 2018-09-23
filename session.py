"""
Integrate the Losswise service into the current project using a context manager.
"""

# EXT
import losswise


class LWSession:
    """
    Context manager that creates a Losswise session that lets you log model parameters and performance.
    If no API key is given, just do nothing (including not throwing any errors). Statements referring to any
    Losswise functionalities can however still be left inside the code.
    """
    def __init__(self, api_key, tag, model_params, **session_params):
        # Set key
        if api_key is not None:
            self.api_key_given = True
            losswise.set_api_key(api_key)

            # Create Session object
            self.session = losswise.Session(tag=tag, params=model_params, **session_params)
        else:
            self.api_key_given = False

        self.graphs = {}

    def create_graph(self, *args, **kwargs):
        """
        Create a graph in which data will be plotted.
        """
        if self.api_key_given:
            self.graphs[args[0]] = self.session.graph(*args, **kwargs)

    def __enter__(self):
        # Little hacky: Return the class itself when with block is entered. This way log() can be used and
        # a function call can easily be ignored when no API is given without writing special subclasses inhering from
        # losswise.session and losswise.graph
        return self

    def __exit__(self, type, value, traceback):
        if self.api_key_given:
            self.session.done()

    def plot(self, *args, title=None):
        if self.api_key_given:
            assert title is not None, "Title of graph needs to be given."
            self.graphs[title].append(*args)
