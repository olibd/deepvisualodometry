import datetime
import functools
import sys
import time

from comet_ml import Experiment, APIExperiment


class CometLogger:
    __experiment: Experiment = None
    __APIExperiment: APIExperiment = None
    gpu_id: int = None
    print_to_comet_only: bool = False

    def __init__(self, experiment: Experiment, gpu_id=None, print_to_comet_only=False):
        if CometLogger.__experiment is not None:
            raise Exception("Cannot re-instantiate since this class is a singleton.")
        else:

            CometLogger.__experiment = experiment
            CometLogger.__APIExperiment = APIExperiment(previous_experiment=experiment.get_key())
            CometLogger.gpu_id = gpu_id
            CometLogger.print_to_comet_only = print_to_comet_only

    @staticmethod
    def get_experiment() -> Experiment:
        if CometLogger.__experiment is None:
            raise Exception("You need to init this singleton with an experiment first.")
        return CometLogger.__experiment

    @staticmethod
    def print(text_output: str):
        """
        Will print the given text to the console and Comet will log it (unless console logging is deactivated in
        the experiment). If CometLogger.__print_to_experiment is set to true, the text will only be printed to
        the Comet experiments and not to the console. This is useful
        when running multiple experiments on separate processes.
        @param text_output:
        """
        if CometLogger.gpu_id is not None:
            text_output = "GPU {}: ".format(CometLogger.gpu_id) + text_output

        if CometLogger.print_to_comet_only:
            text_output = text_output + "\n"
            CometLogger.__APIExperiment.log_output(text_output, timestamp=CometLogger.__TimestampMillisec64())
        else:
            print(text_output)

    @staticmethod
    def fatalprint(text_output: str):
        """
        Prints and log to Comet.ML than kill the process
        @param text_output:
        """
        CometLogger.print(text_output)
        raise Exception(text_output)

    @staticmethod
    def log_output_as_metric(log_key: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)
                CometLogger.get_experiment().log_metric(log_key, ret)
                return ret
            return wrapper
        return decorator

    @staticmethod
    def log_input_as_metric(log_key: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                CometLogger.get_experiment().log_metric(log_key, [args, kwargs])
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def __TimestampMillisec64():
        return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        CometLogger.__experiment = None
        CometLogger.__APIExperiment = None
        CometLogger.gpu_id = None
        CometLogger.__print_to_experiment = False
