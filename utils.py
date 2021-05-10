# coding: utf8
import logging
import sys

LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


def return_logger(verbose, name_fn):
    logger = logging.getLogger(name_fn)
    if verbose < len(LOG_LEVELS):
        logger.setLevel(LOG_LEVELS[verbose])
    else:
        logger.setLevel(logging.DEBUG)
    stdout = logging.StreamHandler(sys.stdout)
    stdout.addFilter(StdLevelFilter())
    stderr = logging.StreamHandler(sys.stderr)
    stderr.addFilter(StdLevelFilter(err=True))
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    # add formatter to ch
    stdout.setFormatter(formatter)
    stderr.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    return logger

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        import numpy as np

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta


def display_table(table_path):
    """Custom function to display the clinicadl tsvtool analysis output"""
    import pandas as pd
    from IPython.display import display

    OASIS_analysis_df = pd.read_csv(table_path, sep='\t')
    OASIS_analysis_df.set_index("diagnosis", drop=True, inplace=True)
    columns = ["n_subjects", "n_scans",
               "mean_age", "std_age", "min_age", "max_age",
               "sexF", "sexM",
               "mean_MMSE", "std_MMSE", "min_MMSE", "max_MMSE",
               "CDR_0", "CDR_0.5", "CDR_1", "CDR_2", "CDR_3"]

    # Print formatted table
    format_columns = ["subjects", "scans", "age", "sex", "MMSE", "CDR"]
    format_df = pd.DataFrame(index=OASIS_analysis_df.index, columns=format_columns)
    for idx in OASIS_analysis_df.index.values:
        row_str = "%i; %i; %.1f ± %.1f [%.1f, %.1f]; %iF / %iM; %.1f ± %.1f [%.1f, %.1f]; 0: %i, 0.5: %i, 1: %i, 2:%i, 3:%i" % tuple([OASIS_analysis_df.loc[idx, col] for col in columns])
        row_list = row_str.split(';')
        format_df.loc[idx] = row_list

    format_df.index.name = None
    display(format_df)
