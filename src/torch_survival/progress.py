import optuna
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn


class OptunaProgressCallback:
    def __init__(self, model_name, n_trials, verbose=1):
        self.verbose = verbose
        if self.verbose > 0:
            # At verbosity = 1 we show a progress bar for trials
            self.progress = Progress(
                TextColumn('{task.description}'),
                BarColumn(),
                TextColumn('{task.fields[score]}'),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            )
            self.tune_task = self.progress.add_task('[green]Optimizing ' + model_name, total=n_trials, score='-.--')

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.verbose > 0:
            score = '{:.2f}'.format(abs(study.best_value))
            self.progress.update(self.tune_task, advance=1, score=score)

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.stop()
