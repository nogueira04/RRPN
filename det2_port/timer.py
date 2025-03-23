from time import time
from collections import defaultdict
from tabulate import tabulate

class Timer:
    def __init__(self):
        self.timings = defaultdict(list)

    class _TimerContext:
        """Internal context manager for measuring execution time."""
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name

        def __enter__(self):
            self.start_time = time()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            elapsed = (time() - self.start_time) * 1000  
            self.parent.timings[self.name].append(elapsed)

    def time(self, name):
        """Returns a context manager for timing a block of code."""
        return self._TimerContext(self, name)

    def get_last_time(self, name):
        return self.timings[name][-1] if self.timings[name] else 0.0

    def get_mean_time(self, name):
        return sum(self.timings[name]) / len(self.timings[name]) if self.timings[name] else 0.0

    def get_total_time(self, name):
        return sum(self.timings[name]) if self.timings[name] else 0.0

    def get_pipeline_total_time(self):
        """Calculate the total time spent across all steps in the pipeline."""
        return sum(self.get_total_time(step) for step in self.timings)

    def print_summary(self):
        """Prints a structured summary of recorded times."""
        table_data = []
        for step in self.timings:
            table_data.append([
                step,
                # f"{self.get_last_time(step):.3f}",
                f"{self.get_mean_time(step):.3f}",
                f"{self.get_total_time(step):.3f}",
            ])
        
        total_time = self.get_pipeline_total_time()
        table_data.append(["Total Pipeline Time", "-", f"{total_time:.3f}"])

        print("\n" + tabulate(
            table_data,
            headers=["Name", "Mean Time (ms)", "Total Time (ms)"],
            tablefmt="grid",
            numalign="right",
        ))

