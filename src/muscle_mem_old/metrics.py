import copy
import json
import time
from contextlib import contextmanager


class Metrics:
    def __init__(self):
        self.enabled = False
        self.metrics = {
            "query": {"total_time": 0, "count": 0},
            "filter": {"partials": {"total_time": 0, "count": 0}, "capture": {"total_time": 0, "count": 0}, "compare": {"total_time": 0, "count": 0}},
            "runtime": {
                "precheck": {"capture": {"total_time": 0, "count": 0}, "compare": {"total_time": 0, "count": 0}},
                "postcheck": {"capture": {"total_time": 0, "count": 0}, "compare": {"total_time": 0, "count": 0}},
            },
        }

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def report(self):
        """Print a pretty-printed JSON report of all metrics."""
        if not self.enabled:
            return

        # Create a copy of metrics with averages and formatted times added
        report_metrics = copy.deepcopy(self.metrics)

        # Format time in appropriate units and add averages
        def format_time(seconds):
            """Format time in appropriate units based on magnitude."""
            if seconds >= 1.0:
                return f"{seconds:.4f}s"  # seconds
            elif seconds >= 0.001:
                return f"{seconds * 1000:.4f}ms"  # milliseconds
            else:
                return f"{seconds * 1000000:.4f}us"  # microseconds

        def process_metrics(metrics_dict):
            for value in metrics_dict.values():
                if isinstance(value, dict):
                    if "total_time" in value and "count" in value:
                        # Store original time values for calculation
                        raw_time = value["total_time"]

                        # Format the time value
                        value["total_time"] = format_time(raw_time)

                        # Calculate and format average if count > 0
                        if value["count"] > 0:
                            avg_time = raw_time / value["count"]
                            value["avg_time"] = format_time(avg_time)
                    else:
                        process_metrics(value)

        process_metrics(report_metrics)
        print(json.dumps(report_metrics, indent=4))

    def reset(self):
        """Reset all metrics to zero."""
        self.__init__()

    @contextmanager
    def measure(self, *path):
        if not self.enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            # Navigate to the correct spot in the metrics dictionary
            current = self.metrics
            for part in path[:-1]:
                current = current[part]

            # Update time and count
            current[path[-1]]["total_time"] += time.time() - start
            current[path[-1]]["count"] += 1
