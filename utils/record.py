# import pdb
import time
from tqdm import tqdm


class Recorder(object):
    def __init__(self, work_dir, print_log, log_interval):
        """Initialize the recorder object.
        
        Args:
            work_dir (str): The directory where the log file will be saved.
            print_log (bool): Whether to print the log to the console.
            log_interval (int): The interval at which to print the log.
        """
        self.cur_time = time.time()
        self.print_log_flag = print_log
        self.log_interval = log_interval
        self.log_path = '{}/log.txt'.format(work_dir)
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

    def print_time(self):
        """Print the current local time to the log.
        
        Args:
            str (str): The message to be printed.
            path (str): The path to the log file.
            print_time (bool): Whether to print the time.
        """
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, path=None, print_time=True):
        """
        Print the log message to the console and, if enabled, to a log file.

        Args:
            str (str): The message to be printed.
            path (str): The path to the log file.
            print_time (bool): Whether to print the time.
        """
        if path is None:
            path = self.log_path
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        tqdm.write(str)
        if self.print_log_flag:
            with open(path, 'a') as f:
                f.writelines(str)
                f.writelines("\n")

    def record_time(self):
        """
        Save the current timestamp as a reference for duration measurement.

        Returns:
            float: The current time in seconds.
        """
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        """Calculate the elapsed time since the last call to record_time().

        Process:
        1. Calculate the time that has passed since cur_time.
        2. Update cur_time to the current time.
        """
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def timer_reset(self):
        """
        Reset all time counters to their initial values.
        """
        self.cur_time = time.time()
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

    def split_time(self):
        """
        Calculate the elapsed time since the last call to record_time().
        """
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def timer_reset(self):
        """
        Reset all time counters to their initial values.
        """
        self.cur_time = time.time()
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

    def record_timer(self, key):
        """
        Add the duration of the last interval to a specific timer category.

        Args:
            key (str): Name of the timer category, e.g., dataloader, device, forward, or backward.

        Process:
        1. Calculate the duration since the last record.
        2. Add the duration to timer[key].
        """
        self.timer[key] += self.split_time()

    def print_time_statistics(self):
        """
        Print the percentage of time used for each process stage.

        Process:
        1. Calculate the proportion of time for each category relative to the total time.
        2. Write a summary of the time distribution to the log.
        """
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(self.timer.values()))))
            for k, v in self.timer.items()}
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [GPU]{device}, [Forward]{forward}, [Backward]{backward}'.format(
                **proportion))
