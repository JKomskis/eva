from threading import Lock, Thread
import time
import os

class TraceCollector(object):
    """
    Utility class for logging traces to a file
    """

    _instance = None

    _lock: Lock = Lock()
    _log_folder = "traces/"
    _log_file_path = ""
    _log_file = None
    _tracked_batches = set()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TraceCollector, cls).__new__(cls)
                os.makedirs("traces/", exist_ok=True)
        return cls._instance
    
    def start_new_trace(self):
        if self._log_file is not None:
            self._log_file.close()
        self._log_file_path = f"{self._log_folder}trace_{time.time_ns() // 1000}.log"
        self._log_file = open(self._log_file_path, "a")
        pass

    def close_trace(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
    
    def log_fix(self, file_url: str, batch):
        if self._log_file is not None:
            self._log_file.write(f"Fix,size {batch.batch_size},batch ({id(batch)}),frames {batch.id_intervals()}\n")
            self._tracked_batches.add(id(batch))
    
    def log_unfix(self, file_url: str, batch):
        if self._log_file is not None:
            self._log_file.write(f"Unfix,size {batch.batch_size},batch ({id(batch)}),frames {batch.id_intervals()}\n")
    
    def log_free(self, batch):
        if self._log_file is not None and id(batch) in self._tracked_batches:
            self._log_file.write(f"Free,size {batch.batch_size},batch ({id(batch)}),frames {batch.id_intervals()}\n")
            self._tracked_batches.remove(id(batch))