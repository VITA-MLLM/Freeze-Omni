from dataclasses import dataclass
import json
import torch
import threading
import queue
import subprocess
import concurrent.futures

import numpy as np
import torchaudio.compliance.kaldi as k
import soundfile as sf

class PCMQueue:
    def __init__(self):
        """
        Initialize the PCMQueue with an empty queue, an empty buffer, and a lock for thread safety.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        self.queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()

    def put(self, items):
        """
        Add items to the buffer in a thread-safe manner.
        
        Parameters:
        - items (list or array-like): The items to be added to the buffer, a numpy array of dtype np.float32.
        
        Returns:
        - None
        """
        with self.lock:
            self.buffer = np.concatenate((self.buffer, np.array(items, dtype=np.float32)))

    def get(self, length):
        """
        Retrieve a specified number of elements from the buffer in a thread-safe manner.
        
        Parameters:
        - length (int): The number of elements to retrieve from the buffer.
        
        Returns:
        - numpy.ndarray or None: A numpy array containing the requested number of elements if available, otherwise None.
        """
        with self.lock:
            if len(self.buffer) < length:
                return None
            result = self.buffer[:length]
            self.buffer = self.buffer[length:]
            return result
    
    def clear(self):
        """
        Clear the buffer in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)

    def has_enough_data(self, length):
        """
        Check if the buffer contains enough data to fulfill a request of a specified length.
        
        Parameters:
        - length (int): The number of elements required.
        
        Returns:
        - bool: True if the buffer contains enough data, False otherwise.
        """
        with self.lock:
            return len(self.buffer) >= length

class ThreadSafeQueue:
    def __init__(self):
        """
        Initialize the ThreadSafeQueue with an empty queue and a lock for thread safety.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        self._queue = queue.Queue()
        self._lock = threading.Lock()

    def put(self, item):
        """
        Add an item to the queue in a thread-safe manner.
        
        Parameters:
        - item (any): The item to be added to the queue.
        
        Returns:
        - None
        """
        with self._lock:
            self._queue.put(item)

    def get(self):
        """
        Retrieve an item from the queue in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - any or None: The retrieved item if the queue is not empty, otherwise None.
        """
        with self._lock:
            if not self._queue.empty():
                return self._queue.get()
            else:
                return None

    def clear(self):
        """
        Clear the queue in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        with self._lock:
            while not self._queue.empty():
                self._queue.get()

    def is_empty(self):
        """
        Check if the queue is empty in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - bool: True if the queue is empty, False otherwise.
        """
        with self._lock:
            return self._queue.empty()

    def size(self):
        """
        Get the current size of the queue in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - int: The number of items currently in the queue.
        """
        with self._lock:
            return self._queue.qsize()
