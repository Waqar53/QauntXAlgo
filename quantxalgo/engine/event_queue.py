"""
Event Queue for the event-driven backtester.

Priority queue for ordering events by timestamp and type.
"""

import heapq
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from threading import Lock

from quantxalgo.core.events import Event

T = TypeVar("T", bound=Event)


class EventQueue(Generic[T]):
    """Thread-safe priority queue for events.
    
    Events are ordered by:
    1. Timestamp (earliest first)
    2. Event type priority (risk > fill > order > signal > market_data)
    
    Example:
        >>> queue = EventQueue[Event]()
        >>> queue.put(market_event)
        >>> queue.put(signal_event)
        >>> event = queue.get()  # Returns event with earliest timestamp
    """
    
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._lock = Lock()
        self._counter = 0  # For stable sorting
    
    def put(self, event: T) -> None:
        """Add an event to the queue.
        
        Args:
            event: Event to add.
        """
        with self._lock:
            # Use counter for stable sorting when timestamps match
            heapq.heappush(self._heap, event)
    
    def get(self) -> T:
        """Remove and return the next event.
        
        Returns:
            The next event in priority order.
            
        Raises:
            IndexError: If queue is empty.
        """
        with self._lock:
            return heapq.heappop(self._heap)
    
    def peek(self) -> T:
        """Return the next event without removing it.
        
        Returns:
            The next event in priority order.
            
        Raises:
            IndexError: If queue is empty.
        """
        with self._lock:
            return self._heap[0]
    
    def empty(self) -> bool:
        """Check if queue is empty.
        
        Returns:
            True if queue has no events.
        """
        with self._lock:
            return len(self._heap) == 0
    
    def size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of events in queue.
        """
        with self._lock:
            return len(self._heap)
    
    def clear(self) -> None:
        """Remove all events from queue."""
        with self._lock:
            self._heap.clear()
    
    def __len__(self) -> int:
        return self.size()
    
    def __bool__(self) -> bool:
        return not self.empty()
