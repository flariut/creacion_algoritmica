import time
import threading
import heapq

class Clock:
    """High-resolution clock that serves as the master time source"""
    
    def __init__(self):
        self._start_time = None
        self._paused_time = 0
        self._paused = True
        self._time_scale = 1.0
        self._lock = threading.RLock()
    
    def start(self):
        """Start the clock"""
        with self._lock:
            if self._paused:
                self._start_time = time.perf_counter() - self._paused_time
                self._paused = False
                self._paused_time = 0
                return True
        return False
    
    def pause(self):
        """Pause the clock"""
        with self._lock:
            if not self._paused:
                self._paused_time = self.get_time()
                self._paused = True
                return True
        return False
    
    def stop(self):
        """Stop the clock and reset"""
        with self._lock:
            self._paused = True
            self._paused_time = 0
            self._start_time = None
    
    def get_time(self):
        """Get current time in seconds"""
        with self._lock:
            if self._start_time is None or self._paused:
                return self._paused_time * self._time_scale
            return (time.perf_counter() - self._start_time) * self._time_scale
    
    def set_time_scale(self, scale):
        """Set time scaling factor (e.g., 2.0 for double speed)"""
        with self._lock:
            current_time = self.get_time()
            self._paused_time = current_time / scale
            if not self._paused:
                self._start_time = time.perf_counter() - self._paused_time
            self._time_scale = scale


class TimelineEvent:
    """Base class for events on the timeline"""
    
    def __init__(self, trigger_time):
        self.trigger_time = trigger_time
    
    def execute(self):
        """Execute this event - to be overridden by subclasses"""
        pass
    
    def __lt__(self, other):
        # For priority queue ordering
        return self.trigger_time < other.trigger_time


class PrintEvent(TimelineEvent):
    """Event that prints a character to console"""
    
    def __init__(self, trigger_time, char):
        super().__init__(trigger_time)
        self.char = char
    
    def execute(self):
        current_time = time.perf_counter()
        print(f"{self.char} (scheduled: {self.trigger_time:.3f}s, actual: {current_time:.3f}s)")


class Scheduler:
    """Scheduler that manages and executes timeline events"""
    
    def __init__(self, clock):
        self.clock = clock
        self.events = []
        self._running = False
        self._thread = None
        self._event_lock = threading.Lock()
    
    def add_event(self, event):
        """Add an event to the scheduler"""
        with self._event_lock:
            heapq.heappush(self.events, event)
    
    def start(self):
        """Start the scheduler in a separate thread"""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._run_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _run_loop(self):
        """Main scheduler loop"""
        while self._running:
            current_time = self.clock.get_time()
            
            # Process all events that are due
            with self._event_lock:
                while self.events and self.events[0].trigger_time <= current_time:
                    event = heapq.heappop(self.events)
                    event.execute()
            
            # Sleep briefly to avoid consuming too much CPU
            time.sleep(0.001)


class Sequencer:
    """A simple sequencer that schedules events at regular intervals"""
    
    def __init__(self, scheduler, bpm=120, steps=16):
        self.scheduler = scheduler
        self.bpm = bpm
        self.steps = steps
        self.sequence = [None] * steps
        self.current_step = 0
        self.step_interval = 60.0 / bpm / 4  # 16th notes
        self.next_step_time = 0
        self._running = False
    
    def set_step(self, step, char):
        """Set a character at the specified step"""
        if 0 <= step < self.steps:
            self.sequence[step] = char
    
    def start(self):
        """Start the sequencer"""
        if self._running:
            return
            
        self._running = True
        self.next_step_time = self.scheduler.clock.get_time()
        self._schedule_next_step()
    
    def stop(self):
        """Stop the sequencer"""
        self._running = False
    
    def _schedule_next_step(self):
        """Schedule the next step in the sequence"""
        if not self._running:
            return
            
        # Schedule the current step
        char = self.sequence[self.current_step]
        if char:
            event = PrintEvent(self.next_step_time, char)
            self.scheduler.add_event(event)
        
        # Calculate time for next step
        self.next_step_time += self.step_interval
        self.current_step = (self.current_step + 1) % self.steps
        
        # Schedule a callback for the next step
        delay = max(0, self.next_step_time - self.scheduler.clock.get_time())
        threading.Timer(delay, self._schedule_next_step).start()


# Example usage
if __name__ == "__main__":
    # Create our components
    clock = Clock()
    scheduler = Scheduler(clock)
    sequencer = Sequencer(scheduler, bpm=120, steps=8)
    
    # Set up a simple sequence
    sequencer.set_step(0, 'X')
    sequencer.set_step(2, 'Y')
    sequencer.set_step(4, 'Z')
    sequencer.set_step(6, '!')
    
    print("Starting sequencer... (Press Ctrl+C to stop)")
    
    try:
        # Start the clock first
        clock.start()
        
        # Start the scheduler and sequencer
        scheduler.start()
        sequencer.start()
        
        # Let it run for 10 seconds
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        sequencer.stop()
        scheduler.stop()
        clock.stop()