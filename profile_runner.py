import cProfile
import pstats
import sys

# Patch sys.argv
sys.argv = ["main.py", "--mock"]

import main
import itertools

def run_limited():
    try:
        # Patch stream to only yield 100 frames
        original_process = main.engine.process_frame
        
        frame_counter = [0]
        
        def mock_process_frame(*args, **kwargs):
            if frame_counter[0] >= 100:
                main.engine._running = False
                return None
            frame_counter[0] += 1
            return original_process(*args, **kwargs)
            
        main.engine.process_frame = mock_process_frame
        
        main.main()
    except Exception as e:
        print("Error during execution:", e)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_limited()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(40)
