# Robot Simulation
I want a simulation of a ground-based autonomous robot. It moves around its environment collecting litter and avoiding people. It is about 60cm by 60cm and is 50cm high, and has four big robust wheels.

The sim needs to support running with a visualisation of what's happening (simple 3d please), or "headless" i.e. with no vis. Visualisation mode needs to run in real time, with the option to speed it up or slow it down by 2x, 4x etc. Headless mode can run without any link to real time - have it run as fast as the hardware can manage.

Regardless of the running mode, a given run must be deterministic, at least on the current seed. The events and end state must be identical regardless of run mode.

Each run happens in a world with varied object and terrain. Initially, have this controlled by a random generator that uses a seed - the terrain for a given run should be deterministic on its seed.

On each step, the sim checks for safety violations e.g. getting within 1 metre of a person, and logs these.

The sim can run a "job" of multiple runs - this is always headless. For now, each run within a job is just a different random situation based on its seed.

After a job, save details about each run (map run number to random seed) so that there is a way to ask the system for a rerun. 

Command line interface:

>robot-sim

runs a single run, with a random situation, with visualisation. It then plays a happy tune (to signal the end of the job) and prints out a list of runs and the safety violations that occurred on each run. 

>robot-sim new-job 10

runs a job of ten runs headlessly, each with a different random situation. It then prints out a list of runs and the safety violations that occurred on each run.

>robot-sim rerun 8

re-runs run 8 of the most recent job, with visualisation

# Implementation
Python please. Conside if SimPy library would be helpful, also consider other options in a similar space. Please use a discrete time model with, initially 30 steps per simulated second. I suspect that the simulation proper needs to be single-threaded.