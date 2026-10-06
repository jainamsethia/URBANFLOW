# Replay

A replay (`.ufr`) records every step's vehicle positions, speeds, links and signal states,
so a run can be watched again, scrubbed and rewound without re-simulating.

```python
from urbanflow import Simulation, bundled
from urbanflow.replay import ReplayReader

sim = Simulation.from_scenario(bundled("grid_3x3"), duration=600, record="grid.ufr")
sim.run()
path = sim.stop_recording()

replay = ReplayReader(path)
frame = replay.frame(300)  # vehicles at step 300 (uid, xy, heading, speed, ...)
print(replay.n_frames, frame.n, replay.vehicles()[int(frame.uid[0])])
```

- Record with `Simulation(record=path)`, `sim.start_recording(path)` / `stop_recording()`,
  or `urbanflow run SCENARIO --record`. `record.every` records every n-th step.
- A recording is first a directory `NAME.ufr.d/` (readable even after a crash), packed into
  the ZIP `NAME.ufr` when it stops. It contains a manifest, the scenario, the render
  geometry, a vehicle table, chunks of binary UFB1 frames, the metrics tables and the summary.
- `ReplayReader` validates every member name, size and version before reading (it never
  unpickles) and gives random access by step with a small chunk cache.
- `urbanflow replay FILE` prints the manifest; `--view` opens it in the workbench, which
  plays, pauses, steps forward and back, rewinds and seeks with a timeline slider.

Frames carry no per-lane heat-map values; replay views show vehicles and signals.
