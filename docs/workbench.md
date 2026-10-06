# Web workbench

```bash
npm --prefix frontend install && npm --prefix frontend run build   # once
uv run urbanflow serve --open
```

- **Sessions**: pick a scenario (bundled or from `<workspace>/scenarios/`), a signal
  controller and a seed, then *New session*. Several sessions can run at once.
- **Controls**: play / pause, step +1 / +60, reset, speed (steps per second or max),
  switch every signal's controller live.
- **Map**: drag to pan, scroll to zoom, double-click to fit. Vehicles are drawn to scale and
  interpolated between frames; signal heads show each movement's state.
- **Views**: vehicles by type or speed; lane heat maps of density, speed, queue,
  congestion, occupancy, utilisation or flow.
- **Inspector**: click a vehicle (road, lane, speed, waiting time, stops, destination) or an
  intersection (controller, phase, time left, movement states; hold a phase, release).
- **KPIs**: live charts of vehicles on the network, mean speed, network queue and
  throughput from the metrics manager.
- **Compare controllers**: runs controllers x seeds in the background and shows means and
  paired differences.
- **Replays**: *Open replay* lists `<workspace>/replays/*.ufr` and `runs/*/replay.ufr`; the
  timeline slider seeks, -1 steps back and negative speeds rewind.

The server binds to 127.0.0.1 by default; see `urbanflow serve` in [Command line](cli.md)
for tokens and other hosts. API reference: `http://127.0.0.1:8000/api/v1/docs`.
