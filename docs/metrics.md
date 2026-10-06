# Metrics

Every simulation carries a metrics manager (`sim.metrics`) that updates after each step.

## Summary

`sim.metrics.summary()` and `SimulationResult.summary` use flattened keys:

| key | meaning |
|---|---|
| `vehicles.generated / inserted / arrived / en_route / backlog / removed / teleported` | counts |
| `travel_time.mean / median / p95 / std` | arrival - insertion, over arrived trips, s |
| `total_time_mean`, `insertion_delay_mean` | arrival - departure; insertion - departure, s |
| `att_censored` | mean total time where unfinished trips count until the end of the run, s |
| `delay.mean / median / p95` | travel time minus free-flow time (sum of dx / v0), s |
| `waiting_time_mean`, `stops_mean` | halting time and moving-to-halting transitions per trip |
| `throughput_vph` | arrivals per hour |
| `space_mean_speed`, `vkt`, `vht` | VKT / VHT, vehicle-kilometres, vehicle-hours |
| `queue.mean_total_veh`, `queue.max_lane_veh` | time-mean network queue; worst lane queue |

Trips departing before `metrics.warmup` and samples before it are left out of the summary
(counts stay totals). A vehicle halts when its speed is below `halting_speed` (0.1 m/s).

## Tables

`sim.metrics.timeseries()`, `.intersections()` and `.trips()` (also
`SimulationResult.tables`) are `{column: numpy array}` dicts:

- **timeseries**: one row per `metrics.interval` (10 s): active, backlog, generated,
  arrived, inserted and arrived in the window, halting, mean speed, VKT/VHT, space-mean
  speed, throughput, waiting (vehicle-seconds), mean and max network queue.
- **intersections**: per sample and junction: stop-line crossings, throughput, mean and max
  queue of the incoming lanes, current phase.
- **trips**: one row per arrived vehicle: ids, type, origin and destination roads, times,
  delay, waiting time, stops, distance.

`result.export(dir, format="csv" | "json" | "parquet")` writes them (Parquet needs polars,
the `data` extra); `urbanflow.metrics.read_table(path)` reads them back.

## Live values

`sim.metrics.latest()` is the last sample; `sim.metrics.history(max_points)` the thinned
timeseries (the workbench uses both).
