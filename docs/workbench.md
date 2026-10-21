# Web workbench

```bash
npm --prefix frontend install && npm --prefix frontend run build   # once, from a checkout
uv run urbanflow serve --open
```

or, with nothing but Docker installed, `docker compose up --build` and open the link it
prints.

The workbench is a plain-language front end to everything below; the **❓ Help** button
explains each part in the app.

## Start screen

1. **Choose a map**: the bundled maps (one crossing, small and big grid towns, main street,
   ambulance rescue, rush hour, bus line, downtown, packed town), your imported maps (with *Download* and *Remove*), or
   *Build a grid* (streets across and down, a few / some / lots of cars).
2. **How should the traffic lights work?** The map's own plan, or Timer (`fixed_time`),
   Sensor (`actuated`), Smart (`max_pressure`), Planner (`webster`) or Ambulance helper
   (`preemption`).
3. **How much traffic?** Light (x0.5), Normal, Heavy (x1.5) or Very heavy (x2) scales every
   flow of the map; *Drivers avoid jams* turns on the `dynamic` router (drivers learn
   travel times and take a faster route when a road gets slow, see
   [Routing](simulation-model.md#routing)); *Record this run* saves a replay from the
   start.

**Import your own map** accepts:

- an **OpenStreetMap** export (`map.osm` from openstreetmap.org → *Export*): streets become
  roads and lanes, traffic-signal nodes become signalised junctions and traffic is added
  between the map's edges (see [Importing OpenStreetMap](scenario-format.md#importing-openstreetmap));
- an **UrbanFlow scenario** (`.json`);
- **CityFlow** road network and flow files (plus an optional config).

Problems are shown in plain words with the exact place in the file; anything converted
approximately is listed after the import. *Advanced options* holds the random seed and the
saved recordings (watch or download).

## Draw your own map

*✏️ Draw your own map* on the start screen opens the map editor on an empty map; *✏️ Edit*
on any map opens it (a bundled map is saved as a copy, `<name>-copy`).

- **Tools**: *👆 Select / move* (click to select, drag a junction to move it), *🛣️ Draw
  road* (click junctions or empty ground one after the other: each click adds a road both
  ways with the chosen number of lanes; Esc stops), *📍 Add junction* and *🧽 Erase*. Scroll
  to zoom, drag the ground to pan; positions snap to 5 m.
- **A junction**: its type (automatic, traffic light, give way, no control, or an edge of
  the map where cars enter and leave; automatic picks edge for a dead end, no control for a
  plain bend and a traffic light otherwise), and for lights the strategy and green time.
- **A road**: lanes and speed limit; remove one direction or both.
- **Traffic**: flows from an edge of the map to another, in cars per hour, editable.
- **Problems**: what still stops the map from running, in plain words; click one to select
  the junction or road it is about. *Save* and *▶ Try it* (save, then run it) work once
  there are none.
- *↶ Undo* / *↷ Redo* (Ctrl+Z / Ctrl+Y), and *{ } JSON* to see or paste the whole scenario
  document.

Every change is an editor operation that the server applies and validates
(`POST /api/v1/scenarios/ops` with `{"scenario": ..., "ops": [...]}`: the result, whether it
is valid and its issues; `POST /api/v1/scenarios/generate` returns a generated map to start
from), so the map you see is exactly what will run. The operations are listed in the
[Python API](python-api.md#editing-operations).

## While it runs

- **Controls**: Play / Pause (Space), speed (slow, normal, fast, as fast as possible),
  the signal control strategy for every light, ⏺ Record / ⏹ Stop recording, ⬇ Download,
  Restart, and *Choose another map*.
- **Map**: drag to pan, scroll to zoom, double-click or F to see the whole map. Cars, buses,
  trucks and (flashing) ambulances are drawn to scale; each lane's traffic light is a
  coloured dot; bus stops are 🚏 signs at the curb. A legend explains the colours. When the
  map is larger than the view, an overview in the corner shows where you are; click it to
  move there.
- **Click a car**: where it is going, its speed, how long it waited at red lights and how
  often it stopped; the roads it will take next are shaded yellow. A bus also shows its next
  stop, or how long it still waits at the stop it is at.
- **Click a junction**: which directions may go now, the light's strategy and when it
  changes; *Manual control* gives green to a chosen direction until *Back to automatic*.
- **Click a road**: where it goes, its lanes, length and speed limit, the cars on it and
  standing still, their average speed, how full it is and the time to drive it.
- **📊 Traffic**: how traffic feels (share of cars standing still), cars driving, cars
  standing still, average speed (km/h) and finished trips, each with a trend line, and
  **📜 What just happened**: every traffic light change ("green for north & south") and every
  stuck car the watchdog moved on, newest first, with filters; click one to select it.
- **🏁 Compare**: runs the same traffic with several strategies and ranks them by average
  trip time; *Show detailed statistics* gives means, 95 % intervals and paired changes, and
  the results download as CSV.
- **⚙️ Advanced**: step through time, and colour the map: vehicles by type, by speed, by
  where they are going (the side of town of their destination) or greyed out with the
  traffic lights drawn large; or the roads by how crowded, average speed, queues, how slowed
  down, how much is covered by cars, how full, or cars per hour; plus the raw numbers.

Recordings play back the same way, with the timeline to jump around, ±1 s steps, the road
colourings (computed from the recorded vehicles: the queue counts the cars standing still on
a lane, and "how slowed down" compares speeds with the speed limit), road views and the light
changes in the event log.

## Downloads

| Item | Contents |
|---|---|
| Every finished trip (CSV) | one row per arrived vehicle: route ends, departure and arrival time, travel time, delay, waiting time, stops, distance |
| The network every 10 seconds (CSV) | vehicles, waiting, arrivals, speeds, queues, throughput |
| Each junction every 10 seconds (CSV) | crossings, throughput, queues, phase |
| This map (JSON) | the scenario, re-importable here or usable with the Python API |
| Last recording (`.ufr`) | the replay; open it with `urbanflow replay FILE --view` |

The server binds to 127.0.0.1 by default. `urbanflow serve --host 0.0.0.0` shares it on
your network behind an access token: open the printed `#token=...` link, or paste the
token when the workbench asks for it. See `urbanflow serve` in [Command line](cli.md). API reference: `http://127.0.0.1:8000/api/v1/docs`.

## Experiments

*🧪 Experiments* on the start screen keeps [experiments](python-api.md#experiments) in the
workspace: each traffic-light strategy runs several times on the same map, every repeat
with different random traffic (the same traffic for every strategy, so the comparison is
paired), in parallel worker processes.

- **New experiment**: a map (bundled or your own), the strategies to test (the first is
  the reference the others are compared with), the amount of traffic, the simulated time,
  the number of repeats, and whether to record the first repeat of each strategy.
- **The list** shows every experiment with its status and progress; it keeps running while
  you watch a simulation or draw a map.
- **An experiment**: live progress (how much of the work is done), *Stop*, *Run the
  unfinished* (completed runs are kept), *Delete* (with its result files).
- **Results**: for each measure (trip time, time lost, time standing still, stops, cars
  arriving per hour, queues) a bar per strategy with its 95 % range, and a sentence saying
  which did best and whether the difference held across the repeats (a paired 95 %
  interval that excludes zero) or may be luck. *Compare against* changes the reference;
  *Show all the numbers* lists every measure with the paired differences.
- **Over time**: cars driving, standing still, average speed, queues and cars waiting to
  enter, averaged over the repeats with a shaded 95 % band.
- **Every simulation**: each run's status, how long it took, its error if it failed, and
  *▶ Watch* for recorded runs (the replay viewer; closing it returns to the experiment).
- **Downloads**: the comparison and every run's results as CSV or JSON.

## Experiments over the API

The server runs [experiments](python-api.md#experiments) for scripts and dashboards. They
are queued and run **one at a time** (each with its own pool of worker processes); clients
poll for progress. Every endpoint is under `/api/v1`:

| endpoint | does |
|---|---|
| `POST /experiments` `{"spec": {...}, "run": false}` | validate (422 with JSON paths such as `variants[0].name`), expand and store as `draft`; `"run": true` queues it at once. At most 2000 runs per experiment (use the CLI beyond that) |
| `GET /experiments?status=&limit=&offset=` | a page of summaries: status, variants, seeds and run counts per status |
| `GET /experiments/{id}` | the spec and every run's status and progress |
| `POST /experiments/{id}/run` `{"resume": true}` | queue it (skipping completed runs unless `resume` is false); 409 while queued or running |
| `POST /experiments/{id}/cancel` | drop it from the queue, or stop it: running workers stop at their next check |
| `DELETE /experiments/{id}?purge=false` | remove it (and with `purge` its run directories); 409 while active |
| `GET /experiments/{id}/compare?baseline=&metric=` | means, 95 % CIs and paired differences of the completed runs |
| `GET /experiments/{id}/timeseries?column=active&max_points=400` | per variant, the mean and 95 % CI over seeds of one `timeseries` column (`active`, `halting`, `speed_mean`, `queue_mean`, `backlog`, ...) |
| `GET /experiments/{id}/export?what=runs\|comparison&format=csv\|json` | a download |
| `GET /runs?experiment_id=`, `GET /runs/{run_id}` | runs, one run with its summary, config and policy |
| `GET /runs/{run_id}/artifacts`, `GET /runs/{run_id}/artifacts/{name}` | the run directory's files (`summary.json`, `tables/trips.csv`, `replay.ufr`, ...); only listed names are served |

Experiments that were queued or running when the server stopped are marked `interrupted`
when it starts again; `POST /experiments/{id}/run` continues them.
