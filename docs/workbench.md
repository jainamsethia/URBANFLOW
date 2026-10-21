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

## While it runs

- **Controls**: Play / Pause (Space), speed (slow, normal, fast, as fast as possible),
  the signal control strategy for every light, ⏺ Record / ⏹ Stop recording, ⬇ Download,
  Restart, and *Choose another map*.
- **Map**: drag to pan, scroll to zoom, double-click or F to see the whole map. Cars, buses,
  trucks and (flashing) ambulances are drawn to scale; each lane's traffic light is a
  coloured dot; bus stops are 🚏 signs at the curb. A legend explains the colours.
- **Click a car**: where it is going, its speed, how long it waited at red lights and how
  often it stopped; the roads it will take next are shaded yellow. A bus also shows its next
  stop, or how long it still waits at the stop it is at.
- **Click a junction**: which directions may go now, the light's strategy and when it
  changes; *Manual control* gives green to a chosen direction until *Back to automatic*.
- **📊 Traffic**: how traffic feels (share of cars standing still), cars driving, cars
  standing still, average speed (km/h) and finished trips, each with a trend line.
- **🏁 Compare**: runs the same traffic with several strategies and ranks them by average
  trip time; *Show detailed statistics* gives means, 95 % intervals and paired changes, and
  the results download as CSV.
- **⚙️ Advanced**: step through time, colour the map by lane density, speed, queue,
  congestion, occupancy, utilisation or flow, and the raw numbers.

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
