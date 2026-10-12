// The help panel: what you see, what the numbers mean, the signal strategies, and how to
// bring your own data in and take results out.

import { brain, brainChoices } from "../lib/friendly";

const section = "space-y-2";
const h = "text-lg font-semibold text-slate-900";

export function Help({ controllers, onClose }: { controllers: string[]; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/40" onClick={onClose} role="dialog" aria-modal aria-label="Help">
      <div className="h-full w-full max-w-xl space-y-6 overflow-y-auto bg-white p-6 text-sm text-slate-700 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-slate-900">❓ How UrbanFlow works</h2>
          <button onClick={onClose} className="rounded-full px-3 py-1 text-slate-500 hover:bg-slate-100" aria-label="Close help">✕</button>
        </div>

        <div className={section}>
          <h3 className={h}>What it is</h3>
          <p>
            UrbanFlow simulates every car on a road network: how it speeds up, slows down, changes lanes and waits at
            traffic lights. You can try different ways of running the traffic lights and see which one gets people where
            they are going fastest.
          </p>
        </div>

        <div className={section}>
          <h3 className={h}>Getting started</h3>
          <ol className="list-decimal space-y-1 pl-5">
            <li>Choose a map (or import your own).</li>
            <li>Choose how the traffic lights work and how much traffic there is.</li>
            <li>Press <b>Start simulation</b>, then <b>▶ Play</b>.</li>
            <li>Click a car to follow it, or a junction to see its traffic light.</li>
          </ol>
        </div>

        <div className={section}>
          <h3 className={h}>On the map</h3>
          <ul className="space-y-1">
            <li>🟢 green dot: cars may go · 🟡 yellow: about to change · 🔴 red: stop</li>
            <li>Blue = car, yellow = bus, purple = truck, red/blue flashing = ambulance</li>
            <li>Yellow roads: where the car you clicked will drive next</li>
            <li>Scroll to zoom, drag to move, double-click (or <b>Show the whole map</b>) to see everything</li>
          </ul>
        </div>

        <div className={section}>
          <h3 className={h}>The numbers</h3>
          <ul className="space-y-1">
            <li><b>How is traffic?</b> The share of cars standing still: under 20% flows nicely, over 60% is a jam.</li>
            <li><b>Cars driving now</b> and <b>cars standing still</b>: right now, on the whole map.</li>
            <li><b>Average speed</b>: of all moving and waiting cars, in km/h.</li>
            <li><b>Cars that got there</b>: trips finished since the start.</li>
            <li>The small lines show the last few minutes; point at them to read a value.</li>
          </ul>
        </div>

        <div className={section}>
          <h3 className={h}>Traffic-light strategies</h3>
          <ul className="space-y-1">
            {brainChoices(controllers).map((c) => {
              const b = brain(c);
              return <li key={c}>{b.icon} <b>{b.title}</b>: {b.blurb}</li>;
            })}
          </ul>
          <p>You can switch strategy while the simulation runs, or take over a single light by clicking its junction.</p>
          <p><b>Compare</b> runs the same traffic with several strategies and ranks them by average trip time.</p>
        </div>

        <div className={section}>
          <h3 className={h}>Your own data</h3>
          <p>
            <b>Import your own map</b> on the start screen accepts an UrbanFlow scenario file or CityFlow's road network and
            flow files. Anything that could not be converted exactly is listed after the import.
          </p>
        </div>

        <div className={section}>
          <h3 className={h}>Keeping results</h3>
          <ul className="space-y-1">
            <li><b>⬇ Download</b> gives you every finished trip, the network every 10 seconds, each junction, and the map itself as files you can open in a spreadsheet.</li>
            <li><b>⏺ Record</b> saves a recording you can watch again (start screen, Advanced options) or download.</li>
          </ul>
        </div>

        <div className={section}>
          <h3 className={h}>Keyboard</h3>
          <ul className="space-y-1">
            <li><kbd className="rounded border px-1">Space</kbd> play / pause · <kbd className="rounded border px-1">F</kbd> show the whole map · <kbd className="rounded border px-1">Esc</kbd> close this panel</li>
          </ul>
        </div>

        <p className="text-xs text-slate-500">
          The simulation uses published traffic models (IDM car following, MOBIL lane changing) and has not been
          calibrated to any real city: use it to compare options, not to predict exact travel times.
        </p>
      </div>
    </div>
  );
}
