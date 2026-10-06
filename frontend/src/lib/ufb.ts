// UFB1 binary frame decoder (plan K.10.5): zero-copy typed-array views over the buffer.

export const MAGIC = 0x31424655; // "UFB1" read as little-endian u32
const align4 = (o: number) => (o + 3) & ~3;

export const FLAG_HALTING = 1;
export const FLAG_BRAKING = 2;
export const FLAG_HELD = 32;

export interface FrameView {
  seq: number;
  step: number;
  time: number;
  laneMetric: number;
  geometryCrc: number;
  n: number;
  uid: Uint32Array;
  xy: Float32Array;
  heading: Float32Array;
  speed: Float32Array;
  link: Uint32Array;
  flags: Uint8Array;
  signals: Uint8Array | null;
  laneValues: Float32Array | null;
}

export function decodeFrame(buf: ArrayBuffer): FrameView {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("not a UFB1 frame");
  if (dv.getUint16(4, true) !== 1) throw new Error("unsupported UFB1 version");
  const H = dv.getUint16(6, true);
  const fl = dv.getUint8(9);
  const n = dv.getUint32(32, true);
  const m = dv.getUint32(36, true);
  const L = dv.getUint32(40, true);
  let o = H;
  const uid = new Uint32Array(buf, o, n);
  o += 4 * n;
  const xy = new Float32Array(buf, o, 2 * n);
  o += 8 * n;
  const heading = new Float32Array(buf, o, n);
  o += 4 * n;
  const speed = new Float32Array(buf, o, n);
  o += 4 * n;
  const link = new Uint32Array(buf, o, n);
  o += 4 * n;
  const flags = new Uint8Array(buf, o, n);
  o = align4(o + n);
  let signals: Uint8Array | null = null;
  if (fl & 1) {
    signals = new Uint8Array(buf, o, m);
    o = align4(o + m);
  }
  const laneValues = fl & 2 ? new Float32Array(buf, o, L) : null;
  return {
    seq: dv.getUint32(12, true),
    step: Number(dv.getBigInt64(16, true)),
    time: dv.getFloat64(24, true),
    laneMetric: dv.getUint16(10, true),
    geometryCrc: dv.getUint32(44, true),
    n,
    uid,
    xy,
    heading,
    speed,
    link,
    flags,
    signals,
    laneValues,
  };
}
