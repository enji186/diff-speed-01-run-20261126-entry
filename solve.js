// A faster drop-in for makefun-diff 1.0.0 (make.fun's own line diff).
// run(input) returns exactly what makefun-diff 1.0.0's diffPatch returns for
// { old, new }: the same patch text, byte for byte. It is a fresh
// implementation of the same rule (RULES.txt / the reference header):
// lines cut on '\n' only, common prefix then suffix kept, Myers's greedy
// search with deletion-before-insertion on ties, no edge pruning, hunks with
// 3 context lines, runs at most 6 kept lines apart merged.
//
// Speed comes from removing the reference's costs, not from changing the
// result:
//  - lines are interned to integer ids (one shared Map per case), so the
//    hot snake loop compares ints instead of strings;
//  - the Myers search runs on Int32Array V rows (value s = x + 2, 0 = no
//    path) with no per-step object allocation, no bounds checks (guard
//    slots), and a compact per-depth trace for backtracking;
//  - the patch is built with array push + join, as the reference does.

const FORMAT = 'makefun-diff/1';
const CONTEXT = 3;
const KEEP = 0;
const DELETE = 1;
const INSERT = 2;

/**
 * A text's lines as integer ids. Lines are cut after every '\n' (the only
 * separator); a last line with no '\n' is a line too; the empty text has no
 * lines. Equal lines (same code units, '\n' or its absence included) share
 * one id; `strs` maps each id back to its line string.
 */
function tokenizeIds(text, intern) {
  const ids = [];
  let start = 0;
  for (;;) {
    const idx = text.indexOf('\n', start);
    if (idx === -1) {
      if (start < text.length) ids.push(intern(text.slice(start)));
      break;
    }
    ids.push(intern(text.slice(start, idx + 1)));
    start = idx + 1;
  }
  return Int32Array.from(ids);
}

/**
 * Myers's greedy search on the middle line-id arrays A (length N) and B
 * (length M). Returns { D, trace }: D the edit distance, trace[d] an
 * Int32Array(2d + 5) holding, at index k + d + 2, s = x + 2 where x is the
 * furthest old index reached on diagonal k after d edits and its snake
 * (0 = no path on that diagonal at that depth; the two guard slots on each
 * side stay 0, so reads just outside the live diagonals are safe).
 * The search, its tie-break (deletion wins) and its edge guards are the
 * reference's, step for step.
 *
 * Two speed devices, neither of which changes a value: the two working
 * arrays carry one extra slot at each end holding a sentinel id no real
 * line has (a different one for A and for B), so the snake's equality
 * check alone stops it at either text's end and needs no bounds checks;
 * and the end is only reachable on diagonal N - M, so it is tested once
 * per depth, after the diagonal loop, instead of once per diagonal.
 */
function myersForward(A, B, sA, sB) {
  const N = A.length;
  const M = B.length;
  const Ap = new Int32Array(N + 2);
  Ap.set(A);
  Ap[N] = sA;
  Ap[N + 1] = sA + 1;
  const Bp = new Int32Array(M + 2);
  Bp.set(B);
  Bp[M] = sB;
  Bp[M + 1] = sB + 1;
  const trace = [];
  // d = 0: the initial path, oldPos -1 on diagonal 0, snaked as far as it goes.
  let cur = new Int32Array(5);
  {
    let xx = -1;
    let yy = -1;
    while (Ap[xx + 1] === Bp[yy + 1]) { xx++; yy++; }
    cur[2] = xx + 2;
    trace.push(cur);
    if (xx + 1 >= N && yy + 1 >= M) return { D: 0, trace };
  }
  const kEnd = N - M;
  for (let d = 1; d <= N + M; d++) {
    const prev = trace[d - 1];
    cur = new Int32Array(2 * d + 5);
    // k from -d to d in steps of 2. sRm is the previous depth's value on
    // diagonal k - 1 (carried: it is the previous iteration's sAd);
    // sAd is the previous depth's value on diagonal k + 1.
    let sRm = prev[0];
    let iAd = 2;
    let j = 2;
    for (let k = -d; k <= d; k += 2, iAd += 2, j += 2) {
      const sAd = prev[iAd];
      // The candidate further into the old text, deletion on a tie; -2
      // marks a candidate the guards rule out (its line does not exist).
      const cRm = (sRm !== 0 && sRm <= N) ? sRm - 1 : -2;
      const cAd = (sAd !== 0 && sAd - 2 - k < M) ? sAd - 2 : -2;
      sRm = sAd;
      if (cRm === -2 && cAd === -2) continue;
      let xx = cRm >= cAd ? cRm : cAd;
      let yy = xx - k;
      // Follow every run of lines equal in both as far as it goes.
      while (Ap[xx + 1] === Bp[yy + 1]) { xx++; yy++; }
      cur[j] = xx + 2;
    }
    trace.push(cur);
    // Only diagonal N - M can hold both ends: it reached them when its
    // furthest old index is N - 1, stored as N + 1.
    if (cur[kEnd + d + 2] === N + 1) return { D: d, trace };
  }
  throw new Error('makefun-diff: the search ended without reaching both ends');
}

/**
 * The edit script of the middle, as an array of op codes in order
 * (KEEP / DELETE / INSERT, one per line), recovered by walking the trace
 * back from (N - 1, M - 1) at depth D. Each step recomputes the forward
 * pass's predecessor choice from trace[d - 1] with the same guards and
 * tie-break, so the script is the forward search's own path.
 */
function backtrackOps(A, B, D, trace) {
  const N = A.length;
  const M = B.length;
  const ops = [];
  let x = N - 1;
  let y = M - 1;
  for (let d = D; d >= 1; d--) {
    const k = x - y;
    const prev = trace[d - 1];
    const sRm = prev[k + d];
    const sAd = prev[k + d + 2];
    const canRm = sRm !== 0 && sRm <= N;
    const canAd = sAd !== 0 && sAd - 2 - k < M;
    if (!canRm && !canAd) throw new Error('makefun-diff: backtrack lost the path');
    const byDel = canRm ? sRm - 1 : -2;
    const byIns = canAd ? sAd - 2 : -2;
    let lx;
    let ly;
    let px;
    let py;
    if (byDel >= byIns) {
      // Deletion: from (sRm - 2, sRm - 1 - k) it consumes old line sRm - 1.
      lx = sRm - 1; ly = lx - k; px = sRm - 2; py = ly;
    } else {
      // Insertion: from (sAd - 2, sAd - 2 - k - 1) it consumes new line sAd - 2 - k.
      lx = sAd - 2; ly = lx - k; px = lx; py = ly - 1;
    }
    while (x > lx) { ops.push(KEEP); x--; y--; }
    ops.push(byDel >= byIns ? DELETE : INSERT);
    x = px;
    y = py;
  }
  // The d = 0 snake, from its end back to (-1, -1).
  while (x >= 0) { ops.push(KEEP); x--; y--; }
  ops.reverse();
  return ops;
}

/** A line as the patch writes it: its op, the line, a newline, and "%" on a line of its own when it had none. */
function written(op, line) {
  return line.endsWith('\n') ? op + line : op + line + '\n%\n';
}

/** make.fun's own patch of input.old into input.new. */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  // Intern the lines of both texts in one shared map.
  const strs = [];
  const map = new Map();
  const intern = (s) => {
    let id = map.get(s);
    if (id === undefined) { id = strs.length; map.set(s, id); strs.push(s); }
    return id;
  };
  const a = tokenizeIds(input.old, intern);
  const b = tokenizeIds(input.new, intern);
  const n = a.length;
  const m = b.length;
  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix && a[n - 1 - suffix] === b[m - 1 - suffix]) suffix++;
  const A = a.subarray(prefix, n - suffix);
  const B = b.subarray(prefix, m - suffix);
  // Sentinel ids no real line has: one pair for each text's end.
  const { D, trace } = myersForward(A, B, strs.length, strs.length + 2);
  const ops = backtrackOps(A, B, D, trace);

  // The runs of the edit script, in order.
  const runs = [];
  if (prefix > 0) runs.push({ kind: KEEP, count: prefix });
  for (const op of ops) {
    const r = runs.at(-1);
    if (r !== undefined && r.kind === op) r.count++;
    else runs.push({ kind: op, count: 1 });
  }
  if (suffix > 0) runs.push({ kind: KEEP, count: suffix });

  // The change runs: each a stretch of deletions and insertions between two kept lines, at old line o and new line p.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (const r of runs) {
    if (r.kind === KEEP) {
      o += r.count;
      p += r.count;
      continue;
    }
    let c = changes.at(-1);
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (r.kind === DELETE) {
      c.dels += r.count;
      o += r.count;
      deleted += r.count;
    } else {
      c.ins += r.count;
      p += r.count;
      inserted += r.count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (const c of changes) {
    const h = hunks.at(-1);
    const prev = h?.at(-1);
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }
  const out = [`${FORMAT} old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (const h of hunks) {
    const first = h[0];
    const last = h.at(-1);
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, n - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const c of h) {
      for (; at < c.o; at++) out.push(written('=', strs[a[at]]));
      for (let i = 0; i < c.dels; i++) out.push(written('-', strs[a[c.o + i]]));
      for (let i = 0; i < c.ins; i++) out.push(written('+', strs[b[c.p + i]]));
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) out.push(written('=', strs[a[at]]));
  }
  return out.join('');
}
