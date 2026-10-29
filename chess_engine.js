/* Minimal, dependency-free chess engine for the "Play against AI" page.
 * - Full legal move generation (castling, en passant, promotion)
 * - SAN notation, check / mate / stalemate / 50-move / repetition / insufficient material
 * - Negamax + alpha-beta + quiescence AI with piece-square tables
 * Board index: sq = row * 8 + col, row 0 = rank 8 (top), col 0 = file a.
 */
(function (root) {
  "use strict";
  const FILES = "abcdefgh";
  const VAL = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
  const isWhite = (p) => p && p === p.toUpperCase();
  const colorOf = (p) => (p ? (isWhite(p) ? "w" : "b") : null);
  const sqName = (sq) => FILES[sq % 8] + (8 - Math.floor(sq / 8));
  const nameSq = (n) => (8 - parseInt(n[1], 10)) * 8 + FILES.indexOf(n[0]);

  const KN = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
  const KG = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
  const DIAG = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
  const ORTH = [[-1, 0], [1, 0], [0, -1], [0, 1]];

  // Piece-square tables from White's point of view (row 0 = rank 8)
  const PST = {
    p: [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5,
      0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    n: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30,
      -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    b: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10,
      -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    r: [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
      -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0],
    q: [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5,
      0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
    k: [-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
      -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10,
      20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20],
  };

  class Game {
    constructor(fen) { this.load(fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

    load(fen) {
      const [pos, turn, castle, ep, half, full] = fen.trim().split(/\s+/);
      this.b = new Array(64).fill(null);
      let sq = 0;
      for (const ch of pos) {
        if (ch === "/") continue;
        if (/\d/.test(ch)) sq += parseInt(ch, 10);
        else this.b[sq++] = ch;
      }
      this.turn = turn || "w";
      this.castle = { K: false, Q: false, k: false, q: false };
      if (castle && castle !== "-") for (const c of castle) this.castle[c] = true;
      this.ep = ep && ep !== "-" ? nameSq(ep) : -1;
      this.half = parseInt(half || "0", 10);
      this.full = parseInt(full || "1", 10);
      this.stack = [];
      this.sanHistory = [];
      this.reps = {};
      this.reps[this.key()] = 1;
    }

    key() {
      return this.b.map((p) => p || ".").join("") + this.turn +
        (this.castle.K ? "K" : "") + (this.castle.Q ? "Q" : "") + (this.castle.k ? "k" : "") + (this.castle.q ? "q" : "") + this.ep;
    }

    fen() {
      let s = "";
      for (let r = 0; r < 8; r++) {
        let e = 0;
        for (let c = 0; c < 8; c++) {
          const p = this.b[r * 8 + c];
          if (!p) e++; else { if (e) { s += e; e = 0; } s += p; }
        }
        if (e) s += e;
        if (r < 7) s += "/";
      }
      const cs = ["K", "Q", "k", "q"].filter((c) => this.castle[c]).join("") || "-";
      return `${s} ${this.turn} ${cs} ${this.ep >= 0 ? sqName(this.ep) : "-"} ${this.half} ${this.full}`;
    }

    kingSq(color) {
      const k = color === "w" ? "K" : "k";
      return this.b.indexOf(k);
    }

    attacked(sq, by) {
      const r = Math.floor(sq / 8), c = sq % 8, b = this.b;
      const at = (rr, cc) => (rr >= 0 && rr < 8 && cc >= 0 && cc < 8 ? b[rr * 8 + cc] : undefined);
      const own = (p, t) => p && colorOf(p) === by && p.toLowerCase() === t;
      // pawns
      const pr = by === "w" ? r + 1 : r - 1;
      if (own(at(pr, c - 1), "p") || own(at(pr, c + 1), "p")) return true;
      for (const [dr, dc] of KN) if (own(at(r + dr, c + dc), "n")) return true;
      for (const [dr, dc] of KG) if (own(at(r + dr, c + dc), "k")) return true;
      for (const [dr, dc] of DIAG) {
        let rr = r + dr, cc = c + dc;
        while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
          const p = b[rr * 8 + cc];
          if (p) { if (colorOf(p) === by && (p.toLowerCase() === "b" || p.toLowerCase() === "q")) return true; break; }
          rr += dr; cc += dc;
        }
      }
      for (const [dr, dc] of ORTH) {
        let rr = r + dr, cc = c + dc;
        while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
          const p = b[rr * 8 + cc];
          if (p) { if (colorOf(p) === by && (p.toLowerCase() === "r" || p.toLowerCase() === "q")) return true; break; }
          rr += dr; cc += dc;
        }
      }
      return false;
    }

    inCheck(color) { color = color || this.turn; return this.attacked(this.kingSq(color), color === "w" ? "b" : "w"); }

    pseudo(capturesOnly) {
      const out = [], b = this.b, me = this.turn, opp = me === "w" ? "b" : "w";
      const add = (from, to, flag) => {
        const piece = b[from], cap = flag === "ep" ? b[to + (me === "w" ? 8 : -8)] : b[to];
        if (capturesOnly && !cap && !(piece.toLowerCase() === "p" && (to < 8 || to >= 56))) return;
        if (piece.toLowerCase() === "p" && (to < 8 || to >= 56)) {
          for (const pr of ["q", "r", "b", "n"]) out.push({ from, to, piece, captured: cap, promo: me === "w" ? pr.toUpperCase() : pr, flag });
        } else out.push({ from, to, piece, captured: cap, flag });
      };
      for (let sq = 0; sq < 64; sq++) {
        const p = b[sq];
        if (!p || colorOf(p) !== me) continue;
        const t = p.toLowerCase(), r = Math.floor(sq / 8), c = sq % 8;
        if (t === "p") {
          const dir = me === "w" ? -1 : 1, start = me === "w" ? 6 : 1;
          const one = sq + dir * 8;
          if (r + dir >= 0 && r + dir < 8 && !b[one]) {
            add(sq, one);
            const two = sq + dir * 16;
            if (r === start && !b[two]) add(sq, two, "double");
          }
          for (const dc of [-1, 1]) {
            const cc = c + dc, rr = r + dir;
            if (cc < 0 || cc > 7 || rr < 0 || rr > 7) continue;
            const to = rr * 8 + cc;
            if (b[to] && colorOf(b[to]) === opp) add(sq, to);
            else if (to === this.ep) add(sq, to, "ep");
          }
        } else if (t === "n" || t === "k") {
          for (const [dr, dc] of t === "n" ? KN : KG) {
            const rr = r + dr, cc = c + dc;
            if (rr < 0 || rr > 7 || cc < 0 || cc > 7) continue;
            const to = rr * 8 + cc;
            if (!b[to] || colorOf(b[to]) === opp) add(sq, to);
          }
          if (t === "k" && !capturesOnly) {
            const row = me === "w" ? 7 : 0, base = row * 8;
            if (sq === base + 4 && !this.attacked(sq, opp)) {
              const [kc, qc] = me === "w" ? ["K", "Q"] : ["k", "q"];
              const rook = me === "w" ? "R" : "r";
              if (this.castle[kc] && b[base + 7] === rook && !b[base + 5] && !b[base + 6] &&
                  !this.attacked(base + 5, opp) && !this.attacked(base + 6, opp)) add(sq, base + 6, "castle");
              if (this.castle[qc] && b[base] === rook && !b[base + 1] && !b[base + 2] && !b[base + 3] &&
                  !this.attacked(base + 3, opp) && !this.attacked(base + 2, opp)) add(sq, base + 2, "castle");
            }
          }
        } else {
          const dirs = t === "b" ? DIAG : t === "r" ? ORTH : DIAG.concat(ORTH);
          for (const [dr, dc] of dirs) {
            let rr = r + dr, cc = c + dc;
            while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
              const to = rr * 8 + cc;
              if (!b[to]) add(sq, to);
              else { if (colorOf(b[to]) === opp) add(sq, to); break; }
              rr += dr; cc += dc;
            }
          }
        }
      }
      return out;
    }

    make(m) {
      const b = this.b, me = this.turn;
      this.stack.push({ m, castle: { ...this.castle }, ep: this.ep, half: this.half, full: this.full });
      b[m.to] = m.promo || m.piece;
      b[m.from] = null;
      if (m.flag === "ep") b[m.to + (me === "w" ? 8 : -8)] = null;
      if (m.flag === "castle") {
        if (m.to % 8 === 6) { b[m.to - 1] = b[m.to + 1]; b[m.to + 1] = null; }
        else { b[m.to + 1] = b[m.to - 2]; b[m.to - 2] = null; }
      }
      // castling rights
      const touch = (sq) => {
        if (sq === 60) { this.castle.K = this.castle.Q = false; }
        if (sq === 4) { this.castle.k = this.castle.q = false; }
        if (sq === 63) this.castle.K = false;
        if (sq === 56) this.castle.Q = false;
        if (sq === 7) this.castle.k = false;
        if (sq === 0) this.castle.q = false;
      };
      touch(m.from); touch(m.to);
      this.ep = m.flag === "double" ? (m.from + m.to) / 2 : -1;
      this.half = m.piece.toLowerCase() === "p" || m.captured ? 0 : this.half + 1;
      if (me === "b") this.full++;
      this.turn = me === "w" ? "b" : "w";
    }

    unmake() {
      const s = this.stack.pop(), m = s.m, b = this.b;
      this.turn = this.turn === "w" ? "b" : "w";
      const me = this.turn;
      b[m.from] = m.piece;
      b[m.to] = m.flag === "ep" ? null : m.captured || null;
      if (m.flag === "ep") b[m.to + (me === "w" ? 8 : -8)] = m.captured;
      if (m.flag === "castle") {
        if (m.to % 8 === 6) { b[m.to + 1] = b[m.to - 1]; b[m.to - 1] = null; }
        else { b[m.to - 2] = b[m.to + 1]; b[m.to + 1] = null; }
      }
      this.castle = s.castle; this.ep = s.ep; this.half = s.half; this.full = s.full;
    }

    moves(opts) {
      opts = opts || {};
      const me = this.turn, legal = [];
      for (const m of this.pseudo(opts.capturesOnly)) {
        if (opts.square !== undefined && m.from !== opts.square) continue;
        this.make(m);
        if (!this.attacked(this.kingSq(me), this.turn)) legal.push(m);
        this.unmake();
      }
      return legal;
    }

    insufficient() {
      const minors = [];
      for (const p of this.b) {
        if (!p) continue;
        const t = p.toLowerCase();
        if (t === "k") continue;
        if (t === "n" || t === "b") minors.push(t); else return false;
      }
      return minors.length <= 1;
    }

    status() {
      const legal = this.moves();
      if (!legal.length) return this.inCheck() ? "checkmate" : "stalemate";
      if (this.half >= 100) return "fifty";
      if (this.insufficient()) return "insufficient";
      if ((this.reps[this.key()] || 0) >= 3) return "repetition";
      return null;
    }

    san(m, legal) {
      legal = legal || this.moves();
      let s;
      if (m.flag === "castle") s = m.to % 8 === 6 ? "O-O" : "O-O-O";
      else {
        const t = m.piece.toLowerCase();
        const cap = !!m.captured;
        if (t === "p") s = (cap ? FILES[m.from % 8] + "x" : "") + sqName(m.to) + (m.promo ? "=" + m.promo.toUpperCase() : "");
        else {
          const others = legal.filter((o) => o.piece === m.piece && o.to === m.to && o.from !== m.from);
          let dis = "";
          if (others.length) {
            const sameFile = others.some((o) => o.from % 8 === m.from % 8);
            const sameRank = others.some((o) => Math.floor(o.from / 8) === Math.floor(m.from / 8));
            if (!sameFile) dis = FILES[m.from % 8];
            else if (!sameRank) dis = sqName(m.from)[1];
            else dis = sqName(m.from);
          }
          s = t.toUpperCase() + dis + (cap ? "x" : "") + sqName(m.to);
        }
      }
      this.make(m);
      const replies = this.moves();
      if (this.inCheck()) s += replies.length ? "+" : "#";
      this.unmake();
      return s;
    }

    play(m) {
      const s = this.san(m);
      this.make(m);
      this.sanHistory.push(s);
      const k = this.key();
      this.reps[k] = (this.reps[k] || 0) + 1;
      return s;
    }

    takeback() {
      if (!this.stack.length) return false;
      const k = this.key();
      this.reps[k] = Math.max(0, (this.reps[k] || 1) - 1);
      this.unmake();
      this.sanHistory.pop();
      return true;
    }

    // ---------------- AI ----------------
    evaluate() {
      // from side-to-move perspective
      let score = 0;
      for (let sq = 0; sq < 64; sq++) {
        const p = this.b[sq];
        if (!p) continue;
        const t = p.toLowerCase();
        const pst = isWhite(p) ? PST[t][sq] : PST[t][(7 - Math.floor(sq / 8)) * 8 + (sq % 8)];
        const v = VAL[t] + pst;
        score += isWhite(p) ? v : -v;
      }
      return this.turn === "w" ? score : -score;
    }

    order(ms) {
      return ms.sort((a, b) => {
        const sa = (a.captured ? 10 * VAL[a.captured.toLowerCase()] - VAL[a.piece.toLowerCase()] : 0) + (a.promo ? 800 : 0);
        const sb = (b.captured ? 10 * VAL[b.captured.toLowerCase()] - VAL[b.piece.toLowerCase()] : 0) + (b.promo ? 800 : 0);
        return sb - sa;
      });
    }

    quiesce(alpha, beta, depth) {
      this.nodes++;
      const stand = this.evaluate();
      if (stand >= beta) return beta;
      if (alpha < stand) alpha = stand;
      if (depth <= 0) return alpha;
      for (const m of this.order(this.moves({ capturesOnly: true }))) {
        this.make(m);
        const sc = -this.quiesce(-beta, -alpha, depth - 1);
        this.unmake();
        if (sc >= beta) return beta;
        if (sc > alpha) alpha = sc;
      }
      return alpha;
    }

    negamax(depth, alpha, beta, ply) {
      this.nodes++;
      const ms = this.moves();
      if (!ms.length) return this.inCheck() ? -100000 + ply : 0;
      if (this.half >= 100) return 0;
      if (depth === 0) return this.quiesce(alpha, beta, 4);
      for (const m of this.order(ms)) {
        this.make(m);
        const sc = -this.negamax(depth - 1, -beta, -alpha, ply + 1);
        this.unmake();
        if (sc >= beta) return beta;
        if (sc > alpha) alpha = sc;
      }
      return alpha;
    }

    bestMove(level) {
      // level: 1 easy, 2 medium, 3 hard
      const depth = level >= 3 ? 3 : level === 2 ? 2 : 1;
      const noise = level === 1 ? 120 : level === 2 ? 25 : 0;
      this.nodes = 0;
      let best = null, bestScore = -Infinity;
      const ms = this.order(this.moves());
      for (const m of ms) {
        this.make(m);
        let sc = -this.negamax(depth - 1, -Infinity, -(bestScore - noise), 1);
        this.unmake();
        sc += noise ? (Math.random() * 2 - 1) * noise : 0;
        if (sc > bestScore) { bestScore = sc; best = m; }
      }
      return best;
    }
  }

  Game.sqName = sqName;
  Game.nameSq = nameSq;
  root.ChessGame = Game;
  if (typeof module !== "undefined" && module.exports) module.exports = Game;
})(typeof window !== "undefined" ? window : globalThis);
