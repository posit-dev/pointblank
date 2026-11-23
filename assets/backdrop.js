// The homepage hero's animated backdrop: a field of table cells, with a scan line sweeping across
// it. Now and then a cell it passes fails its check and is flagged with a red point (like the
// logo's), which glows and fades. It's faint behind the logo and tagline, paused out of view, and
// still with reduced motion.
(() => {
  "use strict";

  const hero = document.querySelector(".gd-hero");
  if (!hero) return;
  const canvas = document.createElement("canvas");
  canvas.id = "pbh-backdrop";
  canvas.setAttribute("aria-hidden", "true");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  hero.prepend(canvas);

  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const root = document.documentElement;
  const isDark = () => root.classList.contains("quarto-dark") || document.body.classList.contains("quarto-dark");

  const rand = (lo, hi) => lo + Math.random() * (hi - lo);
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const smooth = (lo, hi, x) => {
    const k = clamp((x - lo) / (hi - lo), 0, 1);
    return k * k * (3 - 2 * k);
  };

  // The canvas's size (in CSS pixels); the logo and tagline's box in it, which is kept clear; and
  // the page's text over it (the lead, the calls to action and the sidebar), kept nearly so
  let W = 0;
  let H = 0;
  let dpr = 1;
  const focus = { left: 0, right: 0, top: 0, bottom: 0 };
  let texts = [];
  const TEXTS = ["#quarto-document-content > p:first-of-type", ".pbh-cta", "#quarto-margin-sidebar"];

  // How much of a thing to draw at a point: less near the logo and tagline, and over text
  const gap = (r, x, y) => Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
  const clearing = (x, y, over = 0.3) => {
    let k = 0.1 + 0.9 * smooth(0, 120, gap(focus, x, y));
    for (const r of texts) k = Math.min(k, over + (1 - over) * smooth(0, 110, gap(r, x, y)));
    return k;
  };

  function layout() {
    const box = hero.getBoundingClientRect();
    const nav = document.getElementById("quarto-header");
    const navHeight = nav ? nav.offsetHeight : 0;
    const heroTop = box.top + window.scrollY;
    // From under the navbar to below the hero (where it fades out), across the page
    const top = navHeight - heroTop;
    const width = root.clientWidth;
    const height = Math.round(heroTop - navHeight + box.height + Math.min(340, window.innerHeight * 0.4));
    canvas.style.left = `${-box.left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    // (The logo shown, for the theme, and the tagline)
    const parts = Array.from(hero.querySelectorAll(".gd-hero-logo, .gd-hero-tagline"))
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0);
    if (parts.length > 0) {
      focus.left = Math.min(...parts.map((r) => r.left));
      focus.right = Math.max(...parts.map((r) => r.right));
      focus.top = Math.min(...parts.map((r) => r.top)) - box.top - top;
      focus.bottom = Math.max(...parts.map((r) => r.bottom)) - box.top - top;
    } else {
      focus.left = width * 0.3;
      focus.right = width * 0.7;
      focus.top = height * 0.2;
      focus.bottom = height * 0.6;
    }
    texts = TEXTS.map((selector) => document.querySelector(selector))
      .filter((el) => el !== null)
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0)
      .map((r) => ({ left: r.left, right: r.right, top: r.top - box.top - top, bottom: r.bottom - box.top - top }));

    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const resized = Math.abs(width - W) > W * 0.1 || Math.abs(height - H) > H * 0.1;
    if (width !== W || height !== H || ratio !== dpr) {
      W = width;
      H = height;
      dpr = ratio;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    return resized;
  }

  /* ── The cells, the scan line and the flags ────────────────────────────────────────────────── */

  // The cells' size and the space between them; how fast the scan line goes (in pixels a second),
  // and how far past the edges; the share of cells that fail as it passes; how long a flag lasts
  const CW = 46;
  const CH = 22;
  const GAP = 6;
  const SPEED = 90;
  const RUN = 240;
  const FAIL = 0.035;
  const LIFE = 5.2;

  let cols = 0;
  let rows = 0;
  let flags = [];
  let checked = [];

  const scanAt = (t) => ((t * SPEED) % (W + RUN)) - RUN / 2;
  const cellX = (c) => c * (CW + GAP) + CW / 2;
  const cellY = (r) => r * (CH + GAP) + CH / 2;

  // The cells of a column the scan line has just passed: each is checked, and some fail
  function check(c, t) {
    for (let r = 0; r < rows; r++) {
      if (Math.random() < FAIL) flags.push({ x: cellX(c), y: cellY(r), born: t });
    }
    checked[c] = t;
  }

  function init(t) {
    cols = Math.ceil(W / (CW + GAP)) + 1;
    rows = Math.ceil(H / (CH + GAP)) + 1;
    checked = new Array(cols).fill(-Infinity);
    flags = [];
    // (As if the scan line had been running: the columns behind it, with the flags still showing)
    const scan = scanAt(t);
    for (let c = 0; c < cols; c++) {
      const behind = (scan - cellX(c)) / SPEED;
      if (behind > 0 && behind < LIFE) check(c, t - behind);
    }
  }

  let lastScan = null;

  function draw(t) {
    const dark = isDark();
    const ink = dark ? "237,230,227" : "60,0,0";
    const scan = scanAt(t);

    // The columns the scan line has crossed since the last frame (and none when it starts over)
    if (lastScan !== null && scan > lastScan) {
      for (let c = 0; c < cols; c++) {
        const x = cellX(c);
        if (x > lastScan && x <= scan) check(c, t);
      }
    }
    lastScan = scan;

    // The cells: brighter near the scan line
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = c * (CW + GAP);
        const y = r * (CH + GAP);
        const lit = Math.max(0, 1 - Math.abs(cellX(c) - scan) / 140);
        const a = (dark ? 0.045 + lit * 0.1 : 0.04 + lit * 0.06) * clearing(cellX(c), cellY(r), 0.25);
        if (a < 0.004) continue;
        ctx.fillStyle = `rgba(${ink},${a})`;
        ctx.beginPath();
        ctx.roundRect(x, y, CW, CH, 4);
        ctx.fill();
      }
    }

    // The scan line, with a fading trail behind it
    const trail = ctx.createLinearGradient(scan - 90, 0, scan + 6, 0);
    trail.addColorStop(0, "rgba(226,0,0,0)");
    trail.addColorStop(1, dark ? "rgba(255,70,60,0.11)" : "rgba(226,0,0,0.06)");
    ctx.fillStyle = trail;
    ctx.fillRect(scan - 90, 0, 96, H);

    // The flags: each comes up quickly, pulses, and fades
    flags = flags.filter((flag) => t - flag.born < LIFE);
    for (const flag of flags) {
      const age = t - flag.born;
      const life = age / LIFE;
      const a = (life < 0.1 ? life / 0.1 : 1 - (life - 0.1) / 0.9) * clearing(flag.x, flag.y, 0.2);
      if (a <= 0.01) continue;
      const halo = 7 + 6 * Math.sin(age * 3.3) ** 2;
      ctx.fillStyle = `rgba(226,0,0,${(dark ? 0.26 : 0.18) * a})`;
      ctx.beginPath();
      ctx.arc(flag.x, flag.y, halo, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(226,0,0,${0.85 * a})`;
      ctx.beginPath();
      ctx.arc(flag.x, flag.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /* ── Running it ────────────────────────────────────────────────────────────────────────────── */

  let t = 0;
  let last = 0;
  let frame = 0;
  let inView = true;
  let ready = false;

  function paint() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    draw(t);
  }

  const running = () => inView && !motion.matches && !document.hidden;

  function tick(now) {
    frame = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    t += dt;
    paint();
    if (running()) frame = requestAnimationFrame(tick);
  }

  function start() {
    if (!ready) return;
    if (running()) {
      if (frame === 0) {
        last = 0;
        frame = requestAnimationFrame(tick);
      }
    } else {
      // (Still: one frame, as it is)
      if (frame !== 0) cancelAnimationFrame(frame);
      frame = 0;
      paint();
    }
  }

  let pending = 0;
  function refit() {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => {
      if (layout() || !ready) {
        init(t);
        lastScan = null;
      }
      ready = true;
      canvas.classList.add("pbh-on");
      start();
    });
  }

  // (Started a little way into its life, with the scan line partway across)
  t = 20;
  refit();
  if ("ResizeObserver" in window) new ResizeObserver(refit).observe(hero);
  window.addEventListener("resize", refit);
  motion.addEventListener("change", start);
  document.addEventListener("visibilitychange", start);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      inView = entries[entries.length - 1].isIntersecting;
      start();
    }).observe(canvas);
  }
  // (A theme's toggle: redrawn at once, if it's still)
  new MutationObserver(() => {
    if (!running()) paint();
  }).observe(root, { attributes: true, attributeFilter: ["class", "data-bs-theme"] });
})();
