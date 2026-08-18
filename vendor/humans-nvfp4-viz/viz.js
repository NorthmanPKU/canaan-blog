/* Interactive figures for the NVFP4 RL blog.
 *
 * Each <figure class="nvfp4-viz" data-viz="..."> keeps its original <img> as a
 * no-JS fallback; when this script boots it hides the image and builds an
 * interactive widget in its place. Chart data lives in viz-data.js (traced
 * from the original screenshots by tools/trace_curves.py).
 */
(function () {
  "use strict";

  const NV = (window.NVFP4_VIZ = window.NVFP4_VIZ || {});
  NV.data = NV.data || {};

  /* ---------------- tiny DOM/SVG helpers ---------------- */

  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c) node.appendChild(c);
    return node;
  }

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c) node.appendChild(c);
    return node;
  }

  function fmt(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const av = Math.abs(v);
    let out;
    if (av >= 1000) out = v.toFixed(0);
    else if (av >= 100) out = v.toFixed(digits === undefined ? 1 : digits);
    else if (av >= 1) out = v.toFixed(digits === undefined ? 2 : digits);
    else if (av === 0) return "0";
    else if (av < 0.001) return v.toExponential(2);
    else out = v.toFixed(digits === undefined ? 4 : digits);
    return out.includes(".") ? out.replace(/0+$/, "").replace(/\.$/, "") : out;
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (span <= 0) return [lo];
    const step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    let step = mag;
    for (const m of [1, 2, 2.5, 5, 10]) {
      if (step0 <= m * mag) {
        step = m * mag;
        break;
      }
    }
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step) {
      ticks.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    }
    return ticks;
  }

  function extent(arrs, dim) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const arr of arrs) {
      for (const p of arr) {
        if (p[dim] < lo) lo = p[dim];
        if (p[dim] > hi) hi = p[dim];
      }
    }
    return [lo, hi];
  }

  const visibilityWatchers = [];

  function whenVisible(node, cb) {
    if (!("IntersectionObserver" in window)) {
      cb(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => cb(e.isIntersecting)),
      { rootMargin: "60px" }
    );
    io.observe(node);
    visibilityWatchers.push({ node, io, cb });
  }

  /* rebuilds replace a figure's DOM wholesale: stop the removed nodes'
     observers and tell their closures they are no longer visible so any
     animation loop keyed on visibility halts */
  function releaseWatchers(fig) {
    for (let i = visibilityWatchers.length - 1; i >= 0; i--) {
      const w = visibilityWatchers[i];
      if (fig.contains(w.node) || !w.node.isConnected) {
        w.io.disconnect();
        w.cb(false);
        visibilityWatchers.splice(i, 1);
      }
    }
  }

  /* ---------------- color utilities ---------------- */

  function clamp01(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  function hexToRgb(c) {
    const m = c.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (!m) return [90, 90, 90];
    let h = m[1];
    if (h.length === 3) h = h.replace(/./g, (ch) => ch + ch);
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  function rgbToHsl(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) return [0, 0, l];
    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h;
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
    return [h * 360, s, l];
  }

  function hslToRgb(h, s, l) {
    h = (((h % 360) + 360) % 360) / 360;
    if (s === 0) {
      const v = Math.round(l * 255);
      return [v, v, v];
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const f = (t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return [Math.round(f(h + 1 / 3) * 255), Math.round(f(h) * 255), Math.round(f(h - 1 / 3) * 255)];
  }

  const BASELINE_COLOR = "#3d3a35";
  const INK = [22, 21, 19];

  /* Calm the sampled matplotlib colors against the warm paper: clamp saturation and
   * lightness per-hue, keep grays gray, and blend in a touch of ink. */
  function harmonize(color) {
    const [r0, g0, b0] = hexToRgb(color);
    let [h, s, l] = rgbToHsl(r0, g0, b0);
    if (s < 0.2) {
      l = clamp01(l, 0.32, 0.74);
    } else {
      s = clamp01(s, 0.45, 0.66);
      l = clamp01(l, 0.34, 0.54);
      if (h >= 40 && h <= 100) {
        l = Math.min(l, 0.42);
        s = Math.min(s, 0.58);
      }
    }
    const [r, g, b] = hslToRgb(h, s, l);
    const mix = (a, ink) => Math.round(a * 0.94 + ink * 0.06);
    const hex = (v) => v.toString(16).padStart(2, "0");
    return "#" + hex(mix(r, INK[0])) + hex(mix(g, INK[1])) + hex(mix(b, INK[2]));
  }

  const FMT_TOKENS = new Set(["mxfp8", "nvfp4", "mxfp4", "fp8", "fp4"]);

  function isBaselineName(name) {
    const tokens = name.toLowerCase().split(/[-_\s]+/);
    return tokens.includes("bf16") && !tokens.some((t) => FMT_TOKENS.has(t));
  }

  /* One canonical tone per recurring run family so a series keeps its color
   * across every chart in the report (blues: MXFP8; olive/teal: NVFP4;
   * magenta/violet: + 4/6). Chart-local families keep their traced palette. */
  const FAMILY_TONES = [
    [/nvfp4.*4over6.*dequant/, "#7a5bc6"],
    [/nvfp4.*4over6.*high.precision/, "#e06fcf"],
    [/nvfp4.*dequant/, "#219789"],
    [/nvfp4.*high.precision/, "#8faf3f"],
    [/mxfp8.*train.*infer|^mxfp8 bwd$/, "#2c50b8"],
    [/mxfp8.*dequant|^dequantized bwd$/, "#5b82e0"],
    [/mxfp8.*high.precision|^high.precision bwd$/, "#4fb3c9"],
  ];

  function familyTone(name) {
    const n = String(name).toLowerCase().replace(/‑/g, "-");
    for (const [pattern, hex] of FAMILY_TONES) {
      if (pattern.test(n)) return harmonize(hex);
    }
    return null;
  }

  /* Strip token runs shared by every series (prefix and suffix) so labels carry only
   * what distinguishes the runs; the shared part is reported once as a subtitle. */
  function seriesLabels(series) {
    const names = series.map((s) => s.name);
    const arrays = names.map((n) => n.split(/[-_]+/).filter(Boolean));
    const distinct = new Set(names);
    let pre = 0;
    let suf = 0;
    if (distinct.size > 1) {
      const minLen = Math.min(...arrays.map((a) => a.length));
      while (pre < minLen - 1 && arrays.every((a) => a[pre] === arrays[0][pre])) pre++;
      while (
        suf < minLen - 1 - pre &&
        arrays.every((a) => a[a.length - 1 - suf] === arrays[0][arrays[0].length - 1 - suf])
      ) {
        suf++;
      }
    }
    const shared = arrays[0] ? arrays[0].slice(0, pre).concat(suf ? arrays[0].slice(-suf) : []) : [];
    const base = arrays.map((a) => a.slice(pre, a.length - suf).join(" "));
    const runCount = {};
    const labels = names.map((n, i) => {
      runCount[n] = (runCount[n] || 0) + 1;
      return runCount[n] > 1 ? base[i] + " · run " + runCount[n] : base[i];
    });
    return { labels, base, shared };
  }

  function midEllipsis(text, max) {
    if (text.length <= max) return text;
    return text.slice(0, Math.ceil(max * 0.6)) + "…" + text.slice(-Math.floor(max * 0.35));
  }

  /* Uniform per-chart value formatting keyed on the metric. */
  function metricFormatter(data) {
    const key = ((data.yLabel || "") + " " + (data.title || "")).toLowerCase();
    if (/reward/.test(key)) return (v) => v.toFixed(3);
    if (/kl|logprob/.test(key)) {
      return (v) => (v !== 0 && Math.abs(v) < 1e-4 ? v.toExponential(1) : v.toFixed(4));
    }
    if (/grad/.test(key)) return (v) => v.toFixed(3);
    if (/time/.test(key)) return (v) => (Math.abs(v) >= 100 ? Math.round(v) + " s" : v.toFixed(1) + " s");
    if (/token|throughput/.test(key)) return (v) => Math.round(v).toLocaleString("en-US");
    return (v) => fmt(v);
  }

  function fmtDelta(d, fmtVal) {
    return (d < 0 ? "−" : "+") + fmtVal(Math.abs(d));
  }

  function stepWord(xLabel) {
    const words = (xLabel || "step").trim().split(/\s+/);
    return words[words.length - 1].toLowerCase();
  }

  /* ---------------- figure scaffolding ---------------- */

  function card(fig, title, subtitle, hint) {
    const kicker = fig.dataset.figN ? el("span", { class: "viz-kicker", text: "Fig. " + fig.dataset.figN }) : null;
    const head = el(
      "div",
      { class: "viz-head" },
      kicker,
      el("span", { class: "viz-title", text: title || "" }),
      subtitle ? el("span", { class: "viz-sub", text: subtitle }) : null,
      hint ? el("span", { class: "viz-hint", text: hint }) : null
    );
    const body = el("div", { class: "viz-card" }, head);
    fig.appendChild(body);
    return body;
  }

  /* Place a tooltip inside a (possibly horizontally scrolled) plot wrapper.
     cx/cy are px in the wrapper's CONTENT coordinates (svg render scale);
     the result is clamped into the visible window so panned figures on
     phones keep their tooltips near the pointer. */
  function placeTip(plot, tooltip, cx, cy, gap = 12) {
    const view = plot.clientWidth;
    const sl = plot.scrollLeft;
    const ttW = tooltip.offsetWidth;
    const ttH = tooltip.offsetHeight;
    const flip = cx - sl + ttW + gap + 4 > view && cx - sl - ttW - gap >= 4;
    const left = flip ? cx - ttW - gap : cx + gap;
    tooltip.style.right = "";
    tooltip.style.left = Math.round(Math.max(sl + 4, Math.min(left, sl + view - ttW - 4))) + "px";
    tooltip.style.top = Math.round(Math.max(2, Math.min(cy, plot.clientHeight - ttH - 4))) + "px";
  }

  function noteRow(body, caption, how) {
    if (!caption && !how) return;
    body.appendChild(
      el(
        "div",
        { class: "viz-note" },
        caption ? el("span", { class: "cap", text: caption }) : null,
        how ? el("span", { class: "how", text: how }) : null
      )
    );
  }

  /* ---------------- line chart ---------------- */

  const DASH = { dash: "9 6", dot: "2.5 5", dashdot: "10 4 2.5 4" };

  function lineChart(fig, data, opts) {
    opts = opts || {};
    const compact = !!opts.compact;
    /* full-width charts in a phone column re-lay out in a narrower viewBox */
    const narrow = !compact && (fig.clientWidth || 700) < 520;

    const hasTracedSmooth = data.series.some((s) => s.smooth && s.smooth.length);
    const rawOnly = !hasTracedSmooth;

    const { labels, base: baseLabels, shared } = seriesLabels(data.series);
    const distinctCount = new Set(data.series.map((s) => s.name)).size;
    const directLabel = distinctCount <= 5;

    const W = compact ? 560 : narrow ? 470 : 900;
    const H = compact ? 400 : narrow ? 400 : 430;
    const gutter = directLabel ? (compact ? 128 : narrow ? 122 : 168) : compact ? 16 : 18;
    const M = { l: compact ? 48 : narrow ? 46 : 56, t: 30, r: gutter, b: 42 };
    const FS = compact || narrow ? 12 : 11.5;

    const baselineIdx = data.series.findIndex((s) => isBaselineName(s.name));

    /* normalized title/metric label: fold the y-label into the title when redundant */
    const norm = (t) => (t || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    let titleText = data.title || "";
    let metricLabel = data.yLabel || "";
    if (metricLabel && norm(metricLabel).startsWith(norm(titleText))) {
      titleText = metricLabel;
      metricLabel = "";
    }

    const subtitle = shared.length ? "all runs: " + shared.join(" · ") : "";
    const body = card(fig, titleText, subtitle);

    const legend = directLabel ? null : el("div", { class: "viz-legend" });
    const plot = el("div", { class: "viz-plot" });
    const svgRoot = svg("svg", {
      class: "viz-svg",
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      tabindex: "0",
      "aria-label": data.title,
    });
    const tooltip = el("div", { class: "viz-tooltip" });
    const live = el("div", { class: "viz-sr", "aria-live": "polite" });
    const resetBtn = el("button", {
      class: "viz-reset",
      text: "↺ reset zoom",
      onclick: () => {
        state.xDom = fullX.slice();
      state.kbIdx = undefined;
        state.zoomed = false;
        render();
      },
    });
    plot.appendChild(svgRoot);
    plot.appendChild(tooltip);
    plot.appendChild(resetBtn);
    plot.appendChild(live);
    if (legend) body.appendChild(legend);
    body.appendChild(plot);
    noteRow(body, data.caption || "", "hover over points for values · drag to zoom");

    const fullX = data.xDomain || extent(data.series.map((s) => s.smooth || s.raw || []), 0);
    const state = {
      xDom: fullX.slice(),
      hidden: new Set(),
      focus: null, // legend-hover isolate
      hot: null, // pointer-nearest series
      zoomed: false,
    };
    const fmtVal = metricFormatter(data);

    function mainLine(s) {
      return s.smooth && s.smooth.length ? s.smooth : s.raw || [];
    }

    function visibleSeries() {
      return data.series.map((s, i) => ({ s, i })).filter(({ i }) => !state.hidden.has(i));
    }

    function toggleSeries(i, chip) {
      if (state.hidden.has(i)) state.hidden.delete(i);
      else state.hidden.add(i);
      state.kbIdx = undefined;
      if (state.hidden.has(i) && state.focus === i) state.focus = null;
      if (chip) chip.classList.toggle("off", state.hidden.has(i));
      render();
    }

    /* precompute display colors */
    data.series.forEach((s, i) => {
      s.tone = i === baselineIdx ? BASELINE_COLOR : familyTone(s.name) || harmonize(s.color);
    });

    /* legend (dense charts only) */
    let focusTimer = null;
    function setFocus(i) {
      state.focus = i;
      applyEmphasis();
    }
    if (legend) {
      data.series.forEach((s, i) => {
        const chip = el(
          "button",
          {
            class: "viz-chip",
            title: s.name,
            onclick: () => toggleSeries(i, chip),
            onmouseenter: () => {
              clearTimeout(focusTimer);
              focusTimer = setTimeout(() => setFocus(i), 120);
            },
            onmouseleave: () => {
              clearTimeout(focusTimer);
              setFocus(null);
            },
          },
          el("span", {
            class: "swatch",
            style:
              `border-top-color:${s.tone};` +
              (s.dash ? "border-top-style:" + (s.dash === "dot" ? "dotted" : "dashed") : ""),
          }),
          el("span", { text: labels[i] })
        );
        legend.appendChild(chip);
      });
    }

    let scaleX = null;
    let scaleY = null;
    let hoverLayer = null;
    let seriesGs = [];
    let dots = [];
    let crosshair = null;
    let labelNodes = [];

    function yDomain(x0, x1) {
      if (!state.zoomed && data.yDomain) return data.yDomain.slice();
      const arrs = [];
      for (const { s } of visibleSeries()) {
        arrs.push(mainLine(s).filter((p) => p[0] >= x0 && p[0] <= x1));
        if (s.raw && s.raw.length && !rawOnly) {
          arrs.push(s.raw.filter((p) => p[0] >= x0 && p[0] <= x1));
        }
        if (s.band && s.band.length) {
          const inX = s.band.filter((p) => p[0] >= x0 && p[0] <= x1);
          arrs.push(inX.map((p) => [p[0], p[1]]));
          arrs.push(inX.map((p) => [p[0], p[2]]));
        }
      }
      let [lo, hi] = extent(arrs, 1);
      if (!isFinite(lo)) {
        lo = 0;
        hi = 1;
      }
      const pad = (hi - lo) * 0.07 || 1;
      const dom = [lo - pad, hi + pad];
      if (data.yDomain && data.yDomain[0] === 0 && lo >= 0 && lo - pad < 0) dom[0] = 0;
      return dom;
    }

    function pathD(points, x0, x1) {
      let d = "";
      let pen = false;
      for (const p of points) {
        if (p[0] < x0 || p[0] > x1) {
          pen = false;
          continue;
        }
        d += (pen ? "L" : "M") + scaleX(p[0]).toFixed(1) + " " + scaleY(p[1]).toFixed(1);
        pen = true;
      }
      return d;
    }

    function render() {
      clearDragRect();
      if (crosshair) clearHover();
      svgRoot.textContent = "";
      const [x0, x1] = state.xDom;
      const yDom = yDomain(x0, x1);

      scaleX = (v) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
      scaleY = (v) => H - M.b - ((v - yDom[0]) / (yDom[1] - yDom[0])) * (H - M.t - M.b);

      /* grid + axes: horizontal gridlines only, hairline bottom axis */
      const g = svg("g", { class: "viz-axes" });
      for (const t of niceTicks(yDom[0], yDom[1], 5)) {
        const y = scaleY(t);
        g.appendChild(
          svg("line", {
            x1: M.l,
            x2: W - M.r,
            y1: y,
            y2: y,
            stroke: "#eae4d8",
            "stroke-width": 1,
            "shape-rendering": "crispEdges",
          })
        );
        g.appendChild(
          svg("text", {
            x: M.l - 8,
            y: y + 4,
            "text-anchor": "end",
            "font-size": FS,
            fill: "#8a847a",
            style: "font-variant-numeric: tabular-nums",
            text: fmt(t),
          })
        );
      }
      for (const t of niceTicks(x0, x1, compact || narrow ? 5 : 7)) {
        g.appendChild(
          svg("text", {
            x: scaleX(t),
            y: H - M.b + 18,
            "text-anchor": "middle",
            "font-size": FS,
            fill: "#8a847a",
            style: "font-variant-numeric: tabular-nums",
            text: fmt(t, 0),
          })
        );
      }
      g.appendChild(
        svg("line", {
          x1: M.l,
          x2: W - M.r,
          y1: H - M.b,
          y2: H - M.b,
          stroke: "#c4bcae",
          "stroke-width": 1,
          "shape-rendering": "crispEdges",
        })
      );
      if (metricLabel) {
        g.appendChild(
          svg("text", {
            x: M.l,
            y: 16,
            "font-size": FS - 1,
            "font-weight": 600,
            "letter-spacing": "0.08em",
            fill: "#6d6860",
            text: metricLabel.toUpperCase(),
          })
        );
      }
      if (data.xLabel) {
        g.appendChild(
          svg("text", {
            x: (M.l + W - M.r) / 2,
            y: H - 6,
            "text-anchor": "middle",
            "font-size": FS - 1,
            "font-weight": 600,
            "letter-spacing": "0.08em",
            fill: "#8a847a",
            text: data.xLabel.toUpperCase(),
          })
        );
      }
      svgRoot.appendChild(g);

      const clipId = "clip-" + data.id + (compact ? "-c" : "");
      svgRoot.appendChild(
        svg(
          "defs",
          {},
          svg(
            "clipPath",
            { id: clipId },
            svg("rect", { x: M.l, y: M.t - 2, width: W - M.l - M.r, height: H - M.t - M.b + 4 })
          )
        )
      );

      /* series groups: raw underlay + main line, styled via CSS classes */
      const linesG = svg("g", { "clip-path": `url(#${clipId})` });
      seriesGs = [];
      for (const [i, s] of data.series.entries()) {
        const cls =
          "viz-series" +
          (i === baselineIdx ? " wide" : "") +
          (rawOnly && i !== baselineIdx ? " thin" : "");
        const sg = svg("g", { class: cls });
        if (!state.hidden.has(i)) {
          if (s.band && s.band.length) {
            /* replicate spread: [x, lo, hi] triples filled as an area */
            let up = "";
            let down = "";
            for (const p of s.band) {
              if (p[0] < x0 || p[0] > x1) continue;
              up += (up ? "L" : "M") + scaleX(p[0]).toFixed(1) + " " + scaleY(p[2]).toFixed(1);
              down = "L" + scaleX(p[0]).toFixed(1) + " " + scaleY(p[1]).toFixed(1) + down;
            }
            if (up) {
              sg.appendChild(svg("path", { class: "band", d: up + down + "Z", fill: s.tone, "fill-opacity": 0.16, stroke: "none" }));
            }
          }
          if (s.raw && s.raw.length && !rawOnly) {
            const rd = pathD(s.raw, x0, x1);
            if (rd) {
              sg.appendChild(
                svg("path", { class: "raw", d: rd, fill: "none", stroke: s.tone, "stroke-linejoin": "round" })
              );
            }
          }
          const md = pathD(mainLine(s), x0, x1);
          if (md) {
            sg.appendChild(
              svg("path", {
                class: "main",
                d: md,
                fill: "none",
                stroke: s.tone,
                "stroke-linejoin": "round",
                "stroke-linecap": "round",
                ...(s.dash ? { "stroke-dasharray": DASH[s.dash] || s.dash } : {}),
              })
            );
          }
        }
        linesG.appendChild(sg);
        seriesGs.push(sg);
      }
      svgRoot.appendChild(linesG);

      if (directLabel) drawEndpointLabels(x0, x1);

      /* hover layer */
      hoverLayer = svg("g", {});
      crosshair = svg("line", { class: "viz-crosshair", y1: M.t, y2: H - M.b, style: "display:none" });
      hoverLayer.appendChild(crosshair);
      dots = data.series.map((s) =>
        svg("circle", { class: "viz-dot", r: 3, fill: s.tone, style: "display:none" })
      );
      dots.forEach((d) => hoverLayer.appendChild(d));
      svgRoot.appendChild(hoverLayer);
      attachInteraction();
      resetBtn.classList.toggle("on", state.zoomed);
      applyEmphasis();
    }

    /* direct labels at line ends, one per distinct series name, collision-nudged */
    function drawEndpointLabels(x0, x1) {
      const lineFS = compact ? 11.5 : 11;
      const groups = new Map();
      for (const [i, s] of data.series.entries()) {
        const pts = mainLine(s).filter((p) => p[0] >= x0 && p[0] <= x1);
        if (!pts.length) continue;
        const endY = scaleY(pts[pts.length - 1][1]);
        const rank = state.hidden.has(i) ? 2 : s.dash ? 1 : 0;
        const cur = groups.get(s.name);
        if (!cur || rank < cur.rank) groups.set(s.name, { i, endY, rank });
      }
      const maxChars = compact ? 18 : 24;
      const wrapLabel = (label) => {
        const lines = [];
        let cur = "";
        for (const w of label.split(" ")) {
          if (cur && (cur + " " + w).length > maxChars) {
            lines.push(cur);
            cur = w;
          } else {
            cur = cur ? cur + " " + w : w;
          }
        }
        if (cur) lines.push(cur);
        if (lines.length > 3) {
          lines.length = 3;
          lines[2] = midEllipsis(lines[2], maxChars - 1) + "…";
        }
        return lines;
      };
      const slots = [...groups.entries()].map(([name, v]) => {
        const label = baseLabels[data.series.findIndex((s) => s.name === name)];
        const lines = wrapLabel(label);
        return { name, i: v.i, y: v.endY, lines, h: lines.length * (lineFS + 2) + 6 };
      });
      slots.sort((a, b) => a.y - b.y);
      /* forward pass pushes overlaps down, then clamp to plot and push back up */
      for (let k = 1; k < slots.length; k++) {
        const prev = slots[k - 1];
        if (slots[k].y - prev.y < (prev.h + slots[k].h) / 2) {
          slots[k].y = prev.y + (prev.h + slots[k].h) / 2;
        }
      }
      const maxY = H - M.b - 4;
      for (let k = slots.length - 1; k >= 0; k--) {
        const limit = k === slots.length - 1 ? maxY : slots[k + 1].y - (slots[k + 1].h + slots[k].h) / 2;
        if (slots[k].y > limit) slots[k].y = limit;
      }
      labelNodes = [];
      const lg = svg("g", { class: "viz-endlabels" });
      for (const slot of slots) {
        const s = data.series[slot.i];
        const lineEnd = (() => {
          const pts = mainLine(s).filter((p) => p[0] >= x0 && p[0] <= x1);
          return pts.length ? scaleY(pts[pts.length - 1][1]) : slot.y;
        })();
        if (Math.abs(slot.y - lineEnd) > 8) {
          lg.appendChild(
            svg("line", {
              x1: W - M.r + 2,
              y1: lineEnd,
              x2: W - M.r + 9,
              y2: slot.y,
              stroke: "#c4bcae",
              "stroke-width": 1,
            })
          );
        }
        const firstDy = slot.lines.length > 1 ? -(lineFS + 2) / 2 + 4 : 4;
        const t = svg("text", {
          x: W - M.r + 12,
          y: slot.y,
          "font-size": lineFS,
          fill: s.tone,
          "font-weight": 600,
          class: "viz-endlabel",
          "data-i": slot.i,
        });
        slot.lines.forEach((ln, li) => {
          t.appendChild(
            svg("tspan", { x: W - M.r + 12, dy: li === 0 ? firstDy : lineFS + 2, text: ln })
          );
        });
        t.addEventListener("pointerenter", () => setFocus(slot.i));
        t.addEventListener("pointerleave", () => setFocus(null));
        t.addEventListener("click", () => toggleSeries(slot.i, null));
        lg.appendChild(t);
        labelNodes.push(t);
      }
      svgRoot.appendChild(lg);
    }

    /* emphasis state → CSS classes (no re-render) */
    function applyEmphasis() {
      const anyFocus = state.focus !== null;
      svgRoot.classList.toggle("has-hot", state.hot !== null && !anyFocus);
      seriesGs.forEach((sg, i) => {
        sg.classList.toggle("hot", state.hot === i && !anyFocus);
        sg.classList.toggle("focused", anyFocus && state.focus === i);
        sg.classList.toggle("faded", anyFocus && state.focus !== i);
      });
      labelNodes.forEach((t) => {
        const i = Number(t.dataset.i);
        t.style.opacity = anyFocus && state.focus !== i ? 0.3 : state.hidden.has(i) ? 0.3 : 1;
      });
    }

    /* hover + zoom */
    let dragStart = null;
    let rafPending = false;
    let lastEvt = null;
    let liveTimer = null;

    function svgPoint(evt) {
      const rect = svgRoot.getBoundingClientRect();
      return {
        x: ((evt.clientX - rect.left) / rect.width) * W,
        y: ((evt.clientY - rect.top) / rect.height) * H,
      };
    }

    function nearestIndex(points, xv) {
      let lo = 0;
      let hi = points.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (points[mid][0] < xv) lo = mid;
        else hi = mid;
      }
      return Math.abs(points[lo][0] - xv) < Math.abs(points[hi][0] - xv) ? lo : hi;
    }

    function attachInteraction() {
      const capture = svg("rect", {
        x: M.l,
        y: M.t,
        width: W - M.l - M.r,
        height: H - M.t - M.b,
        fill: "transparent",
      });
      hoverLayer.appendChild(capture);

      capture.addEventListener("pointermove", (evt) => {
        lastEvt = evt;
        if (rafPending) return;
        rafPending = true;
        requestAnimationFrame(() => {
          rafPending = false;
          if (!lastEvt) return;
          const pt = svgPoint(lastEvt);
          if (dragStart) {
            drawDragRect(dragStart.x, pt.x);
            return;
          }
          const xv =
            state.xDom[0] + ((pt.x - M.l) / (W - M.l - M.r)) * (state.xDom[1] - state.xDom[0]);
          drawHover(xv, pt);
        });
      });
      capture.addEventListener("pointerleave", () => {
        lastEvt = null;
        clearHover();
      });
      const cancelDrag = () => {
        dragStart = null;
        lastEvt = null;
        clearDragRect();
      };
      capture.addEventListener("pointercancel", cancelDrag);
      capture.addEventListener("lostpointercapture", cancelDrag);
      capture.addEventListener("pointerdown", (evt) => {
        if (evt.pointerType === "touch") {
          const pt = svgPoint(evt);
          const xv =
            state.xDom[0] + ((pt.x - M.l) / (W - M.l - M.r)) * (state.xDom[1] - state.xDom[0]);
          drawHover(xv, pt);
          return;
        }
        dragStart = svgPoint(evt);
        clearHover();
        capture.setPointerCapture(evt.pointerId);
      });
      capture.addEventListener("pointerup", (evt) => {
        const pt = svgPoint(evt);
        if (dragStart && Math.abs(pt.x - dragStart.x) > 8) {
          const toData = (px) =>
            state.xDom[0] + ((px - M.l) / (W - M.l - M.r)) * (state.xDom[1] - state.xDom[0]);
          state.xDom = [toData(Math.min(dragStart.x, pt.x)), toData(Math.max(dragStart.x, pt.x))];
          state.kbIdx = undefined;
          state.zoomed = true;
          dragStart = null;
          clearDragRect();
          render();
          return;
        }
        dragStart = null;
        clearDragRect();
      });
      capture.addEventListener("dblclick", () => {
        state.xDom = fullX.slice();
      state.kbIdx = undefined;
        state.zoomed = false;
        render();
      });
    }

    svgRoot.addEventListener("blur", () => clearHover());
    svgRoot.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape") {
        if (tooltip.classList.contains("on")) clearHover();
        else if (state.zoomed) {
          state.xDom = fullX.slice();
      state.kbIdx = undefined;
          state.zoomed = false;
          render();
        }
        return;
      }
      if (evt.key !== "ArrowLeft" && evt.key !== "ArrowRight") return;
      evt.preventDefault();
      const vis = visibleSeries();
      if (!vis.length) return;
      const pts = mainLine(vis[0].s).filter((p) => p[0] >= state.xDom[0] && p[0] <= state.xDom[1]);
      if (!pts.length) return;
      const cur = state.kbIdx === undefined ? Math.floor(pts.length / 2) : state.kbIdx;
      const step = (evt.shiftKey ? 10 : 1) * (evt.key === "ArrowRight" ? 1 : -1);
      state.kbIdx = Math.max(0, Math.min(pts.length - 1, cur + step));
      const xv = pts[state.kbIdx][0];
      drawHover(xv, { x: scaleX(xv), y: (M.t + H - M.b) / 2 });
    });

    let dragRect = null;
    function drawDragRect(a, b) {
      if (Math.abs(b - a) < 8) return;
      if (!dragRect) {
        dragRect = svg("rect", { class: "viz-zoom-rect" });
        hoverLayer.appendChild(dragRect);
      }
      dragRect.setAttribute("x", Math.min(a, b));
      dragRect.setAttribute("y", M.t);
      dragRect.setAttribute("width", Math.abs(b - a));
      dragRect.setAttribute("height", H - M.t - M.b);
    }
    function clearDragRect() {
      if (dragRect) {
        dragRect.remove();
        dragRect = null;
      }
    }

    function clearHover() {
      state.hot = null;
      clearTimeout(liveTimer);
      live.textContent = "";
      crosshair.style.display = "none";
      dots.forEach((d) => (d.style.display = "none"));
      tooltip.classList.remove("on");
      applyEmphasis();
    }

    function drawHover(xv, pt) {
      const rows = [];
      for (const { s, i } of visibleSeries()) {
        const main = mainLine(s);
        if (!main.length) continue;
        const j = nearestIndex(main, xv);
        const p = main[j];
        if (p[0] < state.xDom[0] - 1e-9 || p[0] > state.xDom[1] + 1e-9) continue;
        rows.push({ s, i, p });
      }
      if (!rows.length) {
        clearHover();
        return;
      }

      /* nearest series by pixel distance, with hysteresis so crossings don't flicker */
      let nearest = null;
      let bestD = Infinity;
      for (const r of rows) {
        const d = Math.abs(scaleY(r.p[1]) - pt.y);
        if (d < bestD) {
          bestD = d;
          nearest = r;
        }
      }
      let hot = bestD <= 26 ? nearest : null;
      if (hot && state.hot !== null && state.hot !== hot.i) {
        const cur = rows.find((r) => r.i === state.hot);
        if (cur && Math.abs(scaleY(cur.p[1]) - pt.y) - bestD < 6) hot = cur;
      }
      state.hot = hot ? hot.i : null;

      /* series x-grids are not aligned; anchor crosshair + header to the nearest row */
      const anchor = hot || nearest;
      const xPix = scaleX(anchor.p[0]);
      crosshair.setAttribute("x1", xPix);
      crosshair.setAttribute("x2", xPix);
      crosshair.style.display = "";
      dots.forEach((d) => (d.style.display = "none"));
      for (const r of rows) {
        const d = dots[r.i];
        d.setAttribute("cx", scaleX(r.p[0]));
        d.setAttribute("cy", scaleY(r.p[1]));
        d.setAttribute("r", state.hot === r.i ? 5 : 3);
        d.classList.toggle("hot", state.hot === r.i);
        d.style.display = "";
      }

      rows.sort((a, b) => b.p[1] - a.p[1]);
      const baseRow = baselineIdx >= 0 ? rows.find((r) => r.i === baselineIdx) : null;

      tooltip.textContent = "";
      tooltip.appendChild(
        el("div", { class: "tt-x", text: stepWord(data.xLabel) + " " + fmt(anchor.p[0], 0) })
      );
      for (const r of rows) {
        const isHot = state.hot === r.i;
        const rowEl = el(
          "div",
          { class: "tt-row" + (isHot ? " hot" : "") },
          el("span", {
            class: "sw",
            style:
              `border-top-color:${r.s.tone};` +
              (r.s.dash ? "border-top-style:" + (r.s.dash === "dot" ? "dotted" : "dashed") : ""),
          }),
          el("span", { class: "name", text: midEllipsis(labels[r.i], 42) }),
          el("span", { class: "val", text: fmtVal(r.p[1]) })
        );
        if (baseRow) {
          rowEl.appendChild(
            el("span", {
              class: "delta",
              text: r.i === baselineIdx ? "baseline" : fmtDelta(r.p[1] - baseRow.p[1], fmtVal),
            })
          );
        }
        if (rawOnly && isHot) {
          const vals = mainLine(r.s)
            .filter((p) => p[0] >= state.xDom[0] && p[0] <= state.xDom[1])
            .map((p) => p[1])
            .sort((a, b) => a - b);
          const median = vals[Math.floor(vals.length / 2)];
          if (median > 0 && r.p[1] > 3 * median) {
            rowEl.appendChild(el("span", { class: "mult", text: (r.p[1] / median).toFixed(1) + "× the series median" }));
          }
        }
        tooltip.appendChild(rowEl);
      }
      tooltip.classList.add("on");

      const svgRect = svgRoot.getBoundingClientRect();
      placeTip(plot, tooltip, (xPix / W) * svgRect.width, (pt.y / H) * svgRect.height - 24, 16);

      applyEmphasis();

      clearTimeout(liveTimer);
      liveTimer = setTimeout(() => {
        if (hot) {
          live.textContent =
            stepWord(data.xLabel) + " " + fmt(hot.p[0], 0) + ": " + labels[hot.i] + " " + fmtVal(hot.p[1]);
        }
      }, 400);
    }

    render();
  }

  /* ---------------- scatter chart ---------------- */

  function scatterPanel(container, panel, opts) {
    opts = opts || {};
    const W = 440;
    const H = 420;
    const M = { l: 52, r: 14, t: panel.title ? 56 : 34, b: 46 };
    const wrap = el("div", { class: "viz-plot" });
    const svgRoot = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}` });
    const tooltip = el("div", { class: "viz-tooltip" });
    wrap.appendChild(svgRoot);
    wrap.appendChild(tooltip);
    container.appendChild(wrap);

    const dom = panel.domain || [0, Math.max(...panel.points.flat()) * 1.05];
    const sx = (v) => M.l + ((v - dom[0]) / (dom[1] - dom[0])) * (W - M.l - M.r);
    const sy = (v) => H - M.b - ((v - dom[0]) / (dom[1] - dom[0])) * (H - M.t - M.b);
    /* first word of each axis label, e.g. "FP4" / "FP8" */
    const xTok = (panel.xLabel || "x").split(/\s/)[0];
    const yTok = (panel.yLabel || "y").split(/\s/)[0];

    if (panel.title) {
      svgRoot.appendChild(
        svg("text", {
          class: "panel-title",
          x: (M.l + W - M.r) / 2,
          y: 18,
          "text-anchor": "middle",
          text: panel.title.toUpperCase(),
        })
      );
    }
    if (panel.annotation) {
      svgRoot.appendChild(
        svg("text", {
          class: "annot",
          x: M.l + 10,
          y: M.t + 20,
          "text-anchor": "start",
          text: panel.annotation,
        })
      );
    }
    for (const t of niceTicks(dom[0], dom[1], 5)) {
      svgRoot.appendChild(
        svg("line", {
          x1: M.l,
          x2: W - M.r,
          y1: sy(t),
          y2: sy(t),
          stroke: "#eae4d8",
          "shape-rendering": "crispEdges",
        })
      );
      svgRoot.appendChild(
        svg("text", { class: "tick", x: M.l - 6, y: sy(t) + 4, "text-anchor": "end", text: fmt(t, 0) })
      );
      svgRoot.appendChild(
        svg("text", {
          class: "tick",
          x: sx(t),
          y: H - M.b + 16,
          "text-anchor": "middle",
          text: fmt(t, 0),
        })
      );
    }
    svgRoot.appendChild(
      svg("line", {
        x1: sx(dom[0]),
        y1: sy(dom[0]),
        x2: sx(dom[1]),
        y2: sy(dom[1]),
        stroke: "#b3ac9f",
        "stroke-dasharray": "6 5",
      })
    );
    svgRoot.appendChild(
      svg("text", {
        class: "ax-label",
        x: (M.l + W - M.r) / 2,
        y: H - 8,
        "text-anchor": "middle",
        text: (panel.xLabel || "").toUpperCase(),
      })
    );
    svgRoot.appendChild(
      svg("text", {
        class: "ax-label",
        x: M.l,
        y: M.t - 8,
        "text-anchor": "start",
        text: (panel.yLabel || "").toUpperCase(),
      })
    );

    const dots = [];
    for (const p of panel.points) {
      const dot = svg("circle", {
        cx: sx(p[0]),
        cy: sy(p[1]),
        r: 4.2,
        fill: "rgba(23,107,100,0.28)",
        stroke: "#176b64",
        "stroke-opacity": 0.9,
      });
      svgRoot.appendChild(dot);
      dots.push({ dot, p });
    }

    svgRoot.addEventListener("pointermove", (evt) => {
      const rect = svgRoot.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * W;
      const my = ((evt.clientY - rect.top) / rect.height) * H;
      let best = null;
      let bestD = 18 * 18;
      for (const d of dots) {
        const dx = sx(d.p[0]) - mx;
        const dy = sy(d.p[1]) - my;
        const dist = dx * dx + dy * dy;
        if (dist < bestD) {
          bestD = dist;
          best = d;
        }
      }
      dots.forEach((d) => {
        d.dot.setAttribute("r", d === best ? 6.5 : 4.2);
        d.dot.setAttribute("stroke-width", d === best ? 1.8 : 1);
      });
      if (best) {
        const dv = best.p[1] - best.p[0];
        tooltip.textContent = "";
        tooltip.appendChild(
          el(
            "div",
            { class: "tt-row tt-kv" },
            el("span", { class: "name", text: xTok }),
            el("span", { class: "val", text: fmt(best.p[0], 1) })
          )
        );
        tooltip.appendChild(
          el(
            "div",
            { class: "tt-row tt-kv" },
            el("span", { class: "name", text: yTok }),
            el("span", { class: "val", text: fmt(best.p[1], 1) })
          )
        );
        tooltip.appendChild(
          el(
            "div",
            { class: "tt-row tt-kv hot" },
            el("span", { class: "name", text: "Δ" }),
            el("span", { class: "val", text: (dv >= 0 ? "+" : "") + fmt(dv, 1) }),
            el("span", {
              class: "delta",
              text: dv === 0 ? "equal" : yTok + (dv > 0 ? " higher" : " lower"),
            })
          )
        );
        tooltip.classList.add("on");
        placeTip(wrap, tooltip, (sx(best.p[0]) / W) * rect.width, (sy(best.p[1]) / H) * rect.height - 40);
      } else {
        tooltip.classList.remove("on");
      }
    });
    svgRoot.addEventListener("pointerleave", () => {
      tooltip.classList.remove("on");
      dots.forEach((d) => {
        d.dot.setAttribute("r", 4.2);
        d.dot.setAttribute("stroke-width", 1);
      });
    });
  }

  function scatterChart(fig, data) {
    const body = card(fig, data.title, "");
    if (data.panels) {
      const row = el("div", { class: "viz-row" });
      body.appendChild(row);
      for (const panel of data.panels) {
        const cell = el("div", {});
        row.appendChild(cell);
        scatterPanel(cell, panel);
      }
    } else {
      /* card header already shows the title; don't repeat it inside the SVG */
      scatterPanel(body, { ...data, title: null });
    }
    if (data.caption) body.appendChild(el("figcaption", { text: data.caption }));
  }

  /* ---------------- grouped bar chart ---------------- */

  function barPanel(container, panel) {
    const W = 460;
    const H = 400;
    const M = { l: 52, r: 12, t: 46, b: panel.rotateLabels ? 86 : 50 };
    const wrap = el("div", { class: "viz-plot" });
    const svgRoot = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}` });
    const tooltip = el("div", { class: "viz-tooltip" });
    wrap.appendChild(svgRoot);
    wrap.appendChild(tooltip);
    container.appendChild(wrap);

    const yDom = panel.yDomain;
    const sy = (v) => H - M.b - ((v - yDom[0]) / (yDom[1] - yDom[0])) * (H - M.t - M.b);
    const fmtB = (v) => (yDom[1] <= 1 ? v.toFixed(3) : fmt(v, 1));

    svgRoot.appendChild(
      svg("text", {
        class: "panel-title",
        x: (M.l + W - M.r) / 2,
        y: 16,
        "text-anchor": "middle",
        text: panel.title.toUpperCase(),
      })
    );
    if (panel.yLabel) {
      svgRoot.appendChild(
        svg("text", {
          class: "ax-label",
          x: M.l,
          y: M.t - 10,
          "text-anchor": "start",
          text: panel.yLabel.toUpperCase(),
        })
      );
    }
    for (const t of niceTicks(yDom[0], yDom[1], 6)) {
      svgRoot.appendChild(
        svg("line", {
          x1: M.l,
          x2: W - M.r,
          y1: sy(t),
          y2: sy(t),
          stroke: "#eae4d8",
          "shape-rendering": "crispEdges",
        })
      );
      svgRoot.appendChild(
        svg("text", { class: "tick", x: M.l - 6, y: sy(t) + 4, "text-anchor": "end", text: fmt(t) })
      );
    }

    const nCat = panel.categories.length;
    const nSer = panel.series.length;
    const slot = (W - M.l - M.r) / nCat;
    const barW = Math.min(46, (slot * 0.72) / nSer);
    const colors = panel.series.map((ser) => harmonize(ser.color));

    panel.categories.forEach((cat, ci) => {
      const cx = M.l + slot * (ci + 0.5);
      panel.series.forEach((ser, si) => {
        const v = ser.values[ci];
        const err = ser.errors ? ser.errors[ci] : null;
        const x = cx - (nSer * barW) / 2 + si * barW;
        const rect = svg("rect", {
          x: x + 1,
          width: barW - 2,
          y: sy(v),
          height: Math.max(0, sy(yDom[0]) - sy(v)),
          rx: 1,
          fill: colors[si],
        });
        svgRoot.appendChild(rect);
        if (err) {
          const mid = x + barW / 2;
          const g = svg(
            "g",
            { stroke: "#3d3a35", "stroke-width": 1.4 },
            svg("line", { x1: mid, x2: mid, y1: sy(v - err), y2: sy(v + err) }),
            svg("line", { x1: mid - 5, x2: mid + 5, y1: sy(v + err), y2: sy(v + err) }),
            svg("line", { x1: mid - 5, x2: mid + 5, y1: sy(v - err), y2: sy(v - err) })
          );
          svgRoot.appendChild(g);
        }
        rect.addEventListener("pointerenter", () => {
          tooltip.textContent = "";
          tooltip.appendChild(el("div", { class: "tt-x", text: cat + " · " + ser.name }));
          tooltip.appendChild(
            el(
              "div",
              { class: "tt-row tt-kv hot" },
              el("span", { class: "val", text: fmtB(v) + (err ? " ± " + fmtB(err) : "") })
            )
          );
          /* compare against the sibling series in the same category */
          if (nSer === 2) {
            const other = panel.series[1 - si];
            const ov = other.values[ci];
            const oerr = other.errors ? other.errors[ci] : null;
            const d = v - ov;
            let note = "Δ vs " + other.name + ": " + (d >= 0 ? "+" : "") + fmtB(d);
            if (err !== null && oerr !== null) {
              note += Math.abs(d) <= err + oerr ? " — within combined CI" : " — outside combined CI";
            }
            tooltip.appendChild(el("div", { class: "tt-note", text: note }));
          }
          tooltip.classList.add("on");
          const rectB = svgRoot.getBoundingClientRect();
          placeTip(wrap, tooltip, ((x + barW / 2) / W) * rectB.width, (sy(v) / H) * rectB.height - 34, 10);
        });
        rect.addEventListener("pointerleave", () => tooltip.classList.remove("on"));
      });
      const label = svg("text", {
        class: "tick",
        "text-anchor": panel.rotateLabels ? "end" : "middle",
        ...(panel.rotateLabels
          ? { transform: `translate(${cx + 4} ${H - M.b + 14}) rotate(-28)` }
          : { x: cx, y: H - M.b + 16 }),
        text: cat,
      });
      svgRoot.appendChild(label);
    });
    svgRoot.appendChild(
      svg("line", {
        x1: M.l,
        x2: W - M.r,
        y1: sy(yDom[0]),
        y2: sy(yDom[0]),
        stroke: "#c4bcae",
        "shape-rendering": "crispEdges",
      })
    );
  }

  function barChart(fig, data) {
    const body = card(fig, data.title, "");
    const legendSpec = data.panels.find((p) => p.series.length > 1);
    if (legendSpec) {
      const legend = el("div", { class: "viz-legend" });
      for (const s of legendSpec.series) {
        legend.appendChild(
          el(
            "span",
            { class: "viz-chip", style: "cursor:default" },
            el("span", { class: "swatch", style: `border-top-color:${harmonize(s.color)}` }),
            el("span", { text: s.name })
          )
        );
      }
      body.appendChild(legend);
    }
    const row = el("div", { class: "viz-row" });
    body.appendChild(row);
    for (const panel of data.panels) {
      const cell = el("div", {});
      row.appendChild(cell);
      barPanel(cell, panel);
    }
    if (data.caption) body.appendChild(el("figcaption", { text: data.caption }));
  }

  /* ---------------- diagram engine ---------------- */

  const NODE_STYLE = {
    bf16: { fill: "#f4f1ea", stroke: "#b9b2a5" },
    nvfp4: { fill: "#e4eeec", stroke: "#176b64" },
    fprop: { fill: "#e6f1e2", stroke: "#5f9f5d" },
    gemm: { fill: "#f4ecdb", stroke: "#c2a35c" },
    opt: { fill: "#f6e7e2", stroke: "#b3543f" },
    dq: { fill: "#e9eef6", stroke: "#5b8fc9" },
    rowcopy: { fill: "#dceafd", stroke: "#6e99d4" },
    colcopy: { fill: "#eadff0", stroke: "#9a73ad" },
    io: { fill: "#eef1f5", stroke: "#8296ad" },
    plain: { fill: "#fffdf8", stroke: "#55504a" },
  };

  const PARTICLE = {
    hp: { color: "#8a8378", quantized: false },
    q: { color: "#176b64", quantized: true },
    dq: { color: "#5b8fc9", quantized: false, ring: true },
    grad: { color: "#a33e2d", quantized: false },
    opt: { color: "#b3833f", quantized: false },
  };

  function buildDiagram(fig, spec) {
    const body = card(fig, spec.title, spec.subtitle || "", "hover over boxes for details");
    const wrap = el("div", { class: "viz-diagram" });
    body.appendChild(wrap);
    const svgRoot = svg("svg", { viewBox: `0 0 ${spec.w} ${spec.h}` });
    wrap.appendChild(svgRoot);
    const tooltip = el("div", { class: "viz-tooltip" });
    const plotWrap = el("div", { class: "viz-plot" });
    wrap.replaceWith(plotWrap);
    plotWrap.appendChild(wrap);
    plotWrap.appendChild(tooltip);

    const defs = svg("defs", {});
    defs.innerHTML =
      '<marker id="dg-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">' +
      '<path d="M0 0.6 L9.6 5 L0 9.4 z" fill="#55504a"/></marker>';
    svgRoot.appendChild(defs);

    /* edges under nodes */
    const edgeLayer = svg("g", {});
    const nodeLayer = svg("g", {});
    const particleLayer = svg("g", { style: "display:none" });
    svgRoot.appendChild(edgeLayer);
    svgRoot.appendChild(nodeLayer);
    svgRoot.appendChild(particleLayer);

    const edgeEls = {};
    for (const e of spec.edges) {
      const d = "M" + e.p.map((pt) => pt.join(" ")).join(" L");
      const g = svg("g", { class: "dg-edge" });
      const wire = svg("path", {
        class: "wire",
        d,
        ...(e.dashed ? { "stroke-dasharray": "7 6" } : {}),
        "marker-end": e.noArrow ? "" : "url(#dg-arrow)",
      });
      g.appendChild(wire);
      if (e.label) {
        const [lx, ly] = e.labelAt || midpoint(e.p);
        g.appendChild(
          svg("text", {
            x: lx,
            y: ly,
            "font-size": 12.5,
            fill: "#55504a",
            "text-anchor": e.labelAnchor || "middle",
            text: e.label,
          })
        );
      }
      edgeLayer.appendChild(g);
      edgeEls[e.id] = { g, e };
    }

    function midpoint(pts) {
      const i = Math.floor(pts.length / 2) - 1;
      const a = pts[Math.max(0, i)];
      const b = pts[Math.min(pts.length - 1, i + 1)];
      return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 7];
    }

    const nodeEls = {};
    for (const [id, n] of Object.entries(spec.nodes)) {
      const style = NODE_STYLE[n.kind || "plain"];
      const g = svg("g", { class: "dg-node" });
      nodeEls[id] = g;
      g.appendChild(
        svg("rect", {
          x: n.x,
          y: n.y,
          width: n.w,
          height: n.h,
          rx: 10,
          fill: style.fill,
          stroke: style.stroke,
          "stroke-width": 1.8,
          ...(n.dashed ? { "stroke-dasharray": "7 5" } : {}),
        })
      );
      const lines = [n.title].concat(n.sub || []);
      const lh = 17;
      const y0 = n.y + n.h / 2 - ((lines.length - 1) * lh) / 2;
      lines.forEach((txt, i) => {
        g.appendChild(
          svg("text", {
            x: n.x + n.w / 2,
            y: y0 + i * lh + 4.5,
            "text-anchor": "middle",
            "font-size": i === 0 ? 14.5 : 11.5,
            "font-weight": i === 0 ? 600 : 400,
            fill: "#2b2823",
            text: txt,
          })
        );
      });
      nodeLayer.appendChild(g);
      if (n.info) {
        g.addEventListener("pointerenter", () => {
          tooltip.textContent = "";
          tooltip.appendChild(el("div", { class: "tt-x", text: n.title }));
          tooltip.appendChild(el("div", { text: n.info }));
          tooltip.classList.add("on");
          const rect = svgRoot.getBoundingClientRect();
          placeTip(plotWrap, tooltip, ((n.x + n.w / 2) / spec.w) * rect.width, ((n.y + n.h) / spec.h) * rect.height + 6, 8);
          highlight(id, true);
        });
        g.addEventListener("pointerleave", () => {
          tooltip.classList.remove("on");
          highlight(id, false);
        });
      }
    }

    function edgeTouches(e, nodeId) {
      const n = spec.nodes[nodeId];
      if (!n) return false;
      const near = (pt) =>
        pt[0] >= n.x - 14 && pt[0] <= n.x + n.w + 14 && pt[1] >= n.y - 14 && pt[1] <= n.y + n.h + 14;
      return near(e.p[0]) || near(e.p[e.p.length - 1]);
    }

    function highlight(nodeId, on) {
      for (const { g, e } of Object.values(edgeEls)) {
        const touch = edgeTouches(e, nodeId);
        g.classList.toggle("dim", on && !touch);
        g.querySelector(".wire").classList.toggle("hot", on && touch);
      }
    }

    /* flows -> particles */
    const flowRuntime = [];
    for (const flow of spec.flows || []) {
      const pts = [];
      const boundaries = [];
      for (const step of flow.path) {
        const e = spec.edges.find((x) => x.id === step.edge);
        if (!e) continue;
        for (const p of e.p) {
          if (!pts.length || pts[pts.length - 1][0] !== p[0] || pts[pts.length - 1][1] !== p[1]) {
            pts.push(p);
          }
        }
        boundaries.push({ end: pts.length - 1, becomes: step.becomes || null });
      }
      const path = svg("path", {
        d: "M" + pts.map((p) => p.join(" ")).join(" L"),
        fill: "none",
        stroke: "none",
      });
      particleLayer.appendChild(path);
      const total = path.getTotalLength();
      /* cumulative length at each pt index */
      const cum = [0];
      for (let i = 1; i < pts.length; i++) {
        const dx = pts[i][0] - pts[i - 1][0];
        const dy = pts[i][1] - pts[i - 1][1];
        cum.push(cum[i - 1] + Math.hypot(dx, dy));
      }
      const marks = boundaries
        .filter((b) => b.becomes)
        .map((b) => ({ at: cum[b.end] / total, becomes: b.becomes }));
      const count = flow.count || Math.max(1, Math.round(total / 260));
      const particles = [];
      for (let i = 0; i < count; i++) {
        const g = svg("g", {});
        particleLayer.appendChild(g);
        particles.push({ g, offset: i / count, lastState: null });
      }
      flowRuntime.push({ path, total, marks, particles, flow, speed: flow.speed || 1 });
    }

    function particleShape(g, state, flash) {
      g.textContent = "";
      const cfg = PARTICLE[state] || PARTICLE.hp;
      if (cfg.quantized) {
        g.appendChild(
          svg("rect", {
            x: -5,
            y: -5,
            width: 10,
            height: 10,
            rx: 2,
            fill: cfg.color,
            stroke: "#fffdf8",
            "stroke-width": 1.2,
          })
        );
      } else {
        g.appendChild(
          svg("circle", {
            r: 5.5,
            fill: cfg.color,
            "fill-opacity": cfg.ring ? 0.35 : 1,
            stroke: cfg.ring ? cfg.color : "#fffdf8",
            "stroke-width": cfg.ring ? 2 : 1.2,
            ...(cfg.ring ? { "stroke-dasharray": "3 2.4" } : {}),
          })
        );
      }
      if (flash) {
        const ring = svg("circle", {
          r: 6,
          fill: "none",
          stroke: cfg.color,
          "stroke-width": 2,
          opacity: 0.9,
        });
        g.appendChild(ring);
        ring.animate(
          [
            { opacity: 0.9, transform: "scale(1)" },
            { opacity: 0, transform: "scale(2.6)" },
          ],
          { duration: 420, easing: "ease-out" }
        ).onfinish = () => ring.remove();
      }
    }

    let playing = false; /* flow animates only after the reader presses play */
    let visible = false;
    let raf = null;
    let t0 = performance.now();
    let acc = 0;

    function frame(now) {
      raf = null;
      const dt = Math.min(0.05, (now - t0) / 1000);
      t0 = now;
      if (playing && visible) acc += dt;
      for (const fr of flowRuntime) {
        const cycle = fr.total / (110 * fr.speed); /* seconds per loop */
        for (const p of fr.particles) {
          const u = (acc / cycle + p.offset) % 1;
          const pos = fr.path.getPointAtLength(u * fr.total);
          let state = fr.flow.start || "hp";
          let flashed = false;
          for (const m of fr.marks) {
            if (u >= m.at) state = m.becomes;
          }
          if (p.lastU !== undefined) {
            for (const m of fr.marks) {
              const crossed = p.lastU < m.at && u >= m.at;
              if (crossed) flashed = true;
            }
            if (p.lastU > u) p.lastState = null; /* looped */
          }
          if (state !== p.lastState || flashed) {
            particleShape(p.g, state, flashed);
            p.lastState = state;
          }
          p.g.setAttribute("transform", `translate(${pos.x} ${pos.y})`);
          p.lastU = u;
        }
      }
      if (playing && visible) raf = requestAnimationFrame(frame);
    }

    function ensureLoop() {
      if (!raf && playing && visible) {
        t0 = performance.now();
        raf = requestAnimationFrame(frame);
      }
    }

    whenVisible(svgRoot, (v) => {
      visible = v;
      ensureLoop();
    });

    const controls = el("div", { class: "viz-controls" });

    /* optional mode buttons (e.g. default / unquant / dequant backward) */
    if (spec.modes && spec.modes.length) {
      const note = el("span", { class: "viz-sub", text: spec.modes[0].note || "" });
      const btns = [];
      for (const mode of spec.modes) {
        const btn = el("button", {
          class: "viz-btn",
          "aria-pressed": String(mode === spec.modes[0]),
          text: mode.label,
          onclick: () => {
            btns.forEach((b) => b.setAttribute("aria-pressed", "false"));
            btn.setAttribute("aria-pressed", "true");
            const dimSet = new Set(mode.dim || []);
            for (const [nid, g] of Object.entries(nodeEls)) {
              g.classList.toggle("dim", dimSet.has(nid));
            }
            for (const { g, e } of Object.values(edgeEls)) {
              const touchDim = [...dimSet].some((nid) => edgeTouches(e, nid));
              g.classList.toggle("dim", touchDim);
            }
            note.textContent = mode.note || "";
          },
        });
        btns.push(btn);
        controls.appendChild(btn);
      }
      controls.appendChild(note);
    }

    const playBtn = el("button", {
      class: "viz-btn",
      "aria-pressed": "false",
      text: "play flow",
      onclick: () => {
        playing = !playing;
        playBtn.setAttribute("aria-pressed", String(playing));
        playBtn.textContent = playing ? "pause flow" : "play flow";
        particleLayer.style.display = playing ? "" : "none";
        ensureLoop();
      },
    });
    controls.appendChild(playBtn);
    if (spec.flowNote) controls.appendChild(el("span", { class: "viz-sub", text: spec.flowNote }));
    if (spec.legend) {
      const lg = el("div", { class: "dg-legend" });
      for (const item of spec.legend) {
        lg.appendChild(
          el(
            "span",
            { class: "key" },
            el("span", {
              class: "box",
              style: `background:${item.color};${item.dashed ? "border-style:dashed;" : ""}`,
            }),
            el("span", { text: item.label })
          )
        );
      }
      controls.appendChild(lg);
    }
    body.appendChild(controls);
    if (spec.caption) body.appendChild(el("figcaption", { text: spec.caption }));
    frame(performance.now());
  }

  /* ---------------- registry + boot ---------------- */

  NV.registry = NV.registry || {};

  NV.registerChart = function (id, opts) {
    NV.registry[id] = (fig) => {
      const data = NV.data[id];
      if (!data) throw new Error("no data for " + id);
      if (data.type === "scatter" || data.type === "scatterPanels") scatterChart(fig, data);
      else if (data.type === "bars") barChart(fig, data);
      else lineChart(fig, data, opts);
    };
  };

  NV.registerDiagram = function (id, spec) {
    /* spec functions receive a narrow flag so diagrams can re-lay out on phones */
    NV.registry[id] = (fig) => buildDiagram(fig, typeof spec === "function" ? spec((fig.clientWidth || 700) < 520) : spec);
  };

  NV.helpers = { el, svg, fmt, niceTicks, card, whenVisible, buildDiagram, placeTip };

  const widthMode = (fig) => (fig.clientWidth || 700) < 520;

  function boot() {
    document.querySelectorAll("figure.nvfp4-viz").forEach((fig, i) => {
      fig.dataset.figN = i + 1;
      const id = fig.dataset.viz;
      const build = NV.registry[id];
      if (!build || fig.classList.contains("viz-active")) return;
      try {
        /* keep references to the fallback nodes so a re-layout can restore them
           (widgets may move them, e.g. the sim adopts its md figcaption) */
        fig.__vizOriginal = [...fig.childNodes];
        fig.__vizMode = widthMode(fig);
        build(fig);
        fig.classList.add("viz-active");
      } catch (err) {
        console.error("nvfp4-viz: failed to build", id, err);
        releaseWatchers(fig);
        fig.textContent = "";
        for (const node of fig.__vizOriginal) fig.appendChild(node);
      }
    });
  }

  /* figures whose width class changed (rotation, window resize) rebuild with
     the layout for their new size; lever state resets, which is acceptable
     for a breakpoint crossing */
  function rebuildIfModeChanged(fig) {
    const build = NV.registry[fig.dataset.viz];
    if (!build || !fig.__vizOriginal || widthMode(fig) === fig.__vizMode) return;
    fig.__vizMode = widthMode(fig);
    releaseWatchers(fig);
    fig.textContent = "";
    for (const node of fig.__vizOriginal) fig.appendChild(node);
    try {
      build(fig);
    } catch (err) {
      console.error("nvfp4-viz: failed to rebuild", fig.dataset.viz, err);
    }
  }

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      document.querySelectorAll("figure.nvfp4-viz.viz-active").forEach(rebuildIfModeChanged);
    }, 180);
  });

  /* figures hidden inside a closed <details> build at the 700px fallback width;
     re-check their layout when the fold opens ('toggle' does not bubble) */
  document.addEventListener(
    "toggle",
    (evt) => {
      if (!(evt.target instanceof HTMLElement) || !evt.target.open) return;
      evt.target.querySelectorAll("figure.nvfp4-viz.viz-active").forEach(rebuildIfModeChanged);
    },
    true
  );

  /* viz-widgets.js registers every widget and then calls NV.boot(); under
     `defer` this file runs first, so booting here would see an empty registry */
  NV.boot = boot;
})();
