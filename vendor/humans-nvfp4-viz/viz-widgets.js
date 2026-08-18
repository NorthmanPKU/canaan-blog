/* Diagram specs and bespoke widgets for the NVFP4 RL blog.
 * Loaded after viz.js (which supplies NVFP4_VIZ.helpers and the registries).
 */
(function () {
  "use strict";

  const NV = window.NVFP4_VIZ;
  const { el, svg, fmt, card, placeTip } = NV.helpers;

  /* =========================================================
   * RL recipe flow charts (hp / dequantized / dequantized+4over6)
   * ========================================================= */

  function rlRecipe(variant, narrow) {
    const dq = variant !== "hp";
    const four = variant === "dq46";
    const titles = {
      hp: "Naive Per-Token NVFP4 RL Recipe",
      dq: "NVFP4 RL Recipe with Dequantized Backward",
      dq46: "humans& NVFP4 RL Recipe",
    };

    const actSub = ["row-scaled, 1x16 block"].concat(four ? ["4over6 adaptive scaling"] : []);
    const wSub = ["tensor-scaled, 1x16 block"].concat(four ? ["4over6 adaptive scaling"] : []);

    const nodes = {
      bf16act: {
        x: 20, y: 250, w: 190, h: 64, kind: "bf16", title: "BF16 activation",
        info: "The layer input as produced by the previous layer. Kept in BF16; the quantized copy is derived from it every step.",
      },
      nvfp4act: {
        x: 250, y: 170, w: 250, h: dq && four ? 100 : 88, kind: "nvfp4",
        title: "NVFP4 activation", sub: actSub,
        info: "Online per-token activation scaling: each token computes its own FP32 scale at runtime, so quantization is local to the token and needs no calibration artifact.",
      },
      fprop: {
        x: 250, y: 330, w: 250, h: 100, kind: "gemm", title: "Fprop",
        sub: ["NVFP4 tensor core, FP32 acc", "row-scaled epilogue scaling"],
        info: "Forward GEMM runs on NVFP4 tensor cores with FP32 accumulation. This is the same quantized function rollout serves.",
      },
      bf16w: {
        x: 20, y: 440, w: 190, h: 64, kind: "bf16", title: "BF16 weight",
        info: "Training-side BF16 weight, updated by the optimizer, then re-quantized for the next forward pass and for rollout.",
      },
      nvfp4w: {
        x: 250, y: 470, w: 250, h: dq && four ? 100 : 88, kind: "nvfp4",
        title: "NVFP4 weight", sub: wSub,
        info: "MoE expert weights quantized to NVFP4 (one FP32 tensor scale + E4M3 scale per 16-value block). The same bytes feed training forward and rollout.",
      },
      wgrad: {
        x: 640, y: 20, w: 270, h: 80, kind: "gemm", title: "Wgrad",
        sub: ["BF16 tensor core, FP32 acc"],
        info: "Weight-gradient GEMM stays on BF16 tensor cores with FP32 accumulation" + (dq ? ", but reads the dequantized activation, so it differentiates the exact quantized forward function." : ", reading the original BF16 activation — a smooth surrogate of the quantized forward function."),
      },
      opt: {
        x: 640, y: 310, w: 270, h: 86, kind: "opt", title: "Optimizer states",
        sub: ["FP32 master weight"],
        info: "FP32 master weights and optimizer states. Weight decay is set to 0 for NVFP4 weights so tiny continuous shrinkage cannot hop an FP4 bin boundary.",
      },
      dgrad: {
        x: 640, y: 616, w: 270, h: 80, kind: "gemm", title: "Dgrad",
        sub: ["BF16 tensor core, FP32 acc"],
        info: "Data-gradient GEMM in BF16 with FP32 accumulation" + (dq ? ", reading the dequantized weight so backward matches the forward clipping and rounding decisions." : ", reading the original BF16 weight (high-precision surrogate)."),
      },
      bf16grad: {
        x: 990, y: 440, w: 170, h: 64, kind: "bf16", title: "BF16 gradient",
        info: "Incoming gradient from the next layer; stays BF16 end to end.",
      },
    };
    if (dq) {
      nodes.actdq = {
        x: 250, y: 20, w: 250, h: 74, kind: "dq", dashed: true,
        title: "BF16 activation", sub: ["dequantized from NVFP4"],
        info: "dequantize(Q(x)): the exact NVFP4 values decoded back to BF16. Same numbers the forward GEMM consumed — quantization staircase included.",
      };
      nodes.wdq = {
        x: 250, y: 620, w: 250, h: 74, kind: "dq", dashed: true,
        title: "BF16 weight", sub: ["dequantized from NVFP4"],
        info: "dequantize(Q(w)): backward differentiates the function the quantized forward actually evaluated (w_bprop = dequantize(Q(w_bf16))).",
      };
    }

    const edges = [
      { id: "actQ", p: [[150, 250], [150, 217], [250, 217]], label: "quantize", labelAt: [196, 207] },
      { id: "actFprop", p: [[375, nodes.nvfp4act.y + nodes.nvfp4act.h], [375, 330]] },
      { id: "wQ", p: [[150, 504], [150, 517], [250, 517]], label: "quantize", labelAt: [196, 545] },
      { id: "wFprop", p: [[375, 470], [375, 430]] },
      { id: "gradWgrad", p: [[1075, 440], [1075, 60], [910, 60]], label: "keep BF16", labelAt: [1068, 250], labelAnchor: "end" },
      { id: "gradDgrad", p: [[1075, 504], [1075, 656], [910, 656]], label: "keep BF16", labelAt: [1068, 590], labelAnchor: "end" },
      { id: "wgradOpt", p: [[775, 100], [775, 310]] },
      { id: "optW", p: [[775, 396], [775, 450], [214, 450]], dashed: true },
      { id: "fpropNext", p: [[250, 395], [115, 395], [115, 318]], dashed: true, label: "to next layer", labelAt: [182, 385] },
      { id: "wRollout", p: [[500, 517], [612, 517]], dashed: true, label: "to rollout", labelAt: [622, 521], labelAnchor: "start" },
      { id: "dgradPrev", p: [[775, 616], [775, 480], [986, 480]], dashed: true, label: "to prev layer", labelAt: [880, 468] },
    ];
    if (dq) {
      edges.push(
        { id: "actDq", p: [[375, 170], [375, 98]], label: "dequantize", labelAt: [388, 140], labelAnchor: "start" },
        { id: "actdqWgrad", p: [[500, 57], [640, 57]] },
        { id: "wDq", p: [[375, nodes.nvfp4w.y + nodes.nvfp4w.h], [375, 620]], label: "dequantize", labelAt: [388, 600], labelAnchor: "start" },
        { id: "wdqDgrad", p: [[500, 657], [640, 657]] }
      );
    } else {
      edges.push(
        { id: "actKeep", p: [[80, 250], [80, 57], [636, 57]], label: "keep BF16", labelAt: [200, 45] },
        { id: "wKeep", p: [[80, 504], [80, 657], [636, 657]], label: "keep BF16", labelAt: [200, 645] }
      );
    }

    const flows = [
      { path: [{ edge: "actQ", becomes: "q" }, { edge: "actFprop" }], start: "hp" },
      { path: [{ edge: "wQ", becomes: "q" }, { edge: "wFprop" }], start: "hp" },
      { path: [{ edge: "wRollout" }], start: "q", count: 1 },
      { path: [{ edge: "gradWgrad" }], start: "grad" },
      { path: [{ edge: "gradDgrad" }], start: "grad" },
      { path: [{ edge: "wgradOpt", becomes: "opt" }, { edge: "optW" }], start: "grad" },
      { path: [{ edge: "dgradPrev" }], start: "grad", count: 2 },
      { path: [{ edge: "fpropNext" }], start: "hp", count: 1 },
    ];
    if (dq) {
      flows.push(
        { path: [{ edge: "actDq", becomes: "dq" }, { edge: "actdqWgrad" }], start: "q", count: 2 },
        { path: [{ edge: "wDq", becomes: "dq" }, { edge: "wdqDgrad" }], start: "q", count: 2 }
      );
    } else {
      flows.push(
        { path: [{ edge: "actKeep" }], start: "hp" },
        { path: [{ edge: "wKeep" }], start: "hp" }
      );
    }

    /* phones: single-column geometry (node semantics unchanged, edges re-routed) */
    let w = 1180;
    let h = 720;
    let laidEdges = edges;
    if (narrow) {
      const g = rlRecipeNarrowGeometry(variant);
      for (const [id, geo] of Object.entries(g.nodes)) Object.assign(nodes[id], geo);
      laidEdges = g.edges;
      w = g.w;
      h = g.h;
    }
    return {
      w, h,
      title: titles[variant],
      flowNote: "● high-precision · ■ NVFP4-quantized · ◌ dequantized · red = gradients",
      nodes, edges: laidEdges, flows,
      legend: [
        { color: "#d5e8d4", label: "NVFP4 tensor" },
        { color: "#fff2cc", label: "GEMM" },
        { color: "#ffe6cc", label: "optimizer" },
        { color: "#f5f5f5", label: "BF16 tensor" },
      ].concat(dq ? [{ color: "#dae8fc", dashed: true, label: "dequantized backward operand" }] : []),
      caption: dq
        ? "Watch the round dots quantize into squares before the forward GEMM — and dequantize (dashed circles) before feeding the backward GEMMs."
        : "Round dots are BF16 values; they quantize into squares only on the forward path, while backward reads the original BF16 tensors.",
    };
  }

  function rlRecipeNarrowGeometry(variant) {
    const dq = variant !== "hp";
    const four = variant === "dq46";
    const CANVAS_W = 500;
    const CX = 160; // main column center
    const nvfp4Height = four ? 76 : 60; // one extra sub line for 4over6

    const nodes = {};
    let cursorY = 18;
    function place(id, boxWidth, boxHeight, gapAfter) {
      nodes[id] = { x: CX - boxWidth / 2, y: cursorY, w: boxWidth, h: boxHeight };
      cursorY += boxHeight + gapAfter;
      return nodes[id];
    }

    /* Single main column, reading order = data flow:
       forward (act -> fprop <- weights), then backward (dq copies feed the GEMMs). */
    const bf16act = place("bf16act", 170, 48, 36);
    const nvfp4act = place("nvfp4act", 190, nvfp4Height, 30);
    const fprop = place("fprop", 240, 74, 40);
    const nvfp4w = place("nvfp4w", 190, nvfp4Height, 34);
    const bf16w = place("bf16w", 170, 48, 34);
    const actdq = dq ? place("actdq", 170, 58, 26) : null;
    const wgrad = place("wgrad", 240, 58, 30);
    const opt = place("opt", 240, 58, 30);
    nodes.bf16grad = { x: 320, y: opt.y, w: 150, h: 48 }; // right of opt, between wgrad and dgrad
    const bf16grad = nodes.bf16grad;
    const wdq = dq ? place("wdq", 170, 58, 26) : null;
    const dgrad = place("dgrad", 240, 58, 0);

    const h = dgrad.y + dgrad.h + 54; // room for the dgradPrev exit stub + label

    const cy = (n) => n.y + n.h / 2;
    const bot = (n) => n.y + n.h;

    /* hp routes actKeep into wgrad's right edge and wKeep out of bf16w's left edge,
       so the shared entry points shift to keep those long edges crossing-free. */
    const wgradEntryY = dq ? cy(wgrad) : cy(wgrad) + 16;
    const bf16wEntryY = dq ? cy(bf16w) : bot(bf16w) - 12;

    const edges = [
      { id: "actQ", p: [[CX, bot(bf16act)], [CX, nvfp4act.y]], label: "quantize", labelAt: [CX + 10, (bot(bf16act) + nvfp4act.y) / 2 + 4], labelAnchor: "start" },
      { id: "actFprop", p: [[CX, bot(nvfp4act)], [CX, fprop.y]] },
      { id: "wQ", p: [[CX, bf16w.y], [CX, bot(nvfp4w)]], label: "quantize", labelAt: [CX + 10, (bot(nvfp4w) + bf16w.y) / 2 + 4], labelAnchor: "start" },
      { id: "wFprop", p: [[200, nvfp4w.y], [200, bot(fprop)]] },
      { id: "gradWgrad", p: [[395, bf16grad.y], [395, wgradEntryY], [280, wgradEntryY]], label: "keep BF16", labelAt: [403, bot(wgrad) + 19], labelAnchor: "start" },
      { id: "gradDgrad", p: [[395, bot(bf16grad)], [395, cy(dgrad)], [280, cy(dgrad)]], label: "keep BF16", labelAt: [403, bot(bf16grad) + 26], labelAnchor: "start" },
      { id: "wgradOpt", p: [[CX, bot(wgrad)], [CX, opt.y]] },
      { id: "optW", p: [[40, cy(opt)], [28, cy(opt)], [28, bf16wEntryY], [75, bf16wEntryY]], dashed: true },
      { id: "fpropNext", p: [[100, bot(fprop)], [100, bot(fprop) + 20], [26, bot(fprop) + 20]], dashed: true, label: "to next layer", labelAt: [34, bot(fprop) + 34], labelAnchor: "start" },
      { id: "wRollout", p: [[255, cy(nvfp4w)], [340, cy(nvfp4w)]], dashed: true, label: "to rollout", labelAt: [348, cy(nvfp4w) + 4], labelAnchor: "start" },
      { id: "dgradPrev", p: [[CX, bot(dgrad)], [CX, bot(dgrad) + 32]], dashed: true, label: "to prev layer", labelAt: [CX + 12, bot(dgrad) + 26], labelAnchor: "start" },
    ];

    if (dq) {
      edges.push(
        { id: "actDq", p: [[255, cy(nvfp4act)], [464, cy(nvfp4act)], [464, cy(actdq)], [245, cy(actdq)]], label: "dequantize", labelAt: [456, cy(fprop)], labelAnchor: "end" },
        { id: "actdqWgrad", p: [[CX, bot(actdq)], [CX, wgrad.y]] },
        { id: "wDq", p: [[65, cy(nvfp4w)], [12, cy(nvfp4w)], [12, cy(wdq)], [75, cy(wdq)]], label: "dequantize", labelAt: [20, wdq.y - 12], labelAnchor: "start" },
        { id: "wdqDgrad", p: [[CX, bot(wdq)], [CX, dgrad.y]] }
      );
    } else {
      edges.push(
        { id: "actKeep", p: [[245, cy(bf16act)], [464, cy(bf16act)], [464, cy(wgrad) - 16], [280, cy(wgrad) - 16]], label: "keep BF16", labelAt: [456, cy(fprop)], labelAnchor: "end" },
        { id: "wKeep", p: [[75, bf16w.y + 12], [12, bf16w.y + 12], [12, cy(dgrad)], [40, cy(dgrad)]], label: "keep BF16", labelAt: [20, bot(wgrad) + 19], labelAnchor: "start" }
      );
    }

    return { w: CANVAS_W, h, nodes, edges };
  }

  function mxfp8TeNarrowGeometry() {
    /* Single-column narrow layout (w=500). Reading order: forward pass
     * (input -> row/col activation copies -> Fprop -> weight -> row/col weight
     * copies) then backward (gradient -> col/row gradient copies -> Wgrad/Dgrad
     * -> optimizer). Long "saved for backward" edges run on side rails:
     * colIn2wg on the left (x=26), colW2d on the right (x=478), and the dashed
     * optimizer->weight return climbs the far-right margin (x=492). */
    const nodes = {
      input: { x: 150, y: 20, w: 200, h: 56 },
      colIn: { x: 40, y: 118, w: 200, h: 64 },
      rowIn: { x: 260, y: 118, w: 200, h: 64 },
      fprop: { x: 140, y: 230, w: 220, h: 72 },
      weight: { x: 150, y: 350, w: 200, h: 56 },
      rowW: { x: 40, y: 454, w: 200, h: 64 },
      colW: { x: 260, y: 454, w: 200, h: 64 },
      outGrad: { x: 150, y: 566, w: 200, h: 56 },
      colG: { x: 40, y: 670, w: 200, h: 64 },
      rowG: { x: 260, y: 670, w: 200, h: 64 },
      wgrad: { x: 40, y: 782, w: 200, h: 72 },
      dgrad: { x: 260, y: 782, w: 200, h: 72 },
      opt: { x: 150, y: 902, w: 200, h: 72 },
    };
    const edges = [
      { id: "in2col", p: [[205, 76], [205, 98], [140, 98], [140, 118]], label: "quantize", labelAt: [172, 93] },
      { id: "in2row", p: [[295, 76], [295, 98], [360, 98], [360, 118]], label: "quantize", labelAt: [327, 93] },
      { id: "rowIn2f", p: [[360, 182], [360, 206], [305, 206], [305, 230]] },
      { id: "colIn2wg", p: [[140, 182], [140, 200], [26, 200], [26, 818], [40, 818]] },
      { id: "w2row", p: [[205, 406], [205, 428], [100, 428], [100, 454]], label: "quantize", labelAt: [152, 423] },
      { id: "w2col", p: [[295, 406], [295, 428], [360, 428], [360, 454]], label: "quantize", labelAt: [327, 423] },
      { id: "rowW2f", p: [[75, 454], [75, 326], [195, 326], [195, 302]] },
      { id: "colW2d", p: [[460, 486], [478, 486], [478, 818], [460, 818]] },
      { id: "grad2col", p: [[205, 622], [205, 644], [140, 644], [140, 670]], label: "quantize", labelAt: [172, 639] },
      { id: "grad2row", p: [[295, 622], [295, 644], [360, 644], [360, 670]], label: "quantize", labelAt: [327, 639] },
      { id: "colG2wg", p: [[140, 734], [140, 782]] },
      { id: "rowG2d", p: [[360, 734], [360, 782]] },
      { id: "wg2opt", p: [[140, 854], [140, 876], [205, 876], [205, 902]] },
      { id: "opt2w", p: [[350, 938], [492, 938], [492, 378], [350, 378]], dashed: true },
      { id: "f2next", p: [[360, 290], [430, 290]], dashed: true, label: "to next layer", labelAt: [396, 310] },
      { id: "wRollout", p: [[140, 518], [140, 552]], dashed: true, label: "to rollout", labelAt: [150, 545], labelAnchor: "start" },
      { id: "d2prev", p: [[360, 854], [360, 920]], dashed: true, label: "to prev layer", labelAt: [370, 904], labelAnchor: "start" },
    ];
    return { w: 500, h: 1000, nodes, edges };
  }

  NV.registerDiagram("recipe-nvfp4-hp", (narrow) => rlRecipe("hp", narrow));
  NV.registerDiagram("recipe-nvfp4-dq", (narrow) => rlRecipe("dq", narrow));
  NV.registerDiagram("recipe-nvfp4-dq46", (narrow) => rlRecipe("dq46", narrow));

  /* =========================================================
   * Original NVFP4 pretraining recipe
   * ========================================================= */

  NV.registerDiagram("recipe-nvfp4-pretrain", () => ({
    w: 1240, h: 640,
    title: "Original NVFP4 Pretraining Recipe",
    subtitle: "stochastic rounding (SR) + random Hadamard transforms (RHT) stabilize a fully low-precision backward",
    nodes: {
      inAct: { x: 20, y: 40, w: 170, h: 60, kind: "bf16", title: "BF16 Activation", info: "Layer input from layer i−1." },
      qAct: { x: 250, y: 40, w: 180, h: 60, kind: "opt", title: "Quantize to NVFP4", info: "Forward activation quantization (1x16 blocks, E4M3 block scales)." },
      fprop: { x: 500, y: 30, w: 210, h: 80, kind: "nvfp4", title: "FPROP", sub: ["NVFP4 GEMM"], info: "Forward GEMM on NVFP4 tensor cores; output returns to BF16 for the next layer." },
      q2d: { x: 500, y: 160, w: 210, h: 64, kind: "opt", title: "2D Block Quantize", sub: ["to NVFP4"], info: "Weights use 2D (16x16) block scaling so forward and transposed-backward views see consistent scales." },
      w32: { x: 500, y: 270, w: 210, h: 60, kind: "plain", title: "FP32 Weights", info: "Master weights held in FP32 by the optimizer." },
      optim: { x: 500, y: 380, w: 210, h: 60, kind: "opt", title: "Optimizer", info: "Adam-style update in FP32." },
      tW: { x: 250, y: 160, w: 130, h: 60, kind: "plain", title: "Transpose", info: "Transposed quantized weights for the data-gradient GEMM." },
      tAct: { x: 760, y: 120, w: 130, h: 56, kind: "plain", title: "Transpose", info: "Transposed activations for the weight-gradient GEMM." },
      tGrad: { x: 1060, y: 120, w: 130, h: 56, kind: "plain", title: "Transpose", info: "Transposed output gradient for the weight-gradient GEMM." },
      hAct: { x: 745, y: 210, w: 160, h: 60, kind: "opt", title: "Hadamard", sub: ["Transform"], info: "RHT spreads block-level outliers before quantization (weight-gradient GEMM inputs)." },
      hGrad: { x: 1045, y: 210, w: 160, h: 60, kind: "opt", title: "Hadamard", sub: ["Transform"], info: "RHT on the gradient side of the weight-gradient GEMM." },
      qActH: { x: 745, y: 305, w: 160, h: 60, kind: "opt", title: "Quantize", sub: ["to NVFP4"], info: "Round-to-nearest quantization of the transformed activations." },
      qGradH: { x: 1045, y: 305, w: 160, h: 64, kind: "opt", title: "Quantize to NVFP4", sub: ["with SR"], info: "Stochastic rounding keeps the quantized gradient unbiased in expectation." },
      wgrad: { x: 880, y: 405, w: 210, h: 76, kind: "nvfp4", title: "WGRAD", sub: ["NVFP4 GEMM"], info: "Weight-gradient GEMM fully in NVFP4." },
      inGrad: { x: 1030, y: 20, w: 190, h: 64, kind: "bf16", title: "BF16 Activation", sub: ["Gradient"], info: "Gradient arriving from layer i+1." },
      qGradSR: { x: 250, y: 430, w: 200, h: 64, kind: "opt", title: "Quantize to NVFP4", sub: ["with SR"], info: "Gradient quantized with stochastic rounding for the data-gradient GEMM." },
      dgrad: { x: 250, y: 540, w: 210, h: 76, kind: "nvfp4", title: "DGRAD", sub: ["NVFP4 GEMM"], info: "Data-gradient GEMM fully in NVFP4; result goes to layer i−1 in BF16." },
    },
    edges: [
      { id: "inQ", p: [[190, 70], [250, 70]], label: "BF16", labelAt: [220, 88] },
      { id: "qF", p: [[430, 70], [500, 70]], label: "NVFP4", labelAt: [465, 60] },
      { id: "fOut2", p: [[710, 60], [880, 60]], label: "BF16 → layer i+1", labelAt: [795, 50] },
      { id: "w2q", p: [[605, 270], [605, 224]] },
      { id: "q2f", p: [[605, 160], [605, 110]], label: "NVFP4", labelAt: [630, 140], labelAnchor: "start" },
      { id: "opt2w", p: [[605, 380], [605, 330]], label: "FP32", labelAt: [630, 360], labelAnchor: "start" },
      { id: "q2t", p: [[500, 192], [380, 192]], label: "NVFP4", labelAt: [443, 182] },
      { id: "t2d", p: [[315, 220], [315, 540]], label: "NVFP4", labelAt: [292, 380], labelAnchor: "end" },
      { id: "g2t", p: [[1125, 84], [1125, 120]] },
      { id: "actTap", p: [[220, 70], [220, 14], [885, 14], [885, 120]], label: "BF16 activation", labelAt: [875, 106], labelAnchor: "end" },
      { id: "t2h1", p: [[825, 176], [825, 210]] },
      { id: "t2h2", p: [[1125, 176], [1125, 210]] },
      { id: "h2q1", p: [[825, 270], [825, 305]] },
      { id: "h2q2", p: [[1125, 270], [1125, 305]] },
      { id: "q2wg1", p: [[825, 365], [825, 443], [880, 443]], label: "NVFP4", labelAt: [815, 420], labelAnchor: "end" },
      { id: "q2wg2", p: [[1125, 369], [1125, 443], [1090, 443]], label: "NVFP4", labelAt: [1148, 420], labelAnchor: "start" },
      { id: "wg2opt", p: [[880, 460], [605, 460], [605, 440]], label: "BF16", labelAt: [750, 450] },
      { id: "g2qsr", p: [[1030, 36], [730, 36], [730, 140], [350, 140], [350, 430]], label: "BF16", labelAt: [548, 132] },
      { id: "qsr2d", p: [[350, 494], [350, 540]], label: "NVFP4", labelAt: [375, 520], labelAnchor: "start" },
      { id: "d2prev", p: [[250, 578], [100, 578]], label: "BF16 → layer i−1", labelAt: [160, 566] },
    ],
    flows: [
      { path: [{ edge: "inQ", becomes: "q" }, { edge: "qF" }], start: "hp" },
      { path: [{ edge: "fOut2" }], start: "hp", count: 1 },
      { path: [{ edge: "w2q", becomes: "q" }, { edge: "q2f" }], start: "hp", count: 1 },
      { path: [{ edge: "q2t" }, { edge: "t2d" }], start: "q", count: 2 },
      { path: [{ edge: "g2t" }, { edge: "t2h2" }, { edge: "h2q2", becomes: "q" }, { edge: "q2wg2" }], start: "grad" },
      { path: [{ edge: "actTap" }, { edge: "t2h1" }, { edge: "h2q1", becomes: "q" }, { edge: "q2wg1" }], start: "hp" },
      { path: [{ edge: "wg2opt", becomes: "opt" }, { edge: "opt2w" }], start: "grad", count: 2 },
      { path: [{ edge: "g2qsr", becomes: "q" }, { edge: "qsr2d" }], start: "grad", count: 3 },
      { path: [{ edge: "d2prev" }], start: "grad", count: 1 },
    ],
    legend: [
      { color: "#d5e8d4", label: "NVFP4 GEMM" },
      { color: "#ffe6cc", label: "quantize / transform" },
      { color: "#f5f5f5", label: "BF16" },
      { color: "#ffffff", label: "FP32 / plain" },
    ],
    caption: "The pretraining recipe quantizes every GEMM operand — including both weight-gradient inputs after Hadamard transforms, and gradients with stochastic rounding.",
  }));

  /* =========================================================
   * TransformerEngine MXFP8 recipe (row-wise / column-wise copies)
   * ========================================================= */

  NV.registerDiagram("recipe-mxfp8-te", (narrow) => teSpec(narrow));

  function teSpec(narrow) {
    const spec = ({
    w: 1180, h: 680,
    title: "TransformerEngine MXFP8 Recipe",
    subtitle: "row-wise and column-wise quantized copies for forward and backward GEMMs",
    nodes: {
      input: { x: 40, y: 220, w: 220, h: 70, kind: "bf16", title: "BF16 activation", info: "Layer input in BF16." },
      weight: { x: 40, y: 374, w: 220, h: 70, kind: "bf16", title: "BF16 weight", info: "BF16 weight, re-quantized each step." },
      colIn: { x: 315, y: 28, w: 245, h: 94, kind: "colcopy", title: "MXFP8 activation", sub: ["col-wise, 1x32 block"], info: "Column-scaled activation copy kept for Wgrad." },
      rowIn: { x: 315, y: 150, w: 245, h: 94, kind: "rowcopy", title: "MXFP8 activation", sub: ["row-wise, 1x32 block"], info: "Row-scaled FP8 activation copy consumed by Fprop." },
      fprop: { x: 305, y: 286, w: 265, h: 96, kind: "fprop", title: "Fprop", sub: ["MXFP8 tensor core, FP32 acc"], info: "Forward GEMM on FP8 tensor cores." },
      rowW: { x: 315, y: 430, w: 245, h: 94, kind: "rowcopy", title: "MXFP8 weight", sub: ["row-wise, 1x32 block"], info: "Row-scaled FP8 weight copy consumed by Fprop and rollout." },
      colW: { x: 315, y: 552, w: 245, h: 94, kind: "colcopy", title: "MXFP8 weight", sub: ["col-wise, 1x32 block"], info: "Column-scaled weight copy kept for Dgrad." },
      wgrad: { x: 620, y: 28, w: 265, h: 94, kind: "gemm", title: "Wgrad", sub: ["BF16 tensor core, FP32 acc"], info: "Weight-gradient GEMM producing the FP32 weight gradient." },
      opt: { x: 632, y: 296, w: 250, h: 94, kind: "opt", title: "Optimizer states", sub: ["FP32 master weight"], info: "FP32 optimizer states and master weights update the BF16 weight." },
      dgrad: { x: 620, y: 552, w: 265, h: 94, kind: "gemm", title: "Dgrad", sub: ["BF16 tensor core, FP32 acc"], info: "Data-gradient GEMM; result is passed to the previous layer." },
      outGrad: { x: 920, y: 388, w: 210, h: 70, kind: "bf16", title: "BF16 gradient", info: "Gradient arriving from layer i+1." },
      colG: { x: 908, y: 274, w: 245, h: 82, kind: "colcopy", title: "MXFP8 gradient", sub: ["col-wise, 1x32 block"], info: "Column-scaled FP8 gradient copy consumed by Wgrad." },
      rowG: { x: 908, y: 494, w: 245, h: 82, kind: "rowcopy", title: "MXFP8 gradient", sub: ["row-wise, 1x32 block"], info: "Row-scaled FP8 gradient copy consumed by Dgrad." },
    },
    edges: [
      { id: "in2col", p: [[150, 220], [150, 75], [315, 75]], label: "quantize", labelAt: [208, 63] },
      { id: "in2row", p: [[150, 255], [150, 197], [315, 197]], label: "quantize", labelAt: [208, 185] },
      { id: "rowIn2f", p: [[438, 244], [438, 286]] },
      { id: "colIn2wg", p: [[560, 75], [620, 75]] },
      { id: "w2row", p: [[150, 444], [150, 477], [315, 477]], label: "quantize", labelAt: [208, 466] },
      { id: "w2col", p: [[150, 444], [150, 599], [315, 599]], label: "quantize", labelAt: [208, 588] },
      { id: "rowW2f", p: [[438, 430], [438, 382]] },
      { id: "colW2d", p: [[560, 599], [620, 599]] },
      { id: "grad2col", p: [[1025, 388], [1025, 356]], label: "quantize", labelAt: [1040, 374], labelAnchor: "start" },
      { id: "grad2row", p: [[1025, 458], [1025, 494]], label: "quantize", labelAt: [1040, 478], labelAnchor: "start" },
      { id: "colG2wg", p: [[1025, 274], [1025, 75], [885, 75]] },
      { id: "rowG2d", p: [[1025, 576], [1025, 599], [885, 599]] },
      { id: "wg2opt", p: [[752, 122], [752, 296]] },
      { id: "opt2w", p: [[757, 390], [757, 409], [260, 409]], dashed: true },
      { id: "f2next", p: [[305, 334], [150, 334], [150, 290]], dashed: true, label: "to next layer", labelAt: [210, 322] },
      { id: "wRollout", p: [[560, 477], [690, 477]], dashed: true, label: "to rollout", labelAt: [625, 464] },
      { id: "d2prev", p: [[752, 552], [752, 423], [920, 423]], dashed: true, label: "to prev layer", labelAt: [835, 411] },
    ],
    flows: [
      { path: [{ edge: "in2row", becomes: "q" }, { edge: "rowIn2f" }], start: "hp", count: 2 },
      { path: [{ edge: "f2next" }], start: "hp", count: 1 },
      { path: [{ edge: "w2row", becomes: "q" }, { edge: "rowW2f" }], start: "hp", count: 1 },
      { path: [{ edge: "w2col", becomes: "q" }, { edge: "colW2d" }], start: "hp", count: 2 },
      { path: [{ edge: "in2col", becomes: "q" }, { edge: "colIn2wg" }], start: "hp", count: 3 },
      { path: [{ edge: "grad2row", becomes: "q" }, { edge: "rowG2d" }], start: "grad", count: 1 },
      { path: [{ edge: "grad2col", becomes: "q" }, { edge: "colG2wg" }], start: "grad", count: 2 },
      { path: [{ edge: "d2prev" }], start: "grad", count: 1 },
      { path: [{ edge: "wg2opt", becomes: "opt" }], start: "grad", count: 1 },
      { path: [{ edge: "opt2w" }], start: "opt", count: 3 },
    ],
    legend: [
      { color: "#dceafd", label: "MXFP8 row-wise copy" },
      { color: "#eadff0", label: "MXFP8 column-wise copy" },
      { color: "#e6f1e2", label: "Fprop" },
      { color: "#fff2cc", label: "backward GEMM" },
      { color: "#ffe6cc", label: "optimizer states" },
      { color: "#f5f5f5", label: "BF16 tensor" },
    ],
    caption: "TransformerEngine's MXFP8 path keeps row-wise and column-wise copies for activation, weight, and incoming gradient so Fprop, Wgrad, and Dgrad can consume the layout they need.",
    });
    if (narrow) {
      const g = mxfp8TeNarrowGeometry();
      for (const [id, geo] of Object.entries(g.nodes)) Object.assign(spec.nodes[id], geo);
      spec.edges = g.edges;
      spec.w = g.w;
      spec.h = g.h;
    }
    return spec;
  }

  /* =========================================================
   * 4/6 error staircase (map-to-6 vs map-to-4)
   * ========================================================= */

  NV.registry["fourover6-staircase"] = function (fig) {
    const body = card(
      fig,
      "Four-over-six: two candidate block scalings",
      "quantization error over the block's value range",
      "drag the slider or hover over the plots"
    );

    const G6 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
    const G4 = [0, 0.5, 1, 1.5, 2, 3, 4];
    /* candidate identity colors: map-to-6 teal, map-to-4 rust; deeper fill = larger gap error */
    const TEAL = "#176b64";
    const RUST = "#a33e2d";
    const COLORS6 = { 0.5: "hsl(172 28% 90%)", 1: "hsl(172 32% 76%)", 2: "hsl(172 36% 58%)" };
    const COLORS4 = { 0.5: "hsl(9 42% 91%)", 1: "hsl(9 50% 74%)" };

    const REDUCE = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const MORPH = REDUCE ? "" : "d 0.45s ease";
    const MOVE_REF = REDUCE ? "" : "y1 0.45s ease, y2 0.45s ease";

    const state = { metric: "mae", frac: 0.83 };
    /* display transform: MAE shows |err|, MSE shows err² — argmin per value is identical,
     * so shapes morph between the two while the winner map stays fixed */
    const tf = (e) => (state.metric === "mse" ? e * e : e);
    const fmtErr = (eAbs) => state.metric === "mse"
      ? "sq err " + (eAbs * eAbs * 100).toFixed(2) + "% of max²"
      : "err " + (eAbs * 100).toFixed(1) + "% of max";

    const row = el("div", { class: "viz-row" });
    body.appendChild(row);
    const envWrap = el("div", {});
    body.appendChild(envWrap);

    const readout = el("div", {
      class: "viz-sub",
      style: "text-align:center;margin-top:8px;font-size:13px",
    });

    function snap(grid, v) {
      let best = grid[0];
      for (const g of grid) if (Math.abs(g - v) < Math.abs(best - v)) best = g;
      return best;
    }

    const panels = [];
    const metricized = []; /* everything with a metric-dependent shape: {update()} */

    function makePanel(titleTxt, grid, colors, yLabelsFor, accent) {
      const bmax = grid[grid.length - 1];
      const W = 460;
      const H = 260;
      const M = { l: 58, r: 16, t: 34, b: 44 };
      const sx = (v) => M.l + (v / bmax) * (W - M.l - M.r);
      /* y maps ABS error through the current metric; headroom above the 1/6 line */
      const sy = (eAbs) => H - M.b - (tf(eAbs) / tf(1 / 5.2)) * (H - M.t - M.b);

      const cell = el("div", {});
      row.appendChild(cell);
      const wrap = el("div", { class: "viz-plot" });
      cell.appendChild(wrap);
      const s = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}` });
      wrap.appendChild(s);

      s.appendChild(
        svg("text", {
          x: M.l, y: 18, "font-size": 11.5, "font-weight": 600,
          "letter-spacing": "0.08em", fill: accent, text: titleTxt.toUpperCase(),
        })
      );
      /* per-gap error shape: triangles under MAE morph into parabolic arcs under MSE */
      const SAMPLES = 22;
      function gapD(a, b) {
        let d = `M${sx(a).toFixed(1)} ${sy(0).toFixed(1)}`;
        for (let k = 1; k < SAMPLES; k++) {
          const v = a + ((b - a) * k) / SAMPLES;
          d += `L${sx(v).toFixed(1)} ${sy(Math.min(v - a, b - v) / bmax).toFixed(1)}`;
        }
        return d + `L${sx(b).toFixed(1)} ${sy(0).toFixed(1)}Z`;
      }
      for (let i = 0; i < grid.length - 1; i++) {
        const a = grid[i];
        const b = grid[i + 1];
        const gap = +(b - a).toFixed(3);
        const shape = svg("path", {
          d: gapD(a, b),
          fill: colors[gap] || "#ddd",
          stroke: accent,
          "stroke-width": 1,
          "stroke-opacity": 0.55,
          "stroke-linejoin": "round",
        });
        shape.style.transition = MORPH;
        s.appendChild(shape);
        metricized.push({ update: () => shape.setAttribute("d", gapD(a, b)) });
      }
      /* dashed reference at map-to-6's worst case */
      const ref = svg("line", {
        x1: M.l, x2: W - M.r, y1: sy(1 / 6), y2: sy(1 / 6),
        stroke: "#a33e2d", "stroke-width": 1, "stroke-dasharray": "6 5", "stroke-opacity": 0.7,
      });
      ref.style.transition = MOVE_REF;
      s.appendChild(ref);
      metricized.push({ update: () => { ref.setAttribute("y1", sy(1 / 6)); ref.setAttribute("y2", sy(1 / 6)); } });

      s.appendChild(
        svg("line", {
          x1: M.l, x2: W - M.r, y1: sy(0), y2: sy(0),
          stroke: "#c4bcae", "stroke-width": 1, "shape-rendering": "crispEdges",
        })
      );
      for (const g of grid) {
        s.appendChild(
          svg("text", {
            x: sx(g), y: H - M.b + 17, "text-anchor": "middle",
            "font-size": 11, fill: "#8a847a", text: String(g),
          })
        );
      }
      const yLabelsG = svg("g", {});
      s.appendChild(yLabelsG);
      function drawYLabels() {
        yLabelsG.textContent = "";
        for (const [eAbs, label] of yLabelsFor(state.metric)) {
          yLabelsG.appendChild(
            svg("text", {
              x: M.l - 7, y: sy(eAbs) + 4, "text-anchor": "end",
              "font-size": 11, fill: "#8a847a", text: label,
            })
          );
          yLabelsG.appendChild(
            svg("line", { x1: M.l - 3, x2: M.l + 3, y1: sy(eAbs), y2: sy(eAbs), stroke: "#c4bcae", "stroke-width": 1 })
          );
        }
      }
      drawYLabels();
      metricized.push({ update: drawYLabels });

      s.appendChild(
        svg("text", {
          x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle",
          "font-size": 10.5, "font-weight": 600, "letter-spacing": "0.06em", fill: "#8a847a",
          text: ("value within the block · block max ↦ " + bmax).toUpperCase(),
        })
      );

      const marker = svg("g", {});
      s.appendChild(marker);

      panels.push({
        setValue(frac) {
          const v = frac * bmax;
          const q = snap(grid, v);
          const err = Math.abs(v - q) / bmax;
          marker.textContent = "";
          marker.appendChild(
            svg("line", { x1: sx(v), x2: sx(v), y1: sy(0), y2: H - M.b - (H - M.t - M.b) * 0.98, stroke: accent, "stroke-width": 1.2, "stroke-dasharray": "3 3" })
          );
          marker.appendChild(
            svg("circle", { cx: sx(v), cy: sy(err), r: 5.5, fill: accent, stroke: "#fffdf8", "stroke-width": 1.5 })
          );
          marker.appendChild(
            svg("text", {
              x: sx(v), y: sy(err) - 12, "text-anchor": "middle",
              "font-size": 11.5, "font-weight": 600, fill: accent,
              text: (q < v ? "← " : q > v ? "→ " : "= ") + q + "  (" + fmtErr(err) + ")",
            })
          );
          return err;
        },
      });
      s.addEventListener("pointermove", (evt) => {
        const rect = s.getBoundingClientRect();
        const x = ((evt.clientX - rect.left) / rect.width) * W;
        const frac = Math.max(0, Math.min(1, (x - M.l) / (W - M.l - M.r)));
        setAll(frac);
      });
    }

    function buildEnvelope() {
      const NARROW3 = (fig.clientWidth || 700) < 520;
      const W3 = NARROW3 ? 520 : 952;
      const H3 = NARROW3 ? 250 : 218;
      const M3 = { l: NARROW3 ? 48 : 58, r: 16, t: NARROW3 ? 40 : 34, b: 44 };
      const F6 = G6.map((g) => g / 6);
      const F4 = G4.map((g) => g / 4);
      const gridErr = (grid, f) => Math.min(...grid.map((g) => Math.abs(f - g)));
      const fx = (f) => M3.l + f * (W3 - M3.l - M3.r);
      const fy = (eAbs) => H3 - M3.b - (tf(eAbs) / tf(1 / 5.2)) * (H3 - M3.t - M3.b);

      const wrap3 = el("div", { class: "viz-plot", style: "margin-top:6px" });
      envWrap.appendChild(wrap3);
      const s3 = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W3} ${H3}` });
      wrap3.appendChild(s3);

      const title3 = svg("text", {
        x: M3.l, y: 18, "font-size": 11.5, "font-weight": 600,
        "letter-spacing": "0.08em", fill: "#6d6860",
      });
      s3.appendChild(title3);
      function drawTitle() {
        const w6 = state.metric === "mse" ? "(1/6)²" : "1/6";
        const w8 = state.metric === "mse" ? "(1/8)²" : "1/8";
        title3.textContent = `4/6 = THE LOWER OF THE TWO — WORST CASE FALLS FROM ${w6} TO ${w8}`;
      }
      drawTitle();
      metricized.push({ update: drawTitle });

      /* sample once in ABS error space; segmentation by winner is metric-independent */
      const N = 960;
      const pts6 = [];
      const pts4 = [];
      const segs = [];
      let cur = null;
      let prev = null;
      for (let i = 0; i <= N; i++) {
        const f = i / N;
        const e6 = gridErr(F6, f);
        const e4 = gridErr(F4, f);
        pts6.push([f, e6]);
        pts4.push([f, e4]);
        const winner = e4 < e6 - 1e-12 ? 4 : e6 < e4 - 1e-12 ? 6 : 0; /* 0 = exact tie */
        if (!cur || winner !== cur.winner) {
          if (cur && prev) {
            const t = (prev.e6 - prev.e4) / (prev.e6 - prev.e4 - (e6 - e4)) || 0;
            const fc = prev.f + (f - prev.f) * Math.max(0, Math.min(1, t));
            const ec = Math.min(gridErr(F6, fc), gridErr(F4, fc));
            cur.pts.push([fc, ec]);
            segs.push(cur);
            cur = { winner, pts: [[fc, ec]] };
          } else {
            if (cur) segs.push(cur);
            cur = { winner, pts: [] };
          }
        }
        cur.pts.push([f, Math.min(e6, e4)]);
        prev = { f, e6, e4 };
      }
      segs.push(cur);
      /* absorb hairline crossing "ties" into the preceding segment */
      const drawSegs = [];
      for (const seg of segs) {
        const extent = seg.pts.length ? seg.pts[seg.pts.length - 1][0] - seg.pts[0][0] : 0;
        if (seg.winner === 0 && extent < 3 / N && drawSegs.length) {
          drawSegs[drawSegs.length - 1].pts.push(...seg.pts);
        } else {
          drawSegs.push(seg);
        }
      }
      function segD(seg) {
        const x0 = fx(seg.pts[0][0]);
        const x1 = fx(seg.pts[seg.pts.length - 1][0]);
        let d = `M${x0.toFixed(1)} ${fy(0).toFixed(1)}`;
        for (const [f, e] of seg.pts) d += `L${fx(f).toFixed(1)} ${fy(e).toFixed(1)}`;
        return d + `L${x1.toFixed(1)} ${fy(0).toFixed(1)}Z`;
      }
      for (const seg of drawSegs) {
        if (seg.pts.length < 2) continue;
        const path = svg("path", {
          d: segD(seg),
          fill: seg.winner === 4 ? "hsl(9 48% 88%)" : seg.winner === 6 ? "hsl(172 30% 87%)" : "hsl(40 12% 89%)",
          stroke: seg.winner === 4 ? RUST : seg.winner === 6 ? TEAL : "#8a847a",
          "stroke-width": 1.4,
          "stroke-linejoin": "round",
        });
        path.style.transition = MORPH;
        s3.appendChild(path);
        metricized.push({ update: () => path.setAttribute("d", segD(seg)) });
      }
      function sawD(pts) {
        let d = "";
        for (let i = 0; i < pts.length; i++) {
          d += (i ? "L" : "M") + fx(pts[i][0]).toFixed(1) + " " + fy(pts[i][1]).toFixed(1);
        }
        return d;
      }
      const saw6 = svg("path", { d: sawD(pts6), fill: "none", stroke: TEAL, "stroke-width": 1, "stroke-dasharray": "3 3", "stroke-opacity": 0.55 });
      const saw4 = svg("path", { d: sawD(pts4), fill: "none", stroke: RUST, "stroke-width": 1, "stroke-dasharray": "6 3", "stroke-opacity": 0.55 });
      saw6.style.transition = MORPH;
      saw4.style.transition = MORPH;
      s3.appendChild(saw6);
      s3.appendChild(saw4);
      metricized.push({ update: () => { saw6.setAttribute("d", sawD(pts6)); saw4.setAttribute("d", sawD(pts4)); } });

      const legend = svg("text", { x: W3 - M3.r, y: 18, "text-anchor": "end", "font-size": 10.5, fill: "#8a847a" });
      legend.appendChild(svg("tspan", { text: "filled minimum, colored by the winner: " }));
      legend.appendChild(svg("tspan", { fill: TEAL, "font-weight": 600, text: "map-to-6" }));
      legend.appendChild(svg("tspan", { text: " · " }));
      legend.appendChild(svg("tspan", { fill: RUST, "font-weight": 600, text: "map-to-4" }));
      legend.appendChild(svg("tspan", { text: " · " }));
      legend.appendChild(svg("tspan", { fill: "#8a847a", "font-weight": 600, text: "tie" }));
      s3.appendChild(legend);

      const ref3 = svg("line", {
        x1: M3.l, x2: W3 - M3.r, y1: fy(1 / 6), y2: fy(1 / 6),
        stroke: "#a33e2d", "stroke-width": 1, "stroke-dasharray": "6 5", "stroke-opacity": 0.7,
      });
      ref3.style.transition = MOVE_REF;
      s3.appendChild(ref3);
      metricized.push({ update: () => { ref3.setAttribute("y1", fy(1 / 6)); ref3.setAttribute("y2", fy(1 / 6)); } });

      s3.appendChild(svg("line", { x1: M3.l, x2: W3 - M3.r, y1: fy(0), y2: fy(0), stroke: "#c4bcae", "stroke-width": 1, "shape-rendering": "crispEdges" }));
      const yLabels3G = svg("g", {});
      s3.appendChild(yLabels3G);
      function drawYLabels3() {
        yLabels3G.textContent = "";
        const labels = state.metric === "mse"
          ? [[1 / 6, "(1/6)²"], [1 / 8, "(1/8)²"], [1 / 16, "(1/16)²"]]
          : [[1 / 6, "1/6"], [1 / 8, "1/8"], [1 / 16, "1/16"]];
        for (const [eAbs, label] of labels) {
          yLabels3G.appendChild(svg("text", { x: M3.l - 7, y: fy(eAbs) + 4, "text-anchor": "end", "font-size": 11, fill: "#8a847a", text: label }));
          yLabels3G.appendChild(svg("line", { x1: M3.l - 3, x2: M3.l + 3, y1: fy(eAbs), y2: fy(eAbs), stroke: "#c4bcae", "stroke-width": 1 }));
        }
      }
      drawYLabels3();
      metricized.push({ update: drawYLabels3 });

      for (const t of [0, 0.25, 0.5, 0.75, 1]) {
        s3.appendChild(svg("text", { x: fx(t), y: H3 - M3.b + 17, "text-anchor": "middle", "font-size": 11, fill: "#8a847a", text: String(t) }));
      }
      s3.appendChild(svg("text", {
        x: (M3.l + W3 - M3.r) / 2, y: H3 - 6, "text-anchor": "middle",
        "font-size": 10.5, "font-weight": 600, "letter-spacing": "0.06em", fill: "#8a847a",
        text: "VALUE / BLOCK MAX · BEST OF THE TWO CANDIDATES",
      }));

      const marker3 = svg("g", {});
      s3.appendChild(marker3);
      panels.push({
        setValue(frac) {
          const e6 = gridErr(F6, frac);
          const e4 = gridErr(F4, frac);
          const e = Math.min(e6, e4);
          const winColor = e4 < e6 ? RUST : e6 < e4 ? TEAL : "#6d6860";
          marker3.textContent = "";
          marker3.appendChild(svg("line", { x1: fx(frac), x2: fx(frac), y1: fy(0), y2: H3 - M3.b - (H3 - M3.t - M3.b) * 0.98, stroke: winColor, "stroke-width": 1.2, "stroke-dasharray": "3 3" }));
          marker3.appendChild(svg("circle", { cx: fx(frac), cy: fy(e), r: 5.5, fill: winColor, stroke: "#fffdf8", "stroke-width": 1.5 }));
          marker3.appendChild(svg("text", {
            x: fx(frac), y: fy(e) - 12, "text-anchor": "middle",
            "font-size": 11.5, "font-weight": 600, fill: winColor,
            text: (e4 < e6 ? "via map-to-4" : e6 < e4 ? "via map-to-6" : "tie") + "  (" + fmtErr(e) + ")",
          }));
          return e;
        },
      });
      s3.addEventListener("pointermove", (evt) => {
        const rect = s3.getBoundingClientRect();
        const x = ((evt.clientX - rect.left) / rect.width) * W3;
        const frac = Math.max(0, Math.min(1, (x - M3.l) / (W3 - M3.l - M3.r)));
        setAll(frac);
      });
    }

    makePanel("With largest value scaled to 6", G6, COLORS6,
      (m) => m === "mse"
        ? [[1 / 6, "(1/6)²"], [1 / 12, "(1/12)²"]]
        : [[1 / 6, "1/6"], [1 / 12, "1/12"], [1 / 24, "1/24"]],
      TEAL);
    makePanel("With largest value scaled to 4", G4, COLORS4,
      (m) => m === "mse"
        ? [[1 / 6, "(1/6)²"], [1 / 8, "(1/8)²"]]
        : [[1 / 6, "1/6"], [1 / 8, "1/8"], [1 / 16, "1/16"]],
      RUST);
    buildEnvelope();

    const slider = el("input", {
      type: "range", min: "0", max: "1000", value: "830",
      style: "width:min(360px, 80%)",
      oninput: () => setAll(Number(slider.value) / 1000),
    });
    const metricBtns = {};
    const metricWrap = el("div", { style: "display:flex;gap:6px;align-items:center" });
    for (const m of ["mae", "mse"]) {
      const b = el("button", {
        class: "viz-btn", text: m.toUpperCase(),
        "aria-pressed": String(state.metric === m),
        onclick: () => {
          if (state.metric === m) return;
          state.metric = m;
          for (const [k, btn] of Object.entries(metricBtns)) btn.setAttribute("aria-pressed", String(k === m));
          metricized.forEach((mm) => mm.update());
          setAll(state.frac);
        },
      });
      metricBtns[m] = b;
      metricWrap.appendChild(b);
    }
    const sliderWrap = el(
      "div",
      { class: "viz-controls", style: "justify-content:center;gap:22px" },
      metricWrap,
      el("label", {}, el("span", { text: "value / block max" }), slider)
    );
    body.appendChild(sliderWrap);
    body.appendChild(readout);
    body.appendChild(
      el("figcaption", {
        text: "The same value is quantized under both block scalings; 4/6 computes both candidates' reconstruction error and keeps the better one per block. The bottom panel is the per-value minimum of the two — the envelope 4/6 selects from (per block, so all 16 values share one winner): rust stretches are where map-to-4's finer grid wins, teal where map-to-6's points sit closer (notably around 2/3 of max), and gray where the two tie exactly — bands around the shared grid points 0, 1/4, 1/2, and the block max. Switching MAE/MSE morphs the curves but not the winner map — squaring is monotone, so per value the two metrics always agree; they can differ only when errors are summed over a block's 16 values, which is why the fast-path agreement table reports both.",
      })
    );

    function setAll(frac) {
      state.frac = frac;
      slider.value = String(Math.round(frac * 1000));
      const e6 = panels[0].setValue(frac);
      const e4 = panels[1].setValue(frac);
      panels[2].setValue(frac);
      const pct = (v) => state.metric === "mse" ? (v * v * 100).toFixed(2) + "%" : (v * 100).toFixed(1) + "%";
      let verdict;
      if (Math.abs(e6 - e4) < 1e-9) verdict = "both candidates tie here";
      else if (e4 < e6) verdict = '<b style="color:#a33e2d">map-to-4</b> wins here (' + pct(e4) + " vs " + pct(e6) + ")";
      else verdict = '<b style="color:#176b64">map-to-6</b> wins here (' + pct(e6) + " vs " + pct(e4) + ")";
      const w6 = state.metric === "mse" ? "(1/6)²" : "1/6";
      const w8 = state.metric === "mse" ? "(1/8)²" : "1/8";
      readout.innerHTML =
        "value = <b>" + (frac).toFixed(2) + " × block max</b> → " + verdict +
        " · worst case: map-to-6 <b>" + w6 + "</b>, map-to-4 <b>" + w8 + "</b>, best-of-both <b>" + w8 + "</b>";
    }
    setAll(state.frac);
  };

  /* =========================================================
   * NCU source profile: the capture cropped to its two panes, a
   * selectable text layer locked over the bitmap, and each sampled
   * line's stall breakdown pinned inside the empty pane on hover
   * ========================================================= */

  NV.registry["ncu-fp32-math"] = function (fig) {
    /* the md fallback img already carries the rewritten asset path — reuse its base */
    const fallback = fig.querySelector("img");
    const base = fallback ? fallback.src.replace(/[^/]*$/, "") : "images/ncu/";
    const body = card(
      fig,
      "Profiling the GPU kernel: the canonical 4/6 error path is dequantization compute bound",
      "Nsight Compute capture of quantize_4over6_kernel — warp stall sampling by source line",
      "hover over a sampled line · source text is selectable"
    );

    /* capture geometry, in original crop pixels: lines 194-226 on a fixed pitch */
    const CW = 3356;
    const CH = 1504;
    /* overlay baselines sit BELOW the bitmap's (the selection cell extends one
       ascent above them) so the visible selection box centers on each row */
    const BASE_Y = 102;
    const PITCH = 42.1;
    const NUM_X = 178;
    const CODE_X = 198;
    const PCT_X = 1904;
    const ADV = 15.5; /* the capture's mono advance per character, crop pixels */
    const yOf = (n) => BASE_Y + (n - 194) * PITCH;

    const pad57 = " ".repeat(57);
    const pad66 = " ".repeat(66);
    const SRC = {
      194: "",
      195: "template <typename Cfg, int E4M3_MAX, int SHIFT>",
      196: "__device__ __forceinline__ void accumulate_dequant_error(const uint32_t dequant_bits, const float x,",
      197: pad57 + "const float sf, const float global_amax,",
      198: pad57 + "float *err) {",
      199: "  constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f",
      200: "  constexpr float fp8_max = static_cast<float>(E4M3_MAX);",
      201: "  constexpr float err_denom = fp4_max * fp8_max;",
      202: "  const uint16_t half_bits = (dequant_bits >> SHIFT) & 0xFFFF;",
      203: "",
      204: "  if constexpr (Cfg::err_use_fast_math) {",
      205: "    const float dequant = __half2float(__ushort_as_half(half_bits));",
      206: "    const float val = dequant * sf * global_amax / err_denom;",
      207: "    const float diff = val - x;",
      208: "    *err += compute_error<Cfg::err_mode>(diff);",
      209: "  } else {",
      210: "    const float dequant = __half2float(__ushort_as_half(half_bits));",
      211: "    const float val = __fdiv_rn(__fmul_rn(__fmul_rn(dequant, sf), global_amax), err_denom);",
      212: "    const float diff = __fsub_rn(val, x);",
      213: "    *err = __fadd_rn(*err, compute_error_rn<Cfg::err_mode>(diff));",
      214: "  }",
      215: "}",
      216: "",
      217: "template <typename Cfg, int E4M3_MAX>",
      218: "__device__ __forceinline__ uint32_t cvt_fp32_to_fp4_8x_with_error(const float (&x)[8],",
      219: pad66 + "const float block_scale_inverse,",
      220: pad66 + "const nvfp4_scale_t sf,",
      221: pad66 + "const float global_amax,",
      222: pad66 + "float *err) {",
      223: "  uint32_t out = 0;",
      224: "  uint32_t out_dequant_1 = 0;",
      225: "  uint32_t out_dequant_2 = 0;",
      226: "  uint32_t out_dequant_3 = 0;",
    };
    const STALLS = { 210: "3.63%", 211: "19.12%", 212: "0.23%", 213: "0.32%" };

    /* wrap crops on phones (the inner strip widens so the source pane fills
       the column); the stall card docks below the strip there */
    const wrap = el("div", { class: "ncu-wrap" });
    const inner = el("div", { class: "ncu-inner" });
    inner.appendChild(el("img", {
      src: base + "ncu-crop-web.webp",
      alt: "Nsight Compute source view of quantize_4over6_kernel: source and warp stall sampling panes",
      style: "width:100%;display:block;border-radius:6px",
      loading: "lazy",
      decoding: "async",
    }));
    wrap.appendChild(inner);

    /* transparent text layer: selection targets locked to the bitmap's coordinates */
    const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";
    const layer = svg("svg", {
      viewBox: `0 0 ${CW} ${CH}`,
      style: "position:absolute;inset:0;width:100%;height:100%;user-select:text",
    });
    function txt(x, y, s, anchor, fit) {
      const t = svg("text", {
        x, y, "font-size": 33, "font-family": MONO,
        style: "fill:transparent;cursor:text;white-space:pre", "xml:space": "preserve",
      });
      if (anchor) t.setAttribute("text-anchor", anchor);
      /* lock the run to the capture's character grid so selection tracks the bitmap */
      if (fit && s.length) {
        t.setAttribute("textLength", (s.length * ADV).toFixed(1));
        t.setAttribute("lengthAdjust", "spacingAndGlyphs");
      }
      t.textContent = s;
      return t;
    }
    layer.appendChild(txt(168, 27, "# Source"));
    layer.appendChild(txt(2173, 27, "Warp Stall Sampling", "middle"));
    layer.appendChild(txt(2173, 62, "(All Samples)", "middle"));

    /* the in-capture detail: caption chip + that line's captured stall table,
       pinned over the empty region of the sampling pane. Each card carries its
       transcription so the table is selectable; widths are per-card so the
       narrower capture (line 212) is not stretched */
    const CARDS = {
      210: { w: 840, h: 488, total: 2133, rows: [
        ["Branch Resolving", 1175, "55.09%"], ["Wait", 594, "27.85%"], ["Selected", 165, "7.74%"],
        ["Not Selected", 142, "6.66%"], ["Math Pipe Throttle", 42, "1.97%"], ["No Instructions", 11, "0.52%"],
        ["Dispatch Stall", 4, "0.19%"],
      ] },
      211: { w: 840, h: 568, total: 11251, rows: [
        ["Not Selected", 3519, "31.28%"], ["Wait", 2838, "25.22%"], ["No Instructions", 1550, "13.78%"],
        ["Selected", 1443, "12.83%"], ["Branch Resolving", 742, "6.59%"], ["Math Pipe Throttle", 596, "5.30%"],
        ["Short Scoreboard", 409, "3.64%"], ["Dispatch Stall", 133, "1.18%"], ["Mio Throttle", 21, "0.19%"],
      ] },
      212: { w: 804, h: 408, total: 135, rows: [
        ["Selected", 98, "72.59%"], ["Wait", 16, "11.85%"], ["No Instructions", 10, "7.41%"],
        ["Not Selected", 7, "5.19%"], ["Dispatch Stall", 4, "2.96%"],
      ] },
      213: { w: 840, h: 448, total: 189, rows: [
        ["Selected", 80, "42.33%"], ["Wait", 57, "30.16%"], ["Not Selected", 36, "19.05%"],
        ["No Instructions", 13, "6.88%"], ["Math Pipe Throttle", 2, "1.06%"], ["Dispatch Stall", 1, "0.53%"],
      ] },
    };
    const detail = el("div", { class: "ncu-card" });
    const detailCap = el("div", {
      style: "background:rgba(15,17,20,0.96);color:#d9dee5;" +
        "font-size:11.5px;padding:5px 10px;border-radius:8px 8px 0 0;" +
        "border:1px solid rgba(255,255,255,0.18);border-bottom:none",
    });
    /* the capture has its own bright border baked in — crop it off and draw a
       fainter white hairline instead */
    const detailBody = el("div", {
      style: "position:relative;overflow:hidden;border-radius:0 0 9px 9px;" +
        "border:1px solid rgba(255,255,255,0.18);border-top:none;background:#16181b",
    });
    const detailImg = el("img", {
      alt: "warp stall breakdown for the hovered source line",
      loading: "lazy",
      decoding: "async",
      style: "display:block;width:101%;margin:-0.5%",
    });
    const detailSvg = svg("svg", {
      style: "position:absolute;inset:0;width:100%;height:100%;user-select:text",
    });
    detailBody.appendChild(detailImg);
    detailBody.appendChild(detailSvg);
    detail.appendChild(detailCap);
    detail.appendChild(detailBody);

    const hlRects = {};
    function show(n) {
      const tbl = CARDS[n];
      /* custom props, not inline width: the mobile media query picks its own */
      detail.style.setProperty("--cardw", ((28.5 * tbl.w) / 840).toFixed(2) + "%");
      detail.style.setProperty("--cardwm", ((100 * tbl.w) / 840).toFixed(2) + "%");
      detailImg.src = base + "ncu-row-" + n + ".webp";
      detailCap.innerHTML = `line <b>${n}</b> · <b>${STALLS[n]}</b> of warp stall samples`;
      /* rebuild the card's selectable text layer from its transcription */
      detailSvg.setAttribute("viewBox", `0 0 ${tbl.w} ${tbl.h}`);
      while (detailSvg.firstChild) detailSvg.removeChild(detailSvg.firstChild);
      const cols = tbl.w === 804
        ? { val: 335, pctR: 770, hMetric: 195, hVal: 385, hPct: 612 }
        : { val: 371, pctR: 805, hMetric: 213, hVal: 421, hPct: 647 };
      const cardTxt = (x, y, s2, anchor) => {
        const t = svg("text", {
          x, y, "font-size": 27, "font-family": "system-ui, -apple-system, 'Segoe UI', sans-serif",
          style: "fill:transparent;cursor:text;white-space:pre", "xml:space": "preserve",
        });
        if (anchor) t.setAttribute("text-anchor", anchor);
        t.textContent = s2;
        detailSvg.appendChild(t);
      };
      cardTxt(24, 54, "Total Sample Count: " + tbl.total);
      cardTxt(cols.hMetric, 111, "Metric", "middle");
      cardTxt(cols.hVal, 111, "Value", "middle");
      cardTxt(cols.hPct, 111, "Percentage", "middle");
      tbl.rows.forEach((row, i) => {
        const y = 151 + i * 40;
        cardTxt(112, y, row[0]);
        cardTxt(cols.val, y, String(row[1]));
        cardTxt(cols.pctR, y, row[2], "end");
      });
      for (const [m, r] of Object.entries(hlRects)) {
        const on = Number(m) === n;
        r.setAttribute("fill", on ? "rgba(88,209,192,0.13)" : "transparent");
        r.setAttribute("stroke", on ? "rgba(88,209,192,0.85)" : "none");
      }
    }
    /* hover rects go in first so row text stays selectable above them */
    for (const n of Object.keys(STALLS).map(Number)) {
      const r = svg("rect", {
        x: 24, y: yOf(n) - 30, width: CW - 48, height: 42, rx: 5,
        fill: "transparent", "stroke-width": 1.5, "vector-effect": "non-scaling-stroke",
      });
      r.addEventListener("pointerenter", () => show(n));
      hlRects[n] = r;
      layer.appendChild(r);
    }
    for (const [nStr, code] of Object.entries(SRC)) {
      const n = Number(nStr);
      const num = txt(NUM_X, yOf(n), nStr, "end", true);
      layer.appendChild(num);
      const nodes = [num];
      if (code) {
        const t = txt(CODE_X, yOf(n), code, null, true);
        layer.appendChild(t);
        nodes.push(t);
      }
      if (STALLS[n]) {
        const pct = txt(PCT_X, yOf(n), STALLS[n], "end", true);
        layer.appendChild(pct);
        nodes.push(pct);
        for (const el2 of nodes) el2.addEventListener("pointerenter", () => show(n));
      }
    }
    inner.appendChild(layer);
    wrap.appendChild(detail);
    body.appendChild(wrap);
    show(211); /* seed with the hot line: the strict-FP32 divide chain */
  };

  /* =========================================================
   * Header figure: async RL timeline (perfetto-style)
   * ========================================================= */

  NV.registry["async-rl-timeline"] = function (fig) {
    const body = card(
      fig,
      "RL Training Simulator",
      "samplers generate rollouts, the trainer consumes batches — trade staleness and precision for throughput",
      "hover over any span · click a dashed reward curve"
    );

    /* Mismatch floors and step times are measured means from this post's own
     * figures (train-rollout logprob-diff, rollout-time, and train-time charts,
     * all at the last-15%-BF16 setting): floors 0.0103/0.0203/0.0313/0.0328,
     * rollout 109.4/93.6/83.8/84.8 s, train 64.3s BF16 with dequantized backward
     * costing ~9% on NVFP4. Staleness growth and drift dynamics stay schematic. */
    /* Rollout speed uses the ~2x-per-tier gap observed for large models
     * (SemiAnalysis InferenceX); train-step times stay the post's measured
     * 30B-ablation values (NVFP4 training is not faster at that scale). */
    const PREC = {
      bf16: { label: "BF16", floor: 0.0103, rollout: 1.0, train: { hp: 1.0, dq: 1.0 } },
      fp8: { label: "MXFP8", floor: 0.0203, rollout: 0.5, train: { hp: 0.768, dq: 0.837 } },
      fp4: { label: "NVFP4", floor: 0.0328, rollout: 0.25, train: { hp: 1.022, dq: 1.095 } },
    };
    /* NVFP4-specific flags (measured): 4/6 lowers the floor to 0.0313 and adds
     * no rollout overhead; both default OFF = naive FP4 */
    const FOUR = { floor: 0.0313, rollout: 0.25 };
    const BF16_FLOOR = 0.0103; /* train-inference mismatch exists even in BF16 */
    /* independent BF16-exception toggles scale the quantized excess over the
     * BF16 floor (calibrated to the post's ablations: last-15% cuts the excess
     * to ~0.62x, shared experts to ~0.8x; they compose) */
    const EXCEPT_BASE = 1.6;
    const LAST15_FACTOR = 0.625;
    const SE_FACTOR = 0.8;
    const exceptMult = (cfg) => EXCEPT_BASE * (cfg.layers15 ? LAST15_FACTOR : 1) * (cfg.se ? SE_FACTOR : 1);
    const exceptLabel = (cfg) =>
      cfg.layers15 && cfg.se ? "last 15% + shared experts" : cfg.layers15 ? "last 15%" : cfg.se ? "shared experts" : "none";
    const BITEXACT_OFF = 0.008; /* illustrative: naive FP4's rollout/training quantizers disagreeing (4/6 ships bit-exact) */
    /* every lever acts through the mismatch floor; none gets a hidden
     * dynamics multiplier — the backward mode's cost is spikes (below) */
    /* occasional gradient spikes — an NVFP4-only phenomenon (the post's Adam
     * grad-norm figure): the coarse grid plus a mismatched high-precision
     * backward throws large ones, naive FP4 moderate ones; dequantized
     * backward + 4/6 removes them. At MXFP8 dequantized backward steadies
     * gradient noise (the grad-norm growth figure) without a spike signature,
     * so it gets no drift effect here. A single spike is cheap — each kick is
     * a drift impulse that rho decays — so the cost is cumulative: enough of
     * them ratchet drift toward the runaway regime, and reward feels them only
     * through that accumulated drift. One draw per step, config-independent,
     * so a step that spikes under dq also spikes (harder) under hp */
    const SPIKE_HP = { p: 0.08, kick: 0.4 };
    const SPIKE_NAIVE = { p: 0.05, kick: 0.3 };
    function spikeKick(cfg, u) {
      const pHp = cfg.prec === "fp4" && cfg.bwd === "hp" ? SPIKE_HP.p : 0;
      const pNaive = cfg.prec === "fp4" && !cfg.four ? SPIKE_NAIVE.p : 0;
      if (u < pHp) return SPIKE_HP.kick;
      if (u < pHp + pNaive) return SPIKE_NAIVE.kick;
      return 0;
    }
    function precFloor(cfg) {
      return cfg.prec === "fp4" && cfg.four ? FOUR.floor : PREC[cfg.prec].floor;
    }
    function floorFor(cfg) {
      const excess = (precFloor(cfg) - BF16_FLOOR) * exceptMult(cfg);
      const bit = cfg.prec === "fp4" && !cfg.four ? BITEXACT_OFF : 0;
      return BF16_FLOOR + excess + bit;
    }
    function precLabel(cfg) {
      return PREC[cfg.prec].label + (cfg.prec === "fp4" && cfg.four ? " + 4/6" : "");
    }
    /* weight broadcast: blocks the trainer in proportion to weight bits */
    const BITS = { bf16: 16, fp8: 8, fp4: 4 };
    const SYNC_PER_BIT = 0.04;
    const HORIZON = {
      short: { label: "short", scale: 1.0, tail: 1.7 },
      med: { label: "medium", scale: 1.8, tail: 2.6 },
      long: { label: "long", scale: 3.2, tail: 3.6 },
    };
    const STALE_KL = 0.011; /* KL per step of staleness, superlinear in the gap */
    const staleCost = (gap) => STALE_KL * Math.pow(gap, 1.5);
    /* Drift dynamics: each batch's (corrected) mismatch feeds policy drift,
     * training self-corrects at rate rho, and mismatch x drift feeds back:
     *   drift' = drift + (1 + FB*drift) * mEff - rho * drift
     * with mEff = m^2/(m + knee) (the correction machinery absorbs small
     * mismatch). Below a critical mismatch the drift saturates at a steady
     * state; above it there is no fixed point and the run runs away. */
    const A = 6; /* drift inflow gain */
    const FB = 0.65; /* quadratic feedback: mismatch x drift^2 */
    const PREC_ADAPT = 0.5; /* the policy partly adapts to a consistent numerical distortion */
    const LEAK = 0.05; /* display: drifted policy's contribution to each rollout */
    const RUNAWAY = 2.2; /* drift level we call collapsed (plateaus sit well below) */
    /* trust-region machinery (PPO-style clipping): the knee is the mismatch it
     * absorbs per batch, rho the self-correction rate */
    const CORR_KNEE = 0.02;
    const CORR_RHO = 0.12;
    const S = 4;
    const T_BASE = 48;
    const winFor = (cfg) => Math.round(T_BASE * (1 + (HORIZON[cfg.horizon].scale - 1) * 0.47));
    const TRAIN_BASE = 1.7; /* measured train/rollout ratio: 64.3s vs 109.4s in BF16 */

    const state = {
      stale: 2, prec: "fp8", four: false, bwd: "dq",
      layers15: true, se: false, batch: 4, horizon: "med", wsync: false,
    };

    function lcg(seed) {
      let s = seed >>> 0;
      return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    }

    function simulate(cfg) {
      cfg = cfg || state;
      const rnd = lcg(97);
      const srnd = lcg(41); /* spike draws: one per step, config-independent */
      const hor = HORIZON[cfg.horizon];
      const rolloutScale = cfg.prec === "fp4" && cfg.four ? FOUR.rollout : PREC[cfg.prec].rollout;
      const speed = rolloutScale * hor.scale;
      const dur = () => {
        const u = rnd();
        const base = 1.5 + Math.min(9, -Math.log(1 - u * 0.988) * hor.tail);
        return base * speed;
      };
      const T = winFor(cfg);
      const trainDur = TRAIN_BASE * PREC[cfg.prec].train[cfg.prec === "bf16" ? "hp" : cfg.bwd];
      const floorF = floorFor(cfg);
      const fbEff = FB;
      const syncDur = cfg.wsync ? SYNC_PER_BIT * BITS[cfg.prec] : 0;
      const samplers = Array.from({ length: S }, () => ({ freeAt: 0, cur: null, v: 0 }));
      const rollouts = [];
      const steps = [];
      const syncs = [];
      const syncSpans = [];
      const queue = [];
      let version = 0;
      let training = false;
      let trainEnd = 0;
      let syncEnd = 0;
      let lastBroadcast = 0;
      let drift = 0;
      const dt = 0.02;
      const mixOf = (r) => (r.ticks ? r.vTicks / r.ticks : r.v);
      for (let t = 0; t <= T + 1e-9; t += dt) {
        /* in-flight rollouts accumulate the version they are being sampled with */
        for (const sp of samplers) {
          if (sp.cur !== null) {
            const r = rollouts[sp.cur];
            r.vTicks += sp.v;
            r.ticks += 1;
          }
        }
        for (const sp of samplers) {
          if (sp.cur !== null && t >= sp.freeAt - 1e-9) {
            queue.push(sp.cur);
            sp.cur = null;
          }
        }
        if (training && t >= trainEnd - 1e-9) {
          training = false;
          version += 1;
          if (cfg.wsync) {
            syncEnd = trainEnd + syncDur;
            syncSpans.push({ k: version, start: trainEnd, end: syncEnd });
          }
        }
        if (cfg.wsync && lastBroadcast < version && t >= syncEnd - 1e-9) {
          /* broadcast lands: every sampler switches, including mid-rollout */
          for (const sp of samplers) {
            if (sp.cur !== null) rollouts[sp.cur].wsyncs.push({ t: Math.min(syncEnd, T), v: version });
            sp.v = version;
          }
          lastBroadcast = version;
          syncs.push({ global: true, t: Math.min(syncEnd, T) });
        }
        for (let qi = queue.length - 1; qi >= 0; qi--) {
          const r = rollouts[queue[qi]];
          if (version - mixOf(r) > cfg.stale + 1e-6) {
            r.discarded = true;
            r.mixV = mixOf(r);
            r.staleness = version - r.mixV;
            r.leftQueueAt = t;
            queue.splice(qi, 1);
          }
        }
        if (!training && t >= syncEnd - 1e-9 && queue.length >= cfg.batch && t <= T - 1e-9) {
          queue.sort((a, b) => rollouts[a].end - rollouts[b].end);
          const take = queue.splice(0, cfg.batch);
          let baseSum = 0;
          let staleSum = 0;
          for (const ri of take) {
            const r = rollouts[ri];
            r.mixV = mixOf(r);
            r.consumeV = version;
            r.leftQueueAt = t;
            r.staleness = Math.max(0, version - r.mixV);
            const stale = staleCost(r.staleness);
            staleSum += stale;
            const base = stale + floorF;
            r.driftPart = LEAK * drift;
            r.mismatch = base + r.driftPart;
            r.step = steps.length;
            baseSum += base;
          }
          const mean = baseSum / take.length;
          const mDyn = staleSum / take.length + PREC_ADAPT * floorF;
          const mEff = mDyn > 0 ? (mDyn * mDyn) / (mDyn + CORR_KNEE) : 0;
          drift = Math.max(0, drift + A * mEff + fbEff * mEff * drift * drift - CORR_RHO * drift);
          const kick = spikeKick(cfg, srnd());
          const spiked = kick > 0;
          drift += kick;
          drift = Math.min(drift, RUNAWAY * 4); /* past runaway the number is meaningless */
          training = true;
          trainEnd = t + trainDur;
          steps.push({ start: t, end: trainEnd, k: steps.length, consumed: take, mean, mEff, drift, spiked });
        }
        for (const sp of samplers) {
          if (sp.cur !== null || t < sp.freeAt - 1e-9 || t + 0.3 > T) continue;
          /* bounded rollout buffer: once a full batch is already waiting, generating
           * more only ages in the queue — samplers pause instead */
          if (queue.length >= cfg.batch) continue;
          if (cfg.stale === 0) {
            /* fully synchronous: a rollout started while the trainer steps — or, with
             * the sync-cost model, before the broadcast lands — bakes in staleness > 0
             * and would only be generated to be discarded */
            const inflight = samplers.filter((x) => x.cur !== null).length;
            if (training || (cfg.wsync && lastBroadcast < version) || inflight + queue.length >= cfg.batch) continue;
          }
          /* without the sync-cost model, samplers pull the latest weights for free
           * at rollout start; with it, weights only arrive via the broadcast */
          if (!cfg.wsync && sp.v !== version) {
            syncs.push({ sampler: samplers.indexOf(sp), t });
            sp.v = version;
          }
          const d = dur();
          const idx = rollouts.length;
          rollouts.push({
            sampler: samplers.indexOf(sp), start: t, end: Math.min(t + d, T),
            clipped: t + d > T, v: sp.v, vTicks: 0, ticks: 0, mixV: sp.v,
            discarded: false, step: null, staleness: null, mismatch: null, wsyncs: [],
          });
          sp.cur = idx;
          sp.freeAt = t + d;
        }
      }
      /* the score is the drift itself; collapse = ENDING in runaway (a transient
       * excursion the correction machinery pulls back is not a dead run) */
      let divergedAt = -1;
      if (steps.length && steps[steps.length - 1].drift > RUNAWAY) {
        divergedAt = steps.length - 1;
        while (divergedAt > 0 && steps[divergedAt - 1].drift > RUNAWAY) divergedAt--;
      }
      const score = steps.map((st) => ({ t: st.end, v: st.drift }));
      /* simulated reward: saturating learning where each step's gain shrinks with
       * that batch's mismatch and the drift already accumulated; collapse crashes it */
      const rr = lcg(13);
      let rew = 0.45;
      const reward = steps.map((st, i) => {
        if (divergedAt >= 0 && i >= divergedAt) {
          rew = Math.max(0.08, rew * 0.8 - 0.01);
        } else {
          /* small, corrected mismatch is nearly free (the post's reward curves track
           * the BF16 baseline); the penalty is quadratic, biting only near the edge.
           * spikes carry no direct reward hit — their cost arrives through the
           * drift they accumulate */
          const eff = Math.max(0.15, 1 - Math.pow(st.mean / 0.22, 2)) * (1 - Math.min(0.5, st.drift * 0.12));
          rew += 0.06 * (0.88 - rew) * eff;
        }
        rew += (rr() - 0.5) * 0.012;
        return { t: st.end, v: rew };
      });
      const busy = rollouts.reduce((a, r) => a + (r.end - r.start), 0);
      const consumed = rollouts.filter((r) => r.step !== null);
      return {
        rollouts, steps, syncs, syncSpans, score, reward, divergedAt, T,
        util: busy / (S * T),
        discards: rollouts.filter((r) => r.discarded).length,
        meanStaleness: consumed.length ? consumed.reduce((a, r) => a + r.staleness, 0) / consumed.length : 0,
        meanMismatch: steps.length ? steps.reduce((a, s2) => a + s2.mean, 0) / steps.length : 0,
        finalDrift: drift,
      };
    }

    /* mismatch tint: teal -> rust (mids are muted clay, distinct from the trainer's sand) */
    function tint(m) {
      const t = Math.max(0, Math.min(1, m / 0.16));
      const a = [158, 200, 191];
      const b = [192, 91, 65];
      const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }

    /* layout: phones stack the reward panel ABOVE the timeline at full width,
       in a narrower viewBox so text renders larger; wide screens keep the
       right-hand reward column */
    const NARROW = (fig.clientWidth || 700) < 520;
    const W = NARROW ? 420 : 940;
    const GUT = NARROW ? 66 : 86;
    const R = 12;
    const PLOT_R = NARROW ? W - 14 : W - 244; /* time plots end here */
    const laneH = 24;
    const laneGap = 7;
    const REW_BAND = NARROW ? 212 : 0; /* top band reserved for the reward panel */
    const lanesTop = 24 + REW_BAND;
    const lanesH = (S + 1) * laneH + S * laneGap;
    const scoreTop = lanesTop + lanesH + 40;
    const scoreH = 62;
    const rateTop = scoreTop + scoreH + 32;
    const rateH = 46;
    /* phones drop the batch-mismatch strip entirely */
    const H = NARROW ? scoreTop + scoreH + 30 : rateTop + rateH + 26;
    let curT = T_BASE;
    const sx = (t) => GUT + (t / curT) * (PLOT_R - GUT);
    const laneY = (i) => lanesTop + i * (laneH + laneGap);
    const yFor = (v) => scoreTop + scoreH - (Math.min(v, RUNAWAY * 1.3) / (RUNAWAY * 1.3)) * scoreH;
    const RATE_MAX = 0.22;
    const yRate = (v) => rateTop + rateH - (Math.min(v, RATE_MAX) / RATE_MAX) * rateH;

    /* lever changes morph spans to their new schedule (interaction-triggered only) */
    const REDUCE = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const MOVE = REDUCE ? "" : "x 0.45s ease, y 0.45s ease, width 0.45s ease, fill 0.45s ease, opacity 0.25s ease";
    const MOVE_DOT = REDUCE ? "" : "cx 0.45s ease, cy 0.45s ease, fill 0.45s ease, opacity 0.25s ease";
    const MOVE_TF = REDUCE ? "" : "transform 0.45s ease, opacity 0.25s ease";

    const plotWrap = el("div", { class: "viz-plot" });
    const svgRoot = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}`, style: "cursor:default" });
    const tooltip = el("div", { class: "viz-tooltip" });
    plotWrap.appendChild(svgRoot);
    plotWrap.appendChild(tooltip);
    body.appendChild(plotWrap);

    const stats = el("div", { class: "tl2-key" });
    body.appendChild(stats);

    const controlsQuant = el("div", { class: "viz-controls", style: `gap:6px ${NARROW ? 12 : 18}px` });
    const divider = el("div", { class: "prf-divider" });
    const controls = el("div", { class: "viz-controls", style: `gap:6px ${NARROW ? 12 : 18}px` });
    body.appendChild(controlsQuant);
    body.appendChild(divider);
    body.appendChild(controls);

    const controlSync = []; /* refreshers: presets re-sync every control's pressed state */
    function group(target, label, options, get, set, onchange) {
      const wrap = el("span", { class: "prf-group" }, el("span", { class: "prf-glabel", text: label }));
      const btns = [];
      for (const opt of options) {
        const b = el("button", {
          class: "viz-btn",
          "aria-pressed": String(get() === opt.value),
          text: opt.label,
          onclick: () => {
            set(opt.value);
            btns.forEach((x) => x.setAttribute("aria-pressed", "false"));
            b.setAttribute("aria-pressed", "true");
            if (onchange) onchange();
            update();
          },
        });
        btns.push(b);
        wrap.appendChild(b);
      }
      controlSync.push(() => btns.forEach((x, i) => x.setAttribute("aria-pressed", String(get() === options[i].value))));
      target.appendChild(wrap);
      return wrap;
    }

    /* single flippable button for a binary flag */
    function toggle(target, label, get, set) {
      const b = el("button", {
        class: "viz-btn",
        "aria-pressed": String(get()),
        text: label,
        onclick: () => {
          set(!get());
          b.setAttribute("aria-pressed", String(get()));
          update();
        },
      });
      controlSync.push(() => b.setAttribute("aria-pressed", String(get())));
      target.appendChild(b);
      return b;
    }

    /* quantization row: precision, its flags, backward, layer exceptions */
    let fp4FlagEls = [];
    /* visibility, not display: the flag keeps its slot so the row doesn't reflow */
    const syncFp4Flags = () => {
      fp4FlagEls.forEach((w) => (w.style.visibility = state.prec === "fp4" ? "visible" : "hidden"));
    };
    group(controlsQuant, "precision", Object.entries(PREC).map(([k, pr]) => ({ value: k, label: pr.label })),
      () => state.prec, (v) => (state.prec = v), syncFp4Flags);
    fp4FlagEls = [
      toggle(controlsQuant, "4over6", () => state.four, (v) => (state.four = v)),
    ];
    toggle(controlsQuant, NARROW ? "dequant bwd" : "dequantized backward", () => state.bwd === "dq", (v) => (state.bwd = v ? "dq" : "hp"));
    const layersGroup = el("span", { class: "prf-group" }, el("span", { class: "prf-glabel", text: "bf16 layers" }));
    controlsQuant.appendChild(layersGroup);
    toggle(layersGroup, "last 15%", () => state.layers15, (v) => (state.layers15 = v));
    toggle(layersGroup, "shared experts", () => state.se, (v) => (state.se = v));
    syncFp4Flags();

    /* clicking a dashed reward reference adopts that config */
    function applyReference(patch) {
      Object.assign(state, patch);
      controlSync.forEach((f) => f());
      syncFp4Flags();
      update();
    }

    /* off-policy cap as a scrubbable slider (any integer cap works in the model) */
    const staleWrap = el("span", { class: "prf-group" }, el("span", { class: "prf-glabel", text: "off-policy ≤" }));
    const staleVal = el("span", { class: "prf-sliderval", text: state.stale === 0 ? "0 (sync)" : String(state.stale) });
    const staleSlider = el("input", {
      type: "range", min: "0", max: "8", step: "1", value: String(state.stale),
      "aria-label": "off-policy staleness cap",
      oninput: () => {
        const v = Number(staleSlider.value);
        if (v === state.stale) return;
        state.stale = v;
        staleVal.textContent = v === 0 ? "0 (sync)" : String(v);
        update();
      },
    });
    staleWrap.appendChild(staleSlider);
    staleWrap.appendChild(staleVal);
    controls.appendChild(staleWrap);
    /* phones: the broadcast model is desktop detail; state stays at its default */
    if (!NARROW) toggle(controls, "weight sync", () => state.wsync, (v) => (state.wsync = v));
    /* snap slider over a fixed set of stops */
    function stopSlider(target, label, stops, get, set) {
      const wrap = el("span", { class: "prf-group" }, el("span", { class: "prf-glabel", text: label }));
      const idx0 = Math.max(0, stops.findIndex((st) => st.value === get()));
      const val = el("span", { class: "prf-sliderval", text: stops[idx0].label });
      const slider = el("input", {
        type: "range", min: "0", max: String(stops.length - 1), step: "1", value: String(idx0),
        style: "width:64px",
        "aria-label": label,
        oninput: () => {
          const st = stops[Number(slider.value)];
          if (st.value === get()) return;
          set(st.value);
          val.textContent = st.label;
          update();
        },
      });
      wrap.appendChild(slider);
      wrap.appendChild(val);
      target.appendChild(wrap);
    }

    stopSlider(controls, "batch", [4, 8, 16].map((v) => ({ value: v, label: String(v) })),
      () => state.batch, (v) => (state.batch = v));
    stopSlider(controls, "horizon", Object.entries(HORIZON).map(([k, h]) => ({ value: k, label: h.label })),
      () => state.horizon, (v) => (state.horizon = v));

    /* "how this simulation works" holds the doc's figure caption */
    const captionFold = el("details", { class: "prf-capfold" });
    captionFold.appendChild(el("summary", { text: "how this simulation works" }));
    const docCap = fig.querySelector(":scope > figcaption");
    if (docCap) captionFold.appendChild(docCap);
    /* nested: the full mechanics, for the curious */
    const deepFold = el("details", { class: "prf-capfold" });
    deepFold.appendChild(el("summary", { text: "How this simulation really works" }));
    deepFold.appendChild(
      el("figcaption", {
        html:
          "Mismatch and train-step times are the measured means from this post's logprob-difference and timing " +
          "figures (BF16 has a 0.010 floor; dequantized backward costs ~9% train time and removes the gradient " +
          "spikes in NVFP4; their drift accumulates, and enough of them push a run toward runaway. At MXFP8 it " +
          "steadies gradient noise instead); rollout speed uses the ~2x gap per precision tier observed for " +
          "large models (<a href='https://inferencex.semianalysis.com/inference'>SemiAnalysis InferenceX</a>), " +
          "at which pace samplers can outrun the trainer and pause on a full rollout buffer. Samplers always " +
          "pull the latest weights when starting a rollout; staleness comes from rollouts that don't finish " +
          "before the trainer steps, and the off-policy cap throws out sufficiently stale examples " +
          "(0 is synchronous). The weight-sync button models inflight weight-sync, blocking the trainer in " +
          "proportion to precision, and in-flight rollouts switch to the new weights mid-generation, so " +
          "staleness becomes fractional. Span tint = staleness + precision error + the policy drift already " +
          "accumulated. Regularization absorbs small mismatch and speeds recovery, so drift saturates. Hatched " +
          "spans were discarded for exceeding the cap. Simulated reward: some mismatch is nearly free, but " +
          "learning degrades near the critical level and collapse crashes it. Speed compares train-step " +
          "throughput against fully-synchronous BF16 at the same batch size.",
      })
    );
    captionFold.appendChild(deepFold);
    body.appendChild(captionFold);

    function popover(html, cx, cy) {
      tooltip.innerHTML = html;
      tooltip.classList.add("on");
      const svgRect = svgRoot.getBoundingClientRect();
      placeTip(plotWrap, tooltip, (cx / W) * svgRect.width, (cy / H) * svgRect.height - 40, 10);
    }

    function setAttrs(node, attrs) {
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    }

    function fadeRemove(node) {
      if (REDUCE) {
        node.remove();
        return;
      }
      node.style.opacity = "0";
      setTimeout(() => node.remove(), 300);
    }

    function fadeIn(node, skip) {
      if (skip || REDUCE) return;
      node.style.opacity = "0";
      requestAnimationFrame(() => (node.style.opacity = "1"));
    }

    function bindTip(node) {
      node.addEventListener("pointerenter", () => {
        (node.__consumed || []).forEach((k) => {
          const r = rolloutEls.get(k);
          if (r) r.classList.add("hl");
        });
        popover(node.__tip, node.__cx, node.__cy);
      });
      node.addEventListener("pointerleave", () => {
        (node.__consumed || []).forEach((k) => {
          const r = rolloutEls.get(k);
          if (r) r.classList.remove("hl");
        });
        tooltip.classList.remove("on");
      });
    }

    /* ---- static scaffold, built once ---- */
    const defs = svg("defs", {});
    defs.innerHTML =
      '<pattern id="prf-hatch" width="5" height="5" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">' +
      '<line x1="0" y1="0" x2="0" y2="5" stroke="#c4bcae" stroke-width="1.6"/></pattern>';
    svgRoot.appendChild(defs);

    for (let i = 0; i <= S; i++) {
      svgRoot.appendChild(
        svg("rect", {
          x: GUT, y: laneY(i), width: PLOT_R - GUT, height: laneH,
          fill: i === 0 ? "rgba(194,163,92,0.06)" : "rgba(22,21,19,0.025)", rx: 3,
        })
      );
      svgRoot.appendChild(
        svg("text", {
          x: GUT - 8, y: laneY(i) + laneH / 2 + 3.5, "text-anchor": "end",
          "font-size": 10.5, fill: "#8a847a",
          style: 'font-family:"SF Mono",Menlo,ui-monospace,monospace',
          text: i === 0 ? "trainer" : "sampler " + i,
        })
      );
    }
    const axisY = laneY(S) + laneH + 12;
    svgRoot.appendChild(
      svg("line", { x1: GUT, x2: PLOT_R, y1: axisY, y2: axisY, stroke: "#c4bcae", "stroke-width": 1, "shape-rendering": "crispEdges" })
    );
    svgRoot.appendChild(
      svg("text", { x: PLOT_R, y: axisY + 13, "text-anchor": "end", "font-size": 10, "font-weight": 600, "letter-spacing": "0.08em", fill: "#8a847a", text: "TIME →" })
    );
    const stepTicksG = svg("g", {});
    svgRoot.appendChild(stepTicksG);
    svgRoot.appendChild(
      svg("text", { x: GUT - 8, y: axisY + 3, "text-anchor": "end", "font-size": 9, fill: "#8a847a", text: "steps" })
    );
    svgRoot.appendChild(
      svg("text", {
        x: GUT, y: scoreTop - 8, "font-size": 10, "font-weight": 600, "letter-spacing": "0.08em", fill: "#6d6860",
        text: NARROW ? "POLICY DRIFT" : "POLICY DRIFT (SATURATES BELOW THE CRITICAL MISMATCH; RUNS AWAY ABOVE)",
      })
    );
    svgRoot.appendChild(
      svg("line", { x1: GUT, x2: PLOT_R, y1: scoreTop + scoreH, y2: scoreTop + scoreH, stroke: "#c4bcae", "stroke-width": 1, "shape-rendering": "crispEdges" })
    );

    /* dynamic layers, z-ordered */
    if (!NARROW) {
      svgRoot.appendChild(
        svg("text", {
          x: GUT, y: rateTop - 8, "font-size": 10, "font-weight": 600, "letter-spacing": "0.08em", fill: "#6d6860",
          text: "BATCH MISMATCH PER STEP (AVERAGE OFF-POLICYNESS)",
        })
      );
      svgRoot.appendChild(
        svg("line", { x1: GUT, x2: PLOT_R, y1: rateTop + rateH, y2: rateTop + rateH, stroke: "#c4bcae", "stroke-width": 1, "shape-rendering": "crispEdges" })
      );
    }
    const shadeBottom = NARROW ? scoreTop + scoreH : rateTop + rateH;
    const shadeRect = svg("rect", { y: lanesTop - 4, height: shadeBottom - lanesTop + 4, fill: "rgba(163,62,45,0.09)", x: 0, width: 0 });
    shadeRect.style.opacity = "0";
    shadeRect.style.transition = MOVE;
    svgRoot.appendChild(shadeRect);
    const threshY = yFor(RUNAWAY);
    svgRoot.appendChild(
      svg("line", {
        x1: GUT, x2: PLOT_R, y1: threshY, y2: threshY,
        stroke: "#a33e2d", "stroke-width": 1, "stroke-dasharray": "5 4", "stroke-opacity": 0.6,
      })
    );
    svgRoot.appendChild(
      svg("text", {
        x: PLOT_R, y: threshY - 5, "text-anchor": "end", "font-size": 9.5, fill: "#a33e2d", "fill-opacity": 0.8,
        text: "runaway",
      })
    );
    const shadeLabel = svg("text", { x: 0, y: 0, "font-size": 12.5, "font-weight": 700, fill: "#a33e2d", text: "diverges →" });
    shadeLabel.style.opacity = "0";
    shadeLabel.style.transition = MOVE_TF;
    svgRoot.appendChild(shadeLabel);
    const critG = svg(
      "g",
      {},
      svg("line", {
        x1: GUT, x2: PLOT_R, y1: 0, y2: 0,
        stroke: "#a33e2d", "stroke-width": 1, "stroke-dasharray": "5 4", "stroke-opacity": 0.6,
      }),
      svg("text", {
        x: PLOT_R, y: -5, "text-anchor": "end", "font-size": 9.5, fill: "#a33e2d", "fill-opacity": 0.8,
        text: "critical mismatch",
      })
    );
    critG.style.transition = MOVE_TF;
    if (!NARROW) svgRoot.appendChild(critG);

    /* reward panel: right column on wide screens, top band on phones */
    const RX0 = NARROW ? GUT : PLOT_R + 34;
    const RX1 = W - 14;
    const RY0 = NARROW ? 30 : lanesTop + 10;
    const RY1 = NARROW ? 186 : rateTop + rateH;
    const yRew = (v) => RY1 - (Math.min(v, 0.9) / 0.9) * (RY1 - RY0);
    svgRoot.appendChild(
      svg("text", {
        x: RX0, y: NARROW ? RY0 - 12 : lanesTop - 8, "font-size": 10, "font-weight": 600, "letter-spacing": "0.08em", fill: "#6d6860",
        text: "SIMULATED REWARD",
      })
    );
    for (const tv of [0.3, 0.6, 0.9]) {
      svgRoot.appendChild(
        svg("line", { x1: RX0, x2: RX1, y1: yRew(tv), y2: yRew(tv), stroke: "#eae4d8", "stroke-width": 1, "shape-rendering": "crispEdges" })
      );
      svgRoot.appendChild(
        svg("text", { x: RX0 - 5, y: yRew(tv) + 3.5, "text-anchor": "end", "font-size": 9.5, fill: "#8a847a", text: tv.toFixed(1) })
      );
    }
    svgRoot.appendChild(
      svg("line", { x1: RX0, x2: RX1, y1: RY1, y2: RY1, stroke: "#c4bcae", "stroke-width": 1, "shape-rendering": "crispEdges" })
    );
    svgRoot.appendChild(
      svg("text", { x: RX1, y: RY1 + 13, "text-anchor": "end", "font-size": 10, "font-weight": 600, "letter-spacing": "0.08em", fill: "#8a847a", text: "TIME →" })
    );
    const rewTicksG = svg("g", {});
    svgRoot.appendChild(rewTicksG);
    const rewEndDot = svg("circle", { r: 3, fill: "#176b64" });
    rewEndDot.style.transition = MOVE_DOT;
    const rewEndLabel = svg("text", { x: 0, y: 0, "font-size": 11, "font-weight": 700, fill: "#176b64", "text-anchor": "end" });
    rewEndLabel.style.transition = MOVE_TF;
    const ticksG = svg("g", {});
    const stepsG = svg("g", {});
    const rolloutsG = svg("g", {});
    const scorePathG = svg("g", {});
    const scoreDotsG = svg("g", {});
    const rateBarsG = svg("g", {});
    const rewardG = svg("g", {});
    svgRoot.appendChild(ticksG);
    svgRoot.appendChild(stepsG);
    svgRoot.appendChild(rolloutsG);
    svgRoot.appendChild(scorePathG);
    svgRoot.appendChild(scoreDotsG);
    svgRoot.appendChild(rateBarsG);
    svgRoot.appendChild(rewardG);
    svgRoot.appendChild(rewEndDot);
    svgRoot.appendChild(rewEndLabel);

    const stepEls = new Map();
    const syncEls = new Map();
    const rolloutEls = new Map();
    const tailEls = new Map();
    const segEls = new Map(); /* pre-broadcast sections of weight-synced rollouts */
    const dotEls = new Map();
    const barEls = new Map();
    /* fixed-sample path builder: constant command count lets CSS transition `d`,
     * so the drift and reward lines bend smoothly instead of crossfading */
    const PATH_N = 56;
    function resampleD(series, xFn, yFn) {
      if (!series.length) return "";
      const pts = series.length > 1 ? series : [series[0], series[0]];
      const t0 = pts[0].t;
      const t1 = pts[pts.length - 1].t;
      let d = "";
      let j = 0;
      for (let i = 0; i < PATH_N; i++) {
        const tt = t0 + ((t1 - t0) * i) / (PATH_N - 1);
        while (j < pts.length - 2 && pts[j + 1].t < tt) j++;
        const a = pts[j];
        const b = pts[Math.min(j + 1, pts.length - 1)];
        const f = b.t > a.t ? Math.min(1, Math.max(0, (tt - a.t) / (b.t - a.t))) : 0;
        d += (i ? "L" : "M") + xFn(a.t + (b.t - a.t) * f).toFixed(1) + " " + yFn(a.v + (b.v - a.v) * f).toFixed(1);
      }
      return d;
    }
    const PATH_TR = REDUCE ? "" : "d 0.45s ease, stroke 0.45s ease, opacity 0.25s ease";
    const driftPath = svg("path", { fill: "none", "stroke-width": 1.8, "stroke-linejoin": "round" });
    driftPath.style.transition = PATH_TR;
    driftPath.style.opacity = "0";
    scorePathG.appendChild(driftPath);
    const rewPath = svg("path", { fill: "none", "stroke-width": 1.8, "stroke-linejoin": "round" });
    rewPath.style.transition = PATH_TR;
    rewPath.style.opacity = "0";
    rewardG.appendChild(rewPath);
    const ghostEls = new Map(); /* precision -> {path, label} */
    let first = true;

    function update() {
      const sim = simulate(state);
      curT = sim.T;
      const baseline =
        state.prec === "bf16" && state.stale === 0
          ? sim
          : simulate({ ...state, prec: "bf16", stale: 0 });
      const speedX = sim.steps.length / Math.max(1, baseline.steps.length);

      /* train-step ticks on the time axes: one per completed step, on both
         the lane axis and the reward panel's */
      stepTicksG.textContent = "";
      rewTicksG.textContent = "";
      const xrTick = (t) => RX0 + (t / curT) * (RX1 - RX0 - 6);
      for (const st of sim.steps) {
        stepTicksG.appendChild(
          svg("line", {
            x1: sx(st.end), x2: sx(st.end), y1: axisY - 4, y2: axisY + 4,
            stroke: "#8a847a", "stroke-width": 1, "shape-rendering": "crispEdges",
          })
        );
        rewTicksG.appendChild(
          svg("line", {
            x1: xrTick(st.end), x2: xrTick(st.end), y1: RY1 - 3, y2: RY1 + 3,
            stroke: "#8a847a", "stroke-width": 1, "shape-rendering": "crispEdges",
          })
        );
      }
      /* sync ticks: thin hairlines, rebuilt in place (the spans carry the motion) */
      ticksG.textContent = "";
      for (const sy of sim.syncs) {
        if (sy.global) {
          ticksG.appendChild(
            svg("line", {
              x1: sx(sy.t), x2: sx(sy.t), y1: laneY(1) - 2, y2: laneY(S) + laneH + 2,
              stroke: "#176b64", "stroke-width": 1, "stroke-dasharray": "2.5 3.5", "stroke-opacity": 0.6,
            })
          );
        } else {
          const y0 = laneY(sy.sampler + 1);
          ticksG.appendChild(
            svg("line", {
              x1: sx(sy.t), x2: sx(sy.t), y1: y0 - 2, y2: y0 + laneH + 2,
              stroke: "#176b64", "stroke-width": 1.4, "stroke-opacity": 0.7,
            })
          );
        }
      }

      /* trainer spans, keyed by step index */
      sim.steps.forEach((st) => {
        const x = sx(st.start);
        const w = Math.max(2.5, sx(st.end) - x);
        let e = stepEls.get(st.k);
        if (!e) {
          const rect = svg("rect", {
            class: "prf-span", x: 0, y: laneY(0) + 2, width: w, height: laneH - 4, rx: 3,
            fill: "#f4ecdb", stroke: "#c2a35c", "stroke-width": 1,
          });
          const label = svg("text", {
            x: w / 2, y: laneY(0) + laneH / 2 + 3, "text-anchor": "middle",
            "font-size": 9.5, fill: "#6d6860", "pointer-events": "none",
          });
          const g = svg("g", {});
          g.style.transition = MOVE_TF;
          rect.style.transition = MOVE;
          g.appendChild(rect);
          g.appendChild(label);
          stepsG.appendChild(g);
          bindTip(rect);
          e = { g, rect, label };
          stepEls.set(st.k, e);
          fadeIn(g, first);
        }
        e.g.style.transform = `translate(${x}px, 0px)`;
        e.rect.setAttribute("width", w);
        e.label.setAttribute("x", w / 2);
        e.label.textContent = w > 30 ? "π" + st.k + "→" + (st.k + 1) : "";
        e.rect.__consumed = st.consumed;
        e.rect.__cx = x + w / 2;
        e.rect.__cy = laneY(0) + laneH;
        e.rect.__tip =
          `<div class="tt-x">train step ${st.k + 1}</div>` +
          `consumes ${st.consumed.length} rollouts · batch mismatch <b>${st.mean.toFixed(3)}</b> · drift now <b>${st.drift.toFixed(2)}</b>` +
          (st.spiked ? ` · <b style="color:#a33e2d">gradient spike</b>` : "") + `<br>` +
          `updates π<sub>${st.k}</sub> → π<sub>${st.k + 1}</sub>; samplers pull the new weights when they resync`;
      });
      for (const [k, e] of [...stepEls]) {
        if (k >= sim.steps.length) {
          stepEls.delete(k);
          fadeRemove(e.g);
        }
      }

      /* weight-sync spans on the trainer lane */
      const liveSyncs = new Set((sim.syncSpans || []).map((ss) => ss.k));
      (sim.syncSpans || []).forEach((ss) => {
        const x = sx(ss.start);
        const w = Math.max(2, sx(Math.min(ss.end, sim.T)) - x);
        let rect = syncEls.get(ss.k);
        if (!rect) {
          rect = svg("rect", {
            class: "prf-span", y: laneY(0) + 5, height: laneH - 10, rx: 2,
            fill: "rgba(23,107,100,0.28)", stroke: "#176b64", "stroke-width": 1,
          });
          rect.style.transition = MOVE;
          stepsG.appendChild(rect);
          bindTip(rect);
          syncEls.set(ss.k, rect);
          fadeIn(rect, first);
        }
        setAttrs(rect, { x, width: w });
        rect.__cx = x + w / 2;
        rect.__cy = laneY(0) + laneH;
        rect.__tip =
          `<div class="tt-x">weight sync</div>` +
          `broadcasts π<sub>${ss.k}</sub> to every sampler — in-flight rollouts switch mid-generation<br>` +
          `${BITS[state.prec]}-bit weights block the trainer for ${(ss.end - ss.start).toFixed(2)} time units`;
      });
      for (const [k, rect] of [...syncEls]) {
        if (!liveSyncs.has(k)) {
          syncEls.delete(k);
          fadeRemove(rect);
        }
      }

      /* rollout spans, keyed by creation index (same seed -> the k-th rollout is the
       * same episode across settings, so spans morph meaningfully) */
      const segLive = new Set();
      sim.rollouts.forEach((r, k) => {
        const x = sx(r.start);
        const w = Math.max(2.5, sx(r.end) - x);
        const y = laneY(r.sampler + 1) + 2;
        let tail = tailEls.get(k);
        const tailW = r.leftQueueAt ? sx(Math.min(r.leftQueueAt, sim.T)) - sx(r.end) : 0;
        if (tailW > 2) {
          if (!tail) {
            tail = svg("rect", { height: 4, rx: 2, fill: "rgba(22,21,19,0.13)", "pointer-events": "none" });
            tail.style.transition = MOVE;
            rolloutsG.insertBefore(tail, rolloutsG.firstChild);
            tailEls.set(k, tail);
            fadeIn(tail, first);
          }
          setAttrs(tail, { x: sx(r.end), y: y + (laneH - 4) / 2 - 2, width: tailW });
        } else if (tail) {
          tailEls.delete(k);
          fadeRemove(tail);
        }
        let fill;
        let stroke = "none";
        let dash = null;
        if (r.discarded) {
          fill = "url(#prf-hatch)";
          stroke = "#b9b2a5";
        } else if (r.step !== null) {
          const nSync = r.wsyncs.length;
          const vLast = nSync ? r.wsyncs[nSync - 1].v : r.v;
          const base = r.mismatch - staleCost(r.staleness);
          fill = nSync ? tint(staleCost(Math.max(0, r.consumeV - vLast)) + base) : tint(r.mismatch);
        } else {
          fill = "rgba(22,21,19,0.07)";
          stroke = "#c4bcae";
          dash = "3 3";
        }
        let rect = rolloutEls.get(k);
        if (!rect) {
          rect = svg("rect", { class: "prf-span", x, y, width: w, height: laneH - 4, rx: 3 });
          rect.style.transition = MOVE;
          rolloutsG.appendChild(rect);
          bindTip(rect);
          rolloutEls.set(k, rect);
          fadeIn(rect, first);
        }
        setAttrs(rect, { x, y, width: w, fill, stroke, "stroke-width": stroke === "none" ? 0 : 1 });
        if (dash) rect.setAttribute("stroke-dasharray", dash);
        else rect.removeAttribute("stroke-dasharray");
        rect.__cx = x + w / 2;
        rect.__cy = y + laneH;
        /* sections sampled before a mid-rollout broadcast keep the older version's tint */
        if (r.step !== null && r.wsyncs.length) {
          const base = r.mismatch - staleCost(r.staleness);
          const pts = [{ t: r.start, v: r.v }, ...r.wsyncs];
          for (let i = 0; i < pts.length - 1; i++) {
            const segX = sx(pts[i].t);
            const segW = sx(pts[i + 1].t) - segX;
            const key = `${k}:${i}`;
            let seg = segEls.get(key);
            if (segW < 2) {
              if (seg) { segEls.delete(key); fadeRemove(seg); }
              continue;
            }
            if (!seg) {
              seg = svg("rect", { height: laneH - 4, rx: 3, "pointer-events": "none" });
              seg.style.transition = MOVE;
              rolloutsG.appendChild(seg);
              segEls.set(key, seg);
              fadeIn(seg, first);
            }
            setAttrs(seg, { x: segX, y, width: segW, fill: tint(staleCost(Math.max(0, r.consumeV - pts[i].v)) + base) });
            segLive.add(key);
          }
        }
        let bodyTxt;
        if (r.discarded) {
          bodyTxt = `sampled with ${state.wsync ? "π̄" : "π"}<sub>${state.wsync ? r.mixV.toFixed(1) : r.v}</sub><br><b>discarded</b> — exceeded the staleness cap (${state.stale}) before the trainer could use it`;
        } else if (r.step !== null) {
          const sc = staleCost(r.staleness);
          bodyTxt =
            `sampled with ${state.wsync ? "π̄" : "π"}<sub>${state.wsync ? r.mixV.toFixed(1) : r.v}</sub>` +
            (r.wsyncs.length ? ` (weights updated mid-rollout${r.wsyncs.length > 1 ? " ×" + r.wsyncs.length : ""})` : "") +
            ` · consumed by step ${r.step + 1} at π<sub>${r.consumeV}</sub>` +
            (r.leftQueueAt - r.end > 0.3 ? ` after ${(r.leftQueueAt - r.end).toFixed(1)} in the queue` : "") +
            `<br>` +
            `staleness ${r.staleness.toFixed(1)} → ${sc.toFixed(3)} · precision (${precLabel(state)}, bf16: ${exceptLabel(state)}) ${floorFor(state).toFixed(3)}` +
            (r.driftPart > 0.0005 ? ` · drift so far ${r.driftPart.toFixed(3)}` : "") +
            `<br>mismatch <b>${r.mismatch.toFixed(3)}</b>`;
        } else {
          bodyTxt = `sampled with π<sub>${r.v}</sub><br>${r.clipped ? "still generating at the window edge" : "waiting for the trainer"}`;
        }
        rect.__tip = `<div class="tt-x">sampler ${r.sampler + 1} · rollout</div>` + bodyTxt;
      });
      for (const [k, rect] of [...rolloutEls]) {
        if (k >= sim.rollouts.length) {
          rolloutEls.delete(k);
          fadeRemove(rect);
        }
      }
      for (const [k, tail] of [...tailEls]) {
        if (k >= sim.rollouts.length) {
          tailEls.delete(k);
          fadeRemove(tail);
        }
      }
      for (const [key, seg] of [...segEls]) {
        if (!segLive.has(key)) {
          segEls.delete(key);
          fadeRemove(seg);
        }
      }

      /* score: dots morph, the connecting path crossfades */
      const lineColor = sim.divergedAt >= 0 ? "#a33e2d" : "#176b64";
      sim.score.forEach((pt, i) => {
        let c = dotEls.get(i);
        if (!c) {
          c = svg("circle", { r: 2.6 });
          c.style.transition = MOVE_DOT;
          scoreDotsG.appendChild(c);
          dotEls.set(i, c);
          fadeIn(c, first);
        }
        const spiked = sim.steps[i] && sim.steps[i].spiked;
        setAttrs(c, {
          cx: sx(pt.t), cy: yFor(pt.v), r: spiked ? 3.4 : 2.6,
          fill: sim.divergedAt >= 0 && i >= sim.divergedAt ? "#a33e2d" : "#176b64",
          stroke: spiked ? "#a33e2d" : "none", "stroke-width": spiked ? 1.6 : 0,
        });
      });
      for (const [k, c] of [...dotEls]) {
        if (k >= sim.score.length) {
          dotEls.delete(k);
          fadeRemove(c);
        }
      }
      /* rate strip: one bar per step, tinted on the same mismatch ramp */
      if (!NARROW) sim.steps.forEach((st) => {
        const bx = sx(st.end) - 3;
        const by = yRate(st.mean);
        let bar = barEls.get(st.k);
        if (!bar) {
          bar = svg("rect", { width: 6, rx: 1, "pointer-events": "none" });
          bar.style.transition = MOVE;
          rateBarsG.appendChild(bar);
          barEls.set(st.k, bar);
          fadeIn(bar, first);
        }
        setAttrs(bar, { x: bx, y: by, height: Math.max(1, rateTop + rateH - by), fill: tint(st.mean) });
      });
      for (const [k, bar] of [...barEls]) {
        if (k >= sim.steps.length) {
          barEls.delete(k);
          fadeRemove(bar);
        }
      }
      /* critical mismatch (analytic, in display units) */
      const mEffC = CORR_RHO / (2 * Math.sqrt(A * FB));
      const mC = (mEffC + Math.sqrt(mEffC * mEffC + 4 * CORR_KNEE * mEffC)) / 2;
      const critDisplay = mC + (1 - PREC_ADAPT) * floorFor(state);
      if (!NARROW) critG.style.transform = `translate(0px, ${yRate(critDisplay)}px)`;

      /* drift line morphs: fixed-sample d interpolated by CSS */
      if (sim.score.length) {
        driftPath.setAttribute("d", resampleD(sim.score, sx, yFor));
        driftPath.setAttribute("stroke", lineColor);
        driftPath.style.opacity = "1";
      } else {
        driftPath.style.opacity = "0";
      }

      /* reward panel */
      const xr = (t) => RX0 + (t / curT) * (RX1 - RX0 - 6);
      const rewColor = sim.divergedAt >= 0 ? "#a33e2d" : "#176b64";

      /* dashed ghosts under the same settings: the other precisions in gray, and fp4
       * always as its two canonical variants — naive vs the full recipe. With NVFP4
       * selected the pair brackets the current flags. */
      const ghostSpecs = [];
      for (const pk of Object.keys(PREC)) {
        if (pk === state.prec || pk === "fp4") continue;
        ghostSpecs.push({ key: pk, label: PREC[pk].label, stroke: "#b9b2a5", labelFill: "#8a847a", patch: { prec: pk } });
      }
      ghostSpecs.push({
        key: "fp4naive", label: "naive FP4", stroke: "#cf9583", labelFill: "#ad6b55",
        patch: { prec: "fp4", four: false, bwd: "hp", layers15: false, se: false },
      });
      ghostSpecs.push({
        key: "fp4tricks", label: "NVFP4 (ours)", stroke: "#7ba9a3", labelFill: "#48796f",
        patch: { prec: "fp4", four: true, bwd: "dq", layers15: true, se: true },
      });
      const ghostLive = new Set();
      const ghostEnds = [];
      for (const spec of ghostSpecs) {
        /* a reference identical to the current flags would just redraw the solid curve */
        if (Object.entries(spec.patch).every(([k, v]) => state[k] === v)) continue;
        const gsim = simulate({ ...state, ...spec.patch });
        if (!gsim.reward.length) continue;
        ghostLive.add(spec.key);
        let ge = ghostEls.get(spec.key);
        if (!ge) {
          const gpath = svg("path", {
            fill: "none", stroke: spec.stroke, "stroke-width": 1.3,
            "stroke-dasharray": "4 3", "stroke-linejoin": "round",
          });
          gpath.style.transition = PATH_TR;
          rewardG.insertBefore(gpath, rewPath);
          const glabel = svg("text", {
            x: -2, y: -5, "text-anchor": "end", "font-size": 10.5, fill: spec.labelFill,
            text: spec.label,
          });
          glabel.style.transition = MOVE_TF;
          glabel.style.cursor = "pointer";
          rewardG.appendChild(glabel);
          /* an invisible fat stroke makes the thin dashed curve clickable */
          const ghit = svg("path", {
            fill: "none", stroke: "transparent", "stroke-width": 18,
            "vector-effect": "non-scaling-stroke", style: "cursor:pointer",
          });
          rewardG.appendChild(ghit);
          const emphasize = (on) => {
            gpath.setAttribute("stroke-width", on ? 2.1 : 1.3);
            glabel.setAttribute("font-weight", on ? "700" : "");
          };
          for (const target of [ghit, glabel]) {
            target.addEventListener("pointerenter", () => emphasize(true));
            target.addEventListener("pointerleave", () => emphasize(false));
            target.addEventListener("click", () => {
              emphasize(false);
              applyReference(spec.patch);
            });
          }
          ge = { path: gpath, hit: ghit, label: glabel };
          ghostEls.set(spec.key, ge);
          fadeIn(gpath, first);
          fadeIn(glabel, first);
        }
        const ghostD = resampleD([{ t: 0, v: 0.45 }, ...gsim.reward], xr, yRew);
        ge.path.setAttribute("d", ghostD);
        ge.hit.setAttribute("d", ghostD);
        ge.path.style.opacity = "1";
        const gl = gsim.reward[gsim.reward.length - 1];
        ghostEnds.push({ ge, x: xr(gl.t), y: yRew(gl.v) });
      }
      for (const [pk, ge] of [...ghostEls]) {
        if (!ghostLive.has(pk)) {
          ghostEls.delete(pk);
          fadeRemove(ge.path);
          fadeRemove(ge.label);
          ge.hit.remove();
        }
      }
      /* nudge ghost labels apart */
      ghostEnds.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ghostEnds.length; i++) {
        if (ghostEnds[i].y - ghostEnds[i - 1].y < 13) ghostEnds[i].y = ghostEnds[i - 1].y + 13;
      }
      for (const g2 of ghostEnds) {
        g2.ge.label.style.transform = `translate(${g2.x}px, ${Math.max(RY0 + 8, g2.y)}px)`;
        g2.ge.label.style.opacity = "1";
      }

      if (sim.reward.length) {
        rewPath.setAttribute("d", resampleD([{ t: 0, v: 0.45 }, ...sim.reward], xr, yRew));
        rewPath.setAttribute("stroke", rewColor);
        rewPath.style.opacity = "1";
        const last = sim.reward[sim.reward.length - 1];
        setAttrs(rewEndDot, { cx: xr(last.t), cy: yRew(last.v), fill: rewColor });
        rewEndLabel.textContent = last.v.toFixed(2);
        rewEndLabel.setAttribute("fill", rewColor);
        rewEndLabel.style.transform = `translate(${xr(last.t) - 6}px, ${Math.max(RY0 + 8, yRew(last.v) - 8)}px)`;
        rewEndDot.style.opacity = "1";
        rewEndLabel.style.opacity = "1";
      } else {
        rewPath.style.opacity = "0";
        rewEndDot.style.opacity = "0";
        rewEndLabel.style.opacity = "0";
      }

      /* divergence region slides with its onset step */
      if (sim.divergedAt >= 0) {
        const x0 = sx(sim.steps[sim.divergedAt].end);
        setAttrs(shadeRect, { x: x0, width: Math.max(0, PLOT_R - x0) });
        shadeRect.style.opacity = "1";
        shadeLabel.style.transform = `translate(${Math.min(x0 + 6, PLOT_R - 112)}px, ${lanesTop - 8}px)`;
        shadeLabel.style.opacity = "1";
      } else {
        shadeRect.style.opacity = "0";
        shadeLabel.style.opacity = "0";
      }

      /* fate: with mean effective mismatch mEff, the dynamics have a fixed point
       * iff rho^2 >= 4*A*fb*mEff^2 — no fixed point (or one above runaway) means
       * this config will diverge even if the window ended before it got there */
      let badge = '<span style="color:#176b64;font-weight:600">stable</span>';
      if (sim.divergedAt >= 0) {
        badge = '<span style="background:#a33e2d;color:#fffdf8;font-weight:600;padding:1px 8px;border-radius:999px">diverged</span>';
      } else if (sim.steps.length) {
        const mEffMean = sim.steps.reduce((a, st) => a + st.mEff, 0) / sim.steps.length;
        const fbEff = FB;
        const rhoNow = CORR_RHO;
        const disc = rhoNow * rhoNow - 4 * A * fbEff * mEffMean * mEffMean;
        const plateau = disc > 0 ? (rhoNow - Math.sqrt(disc)) / (2 * fbEff * mEffMean) : Infinity;
        if (!(plateau < RUNAWAY)) {
          badge = '<span style="color:#b3833f;font-weight:600">will diverge ↗</span>';
        }
      }
      const spikes = sim.steps.filter((st) => st.spiked).length;
      stats.innerHTML =
        `<b>${sim.steps.length}</b> train steps · speed <b>${speedX.toFixed(1)}×</b> vs sync BF16 · ` +
        `utilization <b>${Math.round(sim.util * 100)}%</b> · <b>${sim.discards}</b> discarded · ` +
        `mean staleness <b>${sim.meanStaleness.toFixed(1)}</b> · <b>${spikes}</b> grad spike${spikes === 1 ? "" : "s"} · ` +
        `drift <b>${sim.finalDrift > RUNAWAY ? ">" + RUNAWAY : sim.finalDrift.toFixed(2)}</b> · ${badge}`;
      first = false;
    }

    update();
  };

  const LINE_CHARTS = {
    "mxfp8-dq-hp-grad-norm": {},
    "nvfp4-sgd-raw-reward-2": {},
    "nvfp4-sgd-grad-nrom-2": { compact: true },
    "nvfp4-sgd-kl-loss-2": { compact: true },
    "nvfp4-adam-grad-norm": {},
    "dq-train-time": {},
    "4over6-rollout-time": {},
    "recipe-rollout-time": {},
    "nvfp4-4over6-logprob": {},
    "nvfp4-4over6-kl": {},
    "mxfp8-last-2-4-8": {},
    "mxfp8-f1l6-vs-l6": {},
    "mxfp8-se": {},
    "nvfp4-spikes": {},
    "nvfp4-hai-no-spikes": {},
    "nvfp4-4over6-raw-reward": {},
    "online-nvfp4-judge-scatter": { compact: true },
    "online-nvfp4-judge-grades": { compact: true },
  };
  for (const [id, opts] of Object.entries(LINE_CHARTS)) NV.registerChart(id, opts);

  /* =========================================================
   * Section 3: BF16 memory vs train-rollout mismatch tradeoff
   * ========================================================= */

  NV.registry["bf16-memory-vs-mismatch"] = function (fig) {
    const data = NV.data["bf16-memory-vs-mismatch"];
    if (!data) throw new Error("missing data: bf16-memory-vs-mismatch");
    const body = card(
      fig,
      "What a BF16 exception buys",
      "mean train-rollout logprob difference vs model weight memory, per ablation model",
      "hover over any point"
    );

    /* wide: two panels side by side; phones: stacked, full width */
    const NARROW = (fig.clientWidth || 700) < 520;
    const W = NARROW ? 520 : 952;
    const H = NARROW ? 730 : 442;
    const Y_MAX = 0.03;
    const PANELS = NARROW
      ? { qwen3: { x0: 56, x1: 506, hdr: 44, y0: 58, y1: 320, gbMin: 30, gbMax: 64, ticks: [30, 40, 50, 60] },
          ds48b: { x0: 56, x1: 506, hdr: 396, y0: 410, y1: 672, gbMin: 50, gbMax: 100, ticks: [50, 60, 70, 80, 90, 100] } }
      : { qwen3: { x0: 64, x1: 462, hdr: 44, y0: 58, y1: 368, gbMin: 30, gbMax: 64, ticks: [30, 40, 50, 60] },
          ds48b: { x0: 542, x1: 934, hdr: 44, y0: 58, y1: 368, gbMin: 50, gbMax: 100, ticks: [50, 60, 70, 80, 90, 100] } };
    const ysFor = (P, v) => P.y1 - (v / Y_MAX) * (P.y1 - P.y0);

    const plotWrap = el("div", { class: "viz-plot" });
    const root = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}` });
    const tooltip = el("div", { class: "viz-tooltip" });
    plotWrap.appendChild(root);
    plotWrap.appendChild(tooltip);
    body.appendChild(plotWrap);

    root.appendChild(svg("text", {
      x: PANELS.qwen3.x0, y: 24, "font-size": 10.5, fill: "#6d6860", "letter-spacing": "0.06em",
      text: "MEAN TRAIN-ROLLOUT |\u0394 LOGPROB|, STEPS 200\u20131400",
    }));
    if (!NARROW) {
      for (const v of [0.01, 0.02, 0.03]) {
        root.appendChild(svg("text", { x: PANELS.qwen3.x0 - 8, y: ysFor(PANELS.qwen3, v) + 3, "text-anchor": "end", "font-size": 10.5, fill: "#8a847a", text: v.toFixed(2) }));
      }
    }

    /* per-point label nudges (family:label) */
    const NUDGE = {
      "qwen3:none": [10, -4],
      "qwen3:last 4%": [0, -10],
      "qwen3:last 8%": [10, 12],
      "qwen3:last 15%": [-8, 18],
      "qwen3:all weights BF16": [-12, 2],
      "ds48b:none": [10, -4],
      "ds48b:+ shared experts": NARROW ? [10, 14] : [-10, 4],
      "ds48b:+ SE, last 15%": [-10, 10],
      "ds48b:+ SE, first layer + last 15%": [13, -9],
      "ds48b:all weights BF16": [-12, 2],
    };

    function sparkSvg(spark) {
      const sw = 150;
      const sh = 40;
      const sMax = Math.max(...spark.map((pt) => pt[0]));
      const vMax = 0.035;
      const pts = spark.map((pt) => `${(4 + (pt[0] / sMax) * (sw - 8)).toFixed(1)},${(sh - 4 - (pt[1] / vMax) * (sh - 8)).toFixed(1)}`).join(" ");
      return `<svg width="${sw}" height="${sh}" style="display:block;margin-top:5px">` +
        `<rect x="0" y="0" width="${sw}" height="${sh}" fill="rgba(22,21,19,0.04)" rx="3"/>` +
        `<polyline points="${pts}" fill="none" stroke="#176b64" stroke-width="1.4"/></svg>` +
        `<span style="opacity:.65">per-step |\u0394logprob|, steps 0\u2013${sMax}</span>`;
    }

    function bindPoint(node, fam, pt) {
      node.addEventListener("pointerenter", () => {
        tooltip.innerHTML =
          `<div class="tt-x">${fam.name} \u00b7 ${fam.sub}</div>` +
          `<b>${pt.label}</b> \u00b7 <span style="opacity:.7">${pt.run}</span><br>` +
          `mean |\u0394logprob| <b>${pt.mean.toFixed(4)}</b><br>` +
          `model weights <b>${pt.gb} GB</b>` +
          (pt.gbExtra > 0 ? ` (+${pt.gbExtra} GB vs all-MXFP8 experts)` : " (all-MXFP8 experts)") +
          ` \u00b7 ${pt.fracPct}% in BF16` +
          (pt.label === "+ shared experts" ? "<br>1 shared of 257 experts \u2014 DeepSeek-V3-style block" : "") +
          sparkSvg(pt.spark);
        tooltip.classList.add("on");
        const rect = plotWrap.getBoundingClientRect();
        const r = node.getBoundingClientRect();
        placeTip(plotWrap, tooltip, r.left - rect.left + plotWrap.scrollLeft + r.width / 2, r.top - rect.top + 16, 14);
      });
      node.addEventListener("pointerleave", () => tooltip.classList.remove("on"));
    }

    const FAM_STYLE = {
      qwen3: { stroke: "#9aa4ab", fill: "#7c878f", label: "#6d757c" },
      ds48b: { stroke: "#4f938c", fill: "#176b64", label: "#176b64" },
    };

    for (const fam of data.families) {
      const P = PANELS[fam.key];
      const st = FAM_STYLE[fam.key];
      const xs = (gb) => P.x0 + (Math.log(gb / P.gbMin) / Math.log(P.gbMax / P.gbMin)) * (P.x1 - P.x0);

      root.appendChild(svg("text", {
        x: P.x0, y: P.hdr, "font-size": 11.5, fill: "#16211f", "font-weight": 600,
        text: fam.name,
      }));
      root.appendChild(svg("text", {
        x: P.x0 + fam.name.length * 7.8 + 10, y: P.hdr, "font-size": 10.5, fill: "#8a847a",
        text: fam.sub,
      }));
      for (const v of [0.01, 0.02, 0.03]) {
        root.appendChild(svg("line", { x1: P.x0, y1: ysFor(P, v), x2: P.x1, y2: ysFor(P, v), stroke: "#e4ded2", "stroke-width": 1 }));
        if (NARROW) {
          root.appendChild(svg("text", { x: P.x0 - 8, y: ysFor(P, v) + 3, "text-anchor": "end", "font-size": 10.5, fill: "#8a847a", text: v.toFixed(2) }));
        }
      }
      root.appendChild(svg("line", { x1: P.x0, y1: P.y1, x2: P.x1, y2: P.y1, stroke: "#c4bcae", "stroke-width": 1 }));
      for (const gb of P.ticks) {
        root.appendChild(svg("line", { x1: xs(gb), y1: P.y1, x2: xs(gb), y2: P.y1 + 4, stroke: "#c4bcae", "stroke-width": 1 }));
        root.appendChild(svg("text", { x: xs(gb), y: P.y1 + 16, "text-anchor": "middle", "font-size": 10.5, fill: "#8a847a", text: gb }));
      }
      root.appendChild(svg("text", {
        x: (P.x0 + P.x1) / 2, y: P.y1 + 34, "text-anchor": "middle", "font-size": 10.5, fill: "#6d6860",
        text: "model weight memory (GB, log scale)",
      }));

      const sorted = [...fam.points].sort((a, b) => a.gb - b.gb);
      const path = sorted.map((pt, i) => (i ? "L" : "M") + xs(pt.gb).toFixed(1) + " " + ysFor(P, pt.mean).toFixed(1)).join("");
      root.appendChild(svg("path", { d: path, fill: "none", stroke: st.stroke, "stroke-width": 1.4, opacity: 0.65 }));
      for (const pt of sorted) {
        const cx = xs(pt.gb);
        const cy = ysFor(P, pt.mean);
        const isFullBf16 = pt.label === "all weights BF16";
        const dot = isFullBf16
          ? svg("circle", { cx, cy, r: 4.5, fill: "#faf7f0", stroke: st.fill, "stroke-width": 2, style: "cursor:default" })
          : svg("circle", { cx, cy, r: 4.5, fill: st.fill, stroke: "#faf7f0", "stroke-width": 1.5, style: "cursor:default" });
        root.appendChild(dot);
        const [dx, dy] = NUDGE[`${fam.key}:${pt.label}`] || [10, 4];
        root.appendChild(svg("text", {
          x: cx + dx, y: cy + dy, "font-size": 10.5, fill: st.label,
          "text-anchor": dx < 0 ? "end" : "start", text: pt.label,
        }));
        bindPoint(dot, fam, pt);
      }
    }

    body.appendChild(
      el("figcaption", {
        text:
          "Means over train steps 200\u20131400 of the underlying per-step mismatch curves (they are flat \u2014 hover over any point for its raw curve). " +
          "x is total weight memory: MoE expert weights at the ablations' MXFP8 rollout precision (~1.03 bytes/param; BF16 " +
          "exceptions at 2 \u2014 the recipe quantizes only expert weights) plus the always-BF16 rest (attention, embeddings, " +
          "router). Qwen3-30B-A3B sizes come from its public config (29.0B expert + 1.5B other params); the DeepSeek-style " +
          "model's from the post's DeepSeek-V3 ratios \u2014 experts 97% of its 48B parameters, shared experts 1 of 257 per " +
          "block, and its \u201clast 15%\u201d is 6 of 40 layers. Shared experts are nearly free (+0.2 GB) and buy the steepest " +
          "single drop; the first layer adds ~1 GB for no gain (its point sits above the last-15% point); the hollow points are " +
          "the all-BF16 runs (Qwen3's from the final-recipe ablation family) \u2014 nearly twice the memory for the mismatch " +
          "floor no expert-weight exception reaches.",
      })
    );
  };

  /* =========================================================
   * Final recipe: precision format vs mismatch at fixed exceptions
   * ========================================================= */

  NV.registry["recipe-memory-vs-mismatch"] = function (fig) {
    const data = NV.data["recipe-memory-vs-mismatch"];
    if (!data) throw new Error("missing data: recipe-memory-vs-mismatch");
    const body = card(
      fig,
      "Climbing the precision ladder, at no extra memory cost",
      "mean train-rollout logprob difference vs model weight memory \u00b7 Qwen3-30B-A3B recipe ablations, last 8 layers in BF16",
      "hover over any point"
    );

    const NARROW = (fig.clientWidth || 700) < 520;
    const W = NARROW ? 560 : 952;
    const H = 442;
    const T = 52;
    const B = 368;
    const X0 = NARROW ? 56 : 64;
    const X1 = W - 18;
    const GB_MIN = 20;
    const GB_MAX = 65;
    const Y_MAX = 0.04;
    const xs = (gb) => X0 + (Math.log(gb / GB_MIN) / Math.log(GB_MAX / GB_MIN)) * (X1 - X0);
    const ys = (v) => B - (v / Y_MAX) * (B - T);

    const plotWrap = el("div", { class: "viz-plot" });
    const root = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}` });
    const tooltip = el("div", { class: "viz-tooltip" });
    plotWrap.appendChild(root);
    plotWrap.appendChild(tooltip);
    body.appendChild(plotWrap);

    root.appendChild(svg("text", {
      x: X0, y: 24, "font-size": 10.5, fill: "#6d6860", "letter-spacing": "0.06em",
      text: "MEAN TRAIN-ROLLOUT |\u0394 LOGPROB|, STEPS 200\u2013398",
    }));
    root.appendChild(svg("text", {
      x: X1, y: NARROW ? 40 : 24, "text-anchor": "end", "font-size": 10.5, fill: "#8a847a",
      text: "\u25c6 MXFP8 train + infer \u00b7 bars span replicate runs",
    }));
    for (const v of [0.01, 0.02, 0.03, 0.04]) {
      root.appendChild(svg("line", { x1: X0, y1: ys(v), x2: X1, y2: ys(v), stroke: "#e4ded2", "stroke-width": 1 }));
      root.appendChild(svg("text", { x: X0 - 8, y: ys(v) + 3, "text-anchor": "end", "font-size": 10.5, fill: "#8a847a", text: v.toFixed(2) }));
    }
    root.appendChild(svg("line", { x1: X0, y1: B, x2: X1, y2: B, stroke: "#c4bcae", "stroke-width": 1 }));
    for (const gb of [20, 30, 40, 50, 60]) {
      root.appendChild(svg("line", { x1: xs(gb), y1: B, x2: xs(gb), y2: B + 4, stroke: "#c4bcae", "stroke-width": 1 }));
      root.appendChild(svg("text", { x: xs(gb), y: B + 16, "text-anchor": "middle", "font-size": 10.5, fill: "#8a847a", text: gb }));
    }
    root.appendChild(svg("text", {
      x: (X0 + X1) / 2, y: B + 34, "text-anchor": "middle", "font-size": 10.5, fill: "#6d6860",
      text: "model weight memory (GB, log scale)",
    }));

    const FAMILY = {
      "all BF16": "#176b64",
      "MXFP8": "#7c878f",
      "NVFP4": "#a33e2d",
      "NVFP4 + 4/6": "#176b64",
    };
    const MODE_NAME = {
      q: "backward variants pooled",
      ti: "MXFP8 train + infer",
      ref: "all weights BF16",
    };

    /* same memory = same x — backward variants are pooled, so at most the
       train+infer point shares an x with its rung */
    for (const pt of data.points) pt.__x = xs(pt.gb);

    /* frontier through the format rungs: BF16 -> MXFP8 -> NVFP4 */
    const rungOf = (fam) => data.points.find((pt) => pt.family === fam && (pt.mode === "q" || pt.mode === "ref"));
    const frontier = ["all BF16", "MXFP8", "NVFP4"].map(rungOf);
    root.appendChild(svg("path", {
      d: frontier.map((pt, i) => (i ? "L" : "M") + pt.__x.toFixed(1) + " " + ys(pt.mean).toFixed(1)).join(""),
      fill: "none", stroke: "#b9b2a5", "stroke-width": 1.3, opacity: 0.8,
    }));

    function sparkSvg(spark) {
      const sw = 150;
      const sh = 40;
      const sMax = Math.max(...spark.map((pt) => pt[0]));
      const vMax = 0.04;
      const pts = spark.map((pt) => `${(4 + (pt[0] / sMax) * (sw - 8)).toFixed(1)},${(sh - 4 - (pt[1] / vMax) * (sh - 8)).toFixed(1)}`).join(" ");
      return `<svg width="${sw}" height="${sh}" style="display:block;margin-top:5px">` +
        `<rect x="0" y="0" width="${sw}" height="${sh}" fill="rgba(22,21,19,0.04)" rx="3"/>` +
        `<polyline points="${pts}" fill="none" stroke="#176b64" stroke-width="1.4"/></svg>` +
        `<span style="opacity:.65">per-step |\u0394logprob|, one replicate</span>`;
    }

    /* the train+infer diamond shares its rung's (x, y) almost exactly —
       draw it first and larger so it reads as corners behind the circle */
    const drawOrder = [...data.points].sort((a, b) => (a.mode === "ti" ? 0 : 1) - (b.mode === "ti" ? 0 : 1));
    for (const pt of drawOrder) {
      const color = FAMILY[pt.family];
      const cx = pt.__x;
      const cy = ys(pt.mean);
      /* replicate spread: a bar from the lowest to the highest run mean */
      if (pt.hi - pt.lo > 1e-9) {
        const yLo = ys(pt.lo);
        const yHi = ys(pt.hi);
        root.appendChild(svg("line", { x1: cx, y1: yHi, x2: cx, y2: yLo, stroke: color, "stroke-width": 1.2, opacity: 0.65 }));
        for (const yy of [yHi, yLo]) {
          root.appendChild(svg("line", { x1: cx - 3.5, y1: yy, x2: cx + 3.5, y2: yy, stroke: color, "stroke-width": 1.2, opacity: 0.65 }));
        }
      }
      let dot;
      if (pt.mode === "ti") {
        dot = svg("rect", {
          x: cx - 5, y: cy - 5, width: 10, height: 10,
          transform: `rotate(45 ${cx} ${cy})`,
          fill: color, stroke: "#f5f1e8", "stroke-width": 1.2, style: "cursor:default",
        });
      } else if (pt.mode === "ref") {
        dot = svg("circle", { cx, cy, r: 4.5, fill: "#f5f1e8", stroke: color, "stroke-width": 1.8, style: "cursor:default" });
      } else {
        dot = svg("circle", { cx, cy, r: 4.5, fill: color, stroke: "#f5f1e8", "stroke-width": 1.5, style: "cursor:default" });
      }
      root.appendChild(dot);
      dot.addEventListener("pointerenter", () => {
        const runsHtml = Object.entries(pt.runs)
          .map(([name, m]) => `<span style="opacity:.7">${name}</span> ${m.toFixed(4)}`)
          .join("<br>");
        tooltip.innerHTML =
          `<div class="tt-x">${pt.family} \u00b7 ${MODE_NAME[pt.mode]}</div>` +
          `model weights <b>${pt.gb} GB</b> \u00b7 mean |\u0394logprob| <b>${pt.mean.toFixed(4)}</b><br>` +
          (pt.family.includes("4/6") ? `same storage as NVFP4 \u2014 4/6 changes scale selection, not the format<br>` : "") +
          runsHtml + sparkSvg(pt.spark);
        tooltip.classList.add("on");
        const rect = plotWrap.getBoundingClientRect();
        const r = dot.getBoundingClientRect();
        placeTip(plotWrap, tooltip, r.left - rect.left + plotWrap.scrollLeft + r.width / 2, r.top - rect.top + 16, 14);
      });
      dot.addEventListener("pointerleave", () => tooltip.classList.remove("on"));
    }

    /* one label per family, at the family's dot centroid */
    const FAM_NUDGE = {
      "NVFP4": [0, -14, "middle"],
      "NVFP4 + 4/6": [4, 26, "middle"],
      "MXFP8": [0, -14, "middle"],
      "all BF16": [-14, 4, "end"],
    };
    const famPts = {};
    for (const pt of data.points) (famPts[pt.family] = famPts[pt.family] || []).push(pt);
    for (const [fam, arr] of Object.entries(famPts)) {
      const [dx, dy, anchor] = FAM_NUDGE[fam];
      const cx = arr.reduce((a, pt) => a + pt.__x, 0) / arr.length + dx;
      const cy = arr.reduce((a, pt) => a + ys(pt.mean), 0) / arr.length + dy;
      root.appendChild(svg("text", {
        x: cx, y: cy, "font-size": 10.5, fill: FAMILY[fam], "font-weight": 600,
        "text-anchor": anchor, text: fam,
      }));
    }

  };

  /* =========================================================
   * Recipe overview: forward error x backward mismatch quadrants
   * ========================================================= */

  NV.registry["recipe-quadrants"] = function (fig) {
    const body = card(
      fig,
      "Minimizing gradient mismatch & policy error",
      "each technique classified by which error source it addresses",
      "click a cell to jump to its section"
    );

    const NARROW = (fig.clientWidth || 700) < 520;
    const TEAL = "#176b64";
    const RUST = "#a33e2d";
    const CELL = NARROW ? 170 : 190;
    const GAP = NARROW ? 12 : 16;
    const PAD = 0;
    const PANEL = 2 * CELL + GAP;
    const W = NARROW ? 470 : 940;
    const PX = NARROW ? 44 : (W - PANEL) / 2 + 14; /* panel x, with room for the left-side label */
    const PT = 18;
    const H = PT + PANEL + 46;
    const cellX = (col) => PX + PAD + col * (CELL + GAP);
    const cellY = (row) => PT + PAD + row * (CELL + GAP); /* row 0 = top */

    const plotWrap = el("div", { class: "viz-plot" });
    const root = svg("svg", { class: "viz-svg", viewBox: `0 0 ${W} ${H}` });
    const tooltip = el("div", { class: "viz-tooltip" });
    plotWrap.appendChild(root);
    plotWrap.appendChild(tooltip);
    body.appendChild(plotWrap);

    /* axis labels frame the grid: forward error along the bottom, backward
       mismatch reading upward along the left edge */
    root.appendChild(svg("text", {
      x: PX + PANEL / 2, y: PT + PANEL + 32, "text-anchor": "middle", "font-size": 12,
      "font-weight": 700, "letter-spacing": "0.09em", fill: "#44403a",
      text: "POLICY QUANTIZATION ERROR (FORWARD)",
    }));
    const midY = PT + PANEL / 2;
    root.appendChild(svg("text", {
      x: PX - 22, y: midY, "text-anchor": "middle", "font-size": 12, "font-weight": 700,
      "letter-spacing": "0.09em", fill: "#44403a",
      transform: `rotate(-90 ${PX - 22} ${midY})`,
      text: "GRADIENT MISMATCH (BACKWARD)",
    }));

    function jump(headingText) {
      const h = [...document.querySelectorAll("h2")].find((x) => x.textContent.trim().startsWith(headingText));
      if (h) h.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    const CELLS = {
      neither: {
        col: 1, row: 0,
        tip: "Addresses neither error source: quantize and hope. This is where naive quantization lives \u2014 " +
          "trainer-sampler mismatch plus occasional gradient spikes.",
      },
      fwd: {
        col: 0, row: 0,
        tip: "What the sampler generates and the trainer scores must be the same quantized policy: " +
          "per-token scales keep activation quantization local, 4/6 cuts the worst-case FP4 error, " +
          "and the quantizer contract is bit-exact across training and rollout.",
        target: "Four-Over-Six",
      },
      bwd: {
        col: 1, row: 1,
        tip: "The backward pass must differentiate the same function the quantized forward evaluated: " +
          "dequantized backward keeps the forward's clipping and rounding decisions while the GEMM stays high precision.",
        target: "Improving Gradient Stability",
      },
      both: {
        col: 0, row: 1,
        tip: "Keeping the last layers and the always-active shared experts in BF16 removes both their " +
          "forward quantization error and their backward mismatch at a small memory cost.",
        target: "Selective Layer Precisions",
      },
    };

    function bindCell(node, key, title) {
      node.addEventListener("pointerenter", (evt) => {
        tooltip.innerHTML = `<div class="tt-x">${title}</div>` + CELLS[key].tip;
        tooltip.classList.add("on");
        const rect = plotWrap.getBoundingClientRect();
        const r = node.getBoundingClientRect();
        placeTip(plotWrap, tooltip, (r.left + r.right) / 2 - rect.left + plotWrap.scrollLeft, r.top - rect.top + 20, 14);
      });
      node.addEventListener("pointerleave", () => tooltip.classList.remove("on"));
      if (CELLS[key].target) {
        node.style.cursor = "pointer";
        node.setAttribute("tabindex", "0");
        node.setAttribute("role", "link");
        node.setAttribute("aria-label", `${title}: jump to the ${CELLS[key].target} section`);
        node.addEventListener("click", () => jump(CELLS[key].target));
        node.addEventListener("keydown", (evt) => {
          if (evt.key === "Enter" || evt.key === " ") {
            evt.preventDefault();
            jump(CELLS[key].target);
          }
        });
      }
    }

    function cell(key, style) {
      const c = CELLS[key];
      const x = cellX(c.col);
      const y = cellY(c.row);
      const r = svg("rect", {
        x, y, width: CELL, height: CELL, rx: 18,
        fill: style.fill, stroke: style.stroke, "stroke-width": 2,
        ...(style.dashed ? { "stroke-dasharray": "6 5" } : {}),
      });
      root.appendChild(r);
      const midX = x + CELL / 2;
      let ty = y + CELL / 2 - (style.title.length - 1) * 8.5 + 4;
      for (const line of style.title) {
        root.appendChild(svg("text", {
          x: midX, y: ty, "text-anchor": "middle", "font-size": 12, "font-weight": 700,
          "letter-spacing": "0.06em", fill: style.titleColor, "pointer-events": "none", text: line,
        }));
        ty += 17;
      }
      bindCell(r, key, style.title.join(" ").toLowerCase());
      return r;
    }

    cell("bwd", {
      fill: "hsl(9 50% 94%)", stroke: RUST, titleColor: RUST,
      title: ["GRADIENT", "MISMATCH FIX"],
    });
    cell("both", {
      fill: "hsl(40 10% 93%)", stroke: "#8a847a", titleColor: "#3f3a33",
      title: ["SELECTIVE", "HIGH PRECISION"],
    });
    cell("neither", {
      fill: "hsl(40 12% 95%)", stroke: "#c9c1b2", titleColor: "#a8a296", dashed: true,
      title: ["NAIVE", "QUANTIZATION"],
    });
    cell("fwd", {
      fill: "hsl(172 30% 92%)", stroke: TEAL, titleColor: TEAL,
      title: ["QUANTIZATION", "ERROR FIXES"],
    });

    body.appendChild(
      el("p", {
        class: "viz-sr",
        text:
          "Two-by-two grid of the recipe's techniques. Columns: whether a technique addresses policy " +
          "quantization error in the forward pass; rows: whether it addresses gradient mismatch in the " +
          "backward pass. Top left: quantization error fixes (per-token scales, 4/6, the bit-exact contract). " +
          "Top right: naive quantization, which addresses neither. Bottom left: selective high precision. " +
          "Bottom right: the gradient mismatch fix (dequantized backward). The three technique cells link " +
          "to their sections.",
      })
    );

  };

  /* =========================================================
   * NVFP4 as a 16x16 tile in isometric 3D
   * ========================================================= */

  NV.registry["nvfp4-iso-tile"] = function (fig) {
    const body = card(
      fig,
      "NVFP4 Visualization",
      "height = value \u00b7 each block of 16 shares one FP8 scale \u00b7 one FP32 scale per tensor",
      "hover over a block, a scale, or the memory bars"
    );

    const NARROW = (fig.clientWidth || 700) < 520;
    /* phones get a shallower tile (5 rows, still 16-value blocks), chunkier
       tiles, larger labels, and the memory panel docked below the scene */
    const W = NARROW ? 560 : 940;
    const VIEW_TOP = NARROW ? 45 : 70; /* viewBox hugs the scene */
    const H = NARROW ? 575 : 340;
    const TEAL = "#176b64";
    const RUST = "#a33e2d";
    const NR = NARROW ? 5 : 11; /* rows, each sharing one FP8 scale */
    const NC = 16; /* values per row */
    const SCALE_COL = NC + 1.2; /* scale pillars sit past a clear gap */
    const TW = NARROW ? 17 : 15; /* half tile width  */
    const TH = TW / 2;           /* half tile height */
    const HMAX = NARROW ? 110 : 120;
    const X0 = NARROW ? 105 : 300;
    const Y0 = NARROW ? 140 : 150;
    const LFS = NARROW ? 15 : 11; /* scene label font size */
    const px = (c, r) => X0 + (c - r) * TW;
    const py = (c, r) => Y0 + (c + r) * TH;
    const REDUCE = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const MORPH = REDUCE ? "" : "d 0.45s ease, fill 0.45s ease";
    const FADE = REDUCE ? "" : "opacity 0.45s ease";
    const quantDecor = []; /* scale pillars + labels: fade out on the unquantized view */

    const plotWrap = el("div", { class: "viz-plot" });
    const root = svg("svg", { class: "viz-svg", viewBox: `0 ${VIEW_TOP} ${W} ${H}` });
    const tooltip = el("div", { class: "viz-tooltip" });
    plotWrap.appendChild(root);
    plotWrap.appendChild(tooltip);
    body.appendChild(plotWrap);
    /* the quantize/dequantize math, always visible, updated by hovering a bar or scale */
    const mathStrip = el("div", {
      class: "viz-sub",
      style: "text-align:center;margin-top:8px;font-variant-numeric:tabular-nums;font-size:" + ((fig.clientWidth || 700) < 520 ? "11px" : "13.5px"),
    });
    body.appendChild(mathStrip);

    /* --- data: per-row scale (wide dynamic range), original values + 4-bit codes --- */
    const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
    const E4M3 = (() => {
      const vals = [];
      for (let m = 1; m < 8; m++) vals.push((m / 8) * Math.pow(2, -6));
      for (let e = 1; e <= 15; e++) for (let m = 0; m < 8; m++) {
        if (e === 15 && m === 7) continue;
        vals.push((1 + m / 8) * Math.pow(2, e - 7));
      }
      return vals;
    })();
    const snap = (set, v) => {
      let best = set[0];
      for (const g of set) if (Math.abs(g - v) < Math.abs(best - v)) best = g;
      return best;
    };
    let seed = 11;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0), seed / 4294967296);
    const rows = [];
    for (let r = 0; r < NR; r++) {
      const amax = 0.03 + Math.pow(rnd(), 2.2) * 0.95; /* some rows loud, some quiet */
      const orig = [];
      const signs = [];
      for (let c = 0; c < NC; c++) {
        const v = Math.pow(rnd(), 1.4) * amax;
        rnd(); /* draw kept so the seeded landscape is unchanged */
        const sgn = 1; /* shown unsigned: heights carry the story */
        orig.push(sgn * v);
        signs.push(sgn);
      }
      rows.push({ orig, signs, amax });
    }
    /* the FP32 tensor scale comes from the tensor's amax; each block's E4M3
       scale is stored relative to it, so snap the RELATIVE quantity */
    const maxVal = Math.max(...rows.map((rw) => rw.amax));
    const T_SCALE = maxVal / 6;
    for (const rw of rows) {
      rw.rel = snap(E4M3, rw.amax / 6 / T_SCALE);
      rw.scale = rw.rel * T_SCALE;
      rw.codes = rw.orig.map((v) => snap(E2M1, v / rw.scale));
    }
    const LEVELS = [0.5, 1, 1.5, 2, 3, 4, 6];

    /* --- isometric faces --- */
    function shade(base, k, sgn) { /* k: 1 top, .78 left, .6 right; hue by sign */
      return sgn < 0
        ? `hsl(215 ${28 + base * 26}% ${88 - base * 40 * k}%)`
        : `hsl(172 ${26 + base * 22}% ${92 - base * 40 * k}%)`;
    }
    const faceD = (x, y, h) => ({
      top: `M${x} ${y - h} L${x + TW} ${y - h + TH} L${x} ${y - h + 2 * TH} L${x - TW} ${y - h + TH} Z`,
      left: `M${x - TW} ${y + TH - h} L${x} ${y + 2 * TH - h} L${x} ${y + 2 * TH} L${x - TW} ${y + TH} Z`,
      right: `M${x + TW} ${y + TH - h} L${x} ${y + 2 * TH - h} L${x} ${y + 2 * TH} L${x + TW} ${y + TH} Z`,
    });
    const tickD = (x, y, hL) => `M${x - TW} ${y + TH - hL} L${x} ${y + 2 * TH - hL} L${x + TW} ${y + TH - hL}`;

    const sceneG = svg("g", {});
    root.appendChild(sceneG);

    /* --- the global FP32 tensor scale: behind the block-scale column --- */
    const GOLD = { top: "hsl(42 55% 78%)", left: "hsl(42 42% 62%)", right: "hsl(42 42% 50%)", ink: "#8a6d2f" };
    {
      const gc = SCALE_COL;
      const gr = -1.6;
      const gh = HMAX * 1.04; /* the tensor scale's range covers the loudest block */
      const gx = px(gc, gr);
      const gy = py(gc, gr);
      const d = faceD(gx, gy, gh);
      const gG = svg("g", { style: "cursor:default" });
      const gFaces = {
        left: svg("path", { d: d.left, fill: GOLD.left }),
        right: svg("path", { d: d.right, fill: GOLD.right }),
        top: svg("path", { d: d.top, fill: GOLD.top, stroke: GOLD.ink, "stroke-width": 0.8 }),
      };
      for (const f of [gFaces.left, gFaces.right, gFaces.top]) gG.appendChild(f);
      gG.style.transition = FADE;
      quantDecor.push(gG);
      gG.addEventListener("pointerenter", () => {
        for (const f of [gFaces.left, gFaces.right, gFaces.top]) {
          f.setAttribute("stroke", GOLD.ink);
          f.setAttribute("stroke-width", f === gFaces.top ? 1.4 : 0.9);
        }
        showMathTensor();
      });
      gG.addEventListener("pointerleave", () => {
        gFaces.left.removeAttribute("stroke");
        gFaces.left.removeAttribute("stroke-width");
        gFaces.right.removeAttribute("stroke");
        gFaces.right.removeAttribute("stroke-width");
        gFaces.top.setAttribute("stroke-width", 0.8);
      });
      sceneG.appendChild(gG); /* appended first: the rust column paints in front */
      const tensorLbl = svg("text", {
        x: gx, y: gy + 2 * TH - gh - 12, "text-anchor": "middle", "font-size": LFS, "font-weight": 600,
        fill: GOLD.ink, text: "tensor scale",
      });
      tensorLbl.style.transition = FADE;
      quantDecor.push(tensorLbl);
      root.appendChild(tensorLbl);
    }

    /* --- bars, painter order back to front; persistent for quantized<->original morphs --- */
    const rowGs = [];
    for (let r = 0; r < NR; r++) {
      const g = svg("g", {});
      g.style.transition = "opacity 0.15s ease";
      rowGs.push(g);
    }
    const cells = [];
    for (let r = 0; r < NR; r++) {
      for (let c = 0; c < NC; c++) cells.push({ r, c, kind: "v" });
      cells.push({ r, c: SCALE_COL, kind: "s" }); /* the row's scale pillar, past the row end */
    }
    cells.sort((a, b) => a.r + a.c - (b.r + b.c));
    const valueBars = []; /* {r, c, faces} for the mode toggle */
    const scalePillars = []; /* {r, faces, ticks} — hover targets for the block-scale math */
    for (const cell of cells) {
      const rw = rows[cell.r];
      const x = px(cell.c, cell.r);
      const y = py(cell.c, cell.r);
      if (cell.kind === "v") {
        const g = svg("g", {});
        const faces = {
          left: svg("path", {}),
          right: svg("path", {}),
          top: svg("path", { stroke: "hsl(172 30% 55%)", "stroke-width": 0.5 }),
        };
        for (const f of [faces.left, faces.right, faces.top]) {
          f.style.transition = MORPH;
          g.appendChild(f);
        }
        /* the block's representable levels: contour lines on the faces,
           spacing stretched by the row's scale (subtle; dropped on phones
           where they read as a floating lattice) */
        const ticks = [];
        for (const L of NARROW ? [] : LEVELS) {
          const hL = ((L * rw.scale) / maxVal) * HMAX;
          const tk = svg("path", {
            d: tickD(x, y, hL),
            fill: "none", stroke: "hsl(172 45% 28%)", "stroke-width": 0.6, "stroke-opacity": 0.18,
            "pointer-events": "none",
          });
          tk.style.transition = "stroke-opacity 0.12s ease, stroke-width 0.12s ease" + (REDUCE ? "" : ", opacity 0.45s ease");
          ticks.push(tk);
          g.appendChild(tk);
        }
        rowGs[cell.r].appendChild(g);
        valueBars.push({ r: cell.r, c: cell.c, x, y, faces, g, ticks, sign: rw.signs[cell.c] });
      } else {
        const h = Math.max(((rw.scale * 6) / maxVal) * HMAX, 3);
        const d = faceD(x, y, h);
        const g = svg("g", {});
        const faces = {
          left: svg("path", { d: d.left, fill: "hsl(9 42% 72%)" }),
          right: svg("path", { d: d.right, fill: "hsl(9 45% 60%)" }),
          top: svg("path", { d: d.top, fill: "hsl(9 50% 82%)", stroke: RUST, "stroke-width": 0.7 }),
        };
        for (const f of [faces.left, faces.right, faces.top]) g.appendChild(f);
        const ticks = [];
        for (const L of LEVELS) {
          const hL = h * (L / 6);
          const tk = svg("path", {
            d: tickD(x, y, hL),
            fill: "none", stroke: "hsl(9 55% 34%)", "stroke-width": 0.7, "stroke-opacity": 0.45,
            "pointer-events": "none",
          });
          ticks.push(tk);
          g.appendChild(tk);
        }
        g.style.transition = FADE;
        quantDecor.push(g);
        rowGs[cell.r].appendChild(g);
        scalePillars.push({ r: cell.r, g, faces, ticks });
      }
    }
    for (const g of rowGs) sceneG.appendChild(g);

    /* label the rust column: the per-block FP8 scale */
    {
      const bx = px(SCALE_COL, NR - 1);
      const by = py(SCALE_COL, NR - 1);
      const bh = ((rows[NR - 1].scale * 6) / maxVal) * HMAX;
      /* narrow scenes have no clear air beside or below the column (the memory
         panel docks underneath); float the label above the tallest pillar,
         mirroring the tensor-scale label */
      let topY = Infinity;
      let topX = 0;
      for (let r = 0; r < NR; r++) {
        const h = ((rows[r].scale * 6) / maxVal) * HMAX;
        const t = py(SCALE_COL, r) + TH - h;
        if (t < topY) {
          topY = t;
          topX = px(SCALE_COL, r) + TW / 2;
        }
      }
      const bLine = NARROW
        ? svg("line", {
            x1: topX, y1: topY - 4, x2: topX + 8, y2: topY - 20,
            stroke: RUST, "stroke-width": 1, "stroke-dasharray": "3 3",
          })
        : svg("line", {
            x1: bx + TW + 4, y1: by + TH - bh / 2, x2: bx + TW + 24, y2: by + TH - bh / 2,
            stroke: RUST, "stroke-width": 1, "stroke-dasharray": "3 3",
          });
      const bText = NARROW
        ? svg("text", {
            x: topX + 10, y: topY - 28, "text-anchor": "middle", "font-size": LFS,
            "font-weight": 600, fill: RUST, text: "block scale",
          })
        : svg("text", {
            x: bx + TW + 28, y: by + TH - bh / 2 + 3, "font-size": LFS, "font-weight": 600,
            fill: RUST, text: "block scale",
          });
      for (const n of [bLine, bText]) {
        n.style.transition = FADE;
        quantDecor.push(n);
        root.appendChild(n);
      }
    }

    /* label the value field: lower left, balancing the block-scale label,
       with a dashed leader to the field's front-middle tile */
    /* keep the longer "unquantized weights" string inside the narrow viewBox */
    const labelX = Math.max(px(0, NR - 1) + 30, NARROW ? 86 : 0);
    const labelY = py(0, NR - 1) + 2 * TH + 68;
    {
      const mx = px(NC / 2 - 0.5, NR - 1); /* the field's front-middle tile */
      const my = py(NC / 2 - 0.5, NR - 1) + 2 * TH;
      root.appendChild(svg("line", {
        x1: mx - 3, y1: my + 4, x2: labelX + 34, y2: labelY - 5,
        stroke: TEAL, "stroke-width": 1, "stroke-dasharray": "3 3",
      }));
    }
    const fieldLabel = svg("text", {
      x: labelX, y: labelY,
      "text-anchor": "middle", "font-size": LFS, "font-weight": 600, fill: TEAL,
      "pointer-events": "none",
    });
    root.appendChild(fieldLabel);

    /* quantized <-> original heights */
    let showOriginal = false;
    function applyHeights() {
      for (const vb of valueBars) {
        const rw = rows[vb.r];
        const val = showOriginal ? Math.abs(rw.orig[vb.c]) : rw.codes[vb.c] * rw.scale;
        const h = (val / maxVal) * HMAX;
        const base = Math.min(1, val / maxVal);
        const d = faceD(vb.x, vb.y, h);
        if (h < 0.5) {
          vb.faces.left.setAttribute("d", d.left);
          vb.faces.right.setAttribute("d", d.right);
          vb.faces.top.setAttribute("d", d.top);
          vb.faces.left.setAttribute("fill", "hsl(40 12% 90%)");
          vb.faces.right.setAttribute("fill", "hsl(40 12% 90%)");
          vb.faces.top.setAttribute("fill", "hsl(40 12% 90%)");
        } else {
          vb.faces.left.setAttribute("d", d.left);
          vb.faces.right.setAttribute("d", d.right);
          vb.faces.top.setAttribute("d", d.top);
          vb.faces.left.setAttribute("fill", shade(base, 0.78, vb.sign));
          vb.faces.right.setAttribute("fill", shade(base, 0.6, vb.sign));
          vb.faces.top.setAttribute("fill", shade(base, 1, vb.sign));
          vb.faces.top.setAttribute("stroke", vb.sign < 0 ? "hsl(215 35% 48%)" : "hsl(172 30% 55%)");
        }
        /* the E2M1 level contours only mean something on the quantized view */
        for (const tk of vb.ticks) tk.style.opacity = showOriginal ? "0" : "1";
      }
      /* the scales describe the quantized format — fade them with the view, in place */
      for (const n of quantDecor) {
        n.style.opacity = showOriginal ? "0" : "1";
        n.style.pointerEvents = showOriginal ? "none" : "";
      }
      fieldLabel.textContent = showOriginal ? "unquantized weights" : "quantized values";
    }
    applyHeights();

    /* hovering a row's bars (or its scale pillar) highlights that row —
       the painted faces are the pointer targets, so 3D occlusion decides */
    const dim = '<span style="color:#8a847a">';
    const lblV = '<span style="color:#3f8a82">';
    const lblB = '<span style="color:#b06a5b">';
    const lblT = '<span style="color:#a08549">';
    /* both directions, stacked: quantize rounds the weight onto the E2M1 grid,
       dequantize reconstructs the value the GEMM sees (prefixes pad to align in mono) */
    function stripLines(qLine, dqLine) {
      mathStrip.innerHTML =
        `<span style="display:inline-block;text-align:left">` +
        `<span style="display:block">${dim}quantize&nbsp;&nbsp;&nbsp;</span>${qLine}</span>` +
        `<span style="display:block;margin-top:3px">${dim}dequantize&nbsp;</span>${dqLine}</span>` +
        `</span>`;
    }
    const blockHtml = (rw) => `<b style="color:#a33e2d">${rw.rel.toPrecision(3)}</b> ${lblB}(block scale)</span>`;
    const tensorHtml = () => `<b style="color:#8a6d2f">${T_SCALE.toPrecision(3)}</b> ${lblT}(tensor scale)</span>`;
    function showMath(vb) {
      const rw = rows[vb.r];
      const code = rw.codes[vb.c];
      const vCol = vb.sign < 0 ? "hsl(215 55% 35%)" : "#176b64";
      const sgn = vb.sign < 0 ? "\u2212" : "";
      const codeHtml = `<b style="color:${vCol}">${sgn}${code}</b> ${lblV}(value)</span>`;
      stripLines(
        `${codeHtml} ${dim}=</span> ${dim}round(</span>` +
          `<b style="color:${vCol}">${sgn}${Math.abs(rw.orig[vb.c]).toPrecision(3)}</b> ${lblV}(weight)</span>` +
          ` ${dim}\u00f7</span> ${blockHtml(rw)} ${dim}\u00f7</span> ${tensorHtml()}${dim})</span>`,
        `<b style="color:${vCol}">${sgn}${(code * rw.scale).toPrecision(3)}</b>` +
          ` ${dim}=</span> ${codeHtml} ${dim}\u00d7</span> ${blockHtml(rw)} ${dim}\u00d7</span> ${tensorHtml()}`
      );
    }
    /* hovering a scale: the equations go symbolic where the hovered scale
       doesn't pin a number down \u2014 x is any value, w any weight, y any block scale */
    const X_SYM = `<b style="color:#176b64">x</b>`;
    const W_SYM = `<b style="color:#176b64">w</b>`;
    const Y_SYM = `<b style="color:#a33e2d">y</b>`;
    function showMathBlock(r) {
      const rw = rows[r];
      stripLines(
        `${X_SYM} ${lblV}(value)</span> ${dim}=</span> ${dim}round(</span>${W_SYM} ${lblV}(weight)</span>` +
          ` ${dim}\u00f7</span> ${blockHtml(rw)} ${dim}\u00f7</span> ${tensorHtml()}${dim})</span>`,
        `${X_SYM} ${dim}\u00d7</span> <b style="color:#3f3a33">${rw.scale.toPrecision(3)}</b>` +
          ` ${dim}=</span> ${X_SYM} ${lblV}(value)</span> ${dim}\u00d7</span> ${blockHtml(rw)} ${dim}\u00d7</span> ${tensorHtml()}`
      );
    }
    function showMathTensor() {
      stripLines(
        `${X_SYM} ${lblV}(value)</span> ${dim}=</span> ${dim}round(</span>${W_SYM} ${lblV}(weight)</span>` +
          ` ${dim}\u00f7</span> ${Y_SYM} ${lblB}(block scale)</span> ${dim}\u00f7</span> ${tensorHtml()}${dim})</span>`,
        `${X_SYM}${Y_SYM} ${dim}\u00d7</span> <b style="color:#8a6d2f">${T_SCALE.toPrecision(3)}</b>` +
          ` ${dim}=</span> ${X_SYM} ${lblV}(value)</span> ${dim}\u00d7</span> ${Y_SYM} ${lblB}(block scale)</span> ${dim}\u00d7</span> ${tensorHtml()}`
      );
    }
    for (const sp of scalePillars) {
      sp.g.addEventListener("pointerenter", () => {
        const hi = "hsl(9 60% 28%)";
        for (const f of [sp.faces.left, sp.faces.right, sp.faces.top]) {
          f.setAttribute("stroke", hi);
          f.setAttribute("stroke-width", f === sp.faces.top ? 1.4 : 0.9);
        }
        for (const tk of sp.ticks) {
          tk.setAttribute("stroke-opacity", 0.85);
          tk.setAttribute("stroke-width", 1);
        }
        showMathBlock(sp.r);
      });
      sp.g.addEventListener("pointerleave", () => {
        sp.faces.left.removeAttribute("stroke");
        sp.faces.left.removeAttribute("stroke-width");
        sp.faces.right.removeAttribute("stroke");
        sp.faces.right.removeAttribute("stroke-width");
        sp.faces.top.setAttribute("stroke", RUST);
        sp.faces.top.setAttribute("stroke-width", 0.7);
        for (const tk of sp.ticks) {
          tk.setAttribute("stroke-opacity", 0.45);
          tk.setAttribute("stroke-width", 0.7);
        }
      });
    }
    rowGs.forEach((g, r) => {
      g.style.cursor = "default";
      g.addEventListener("pointerenter", () => {
        rowGs.forEach((g2, i) => (g2.style.opacity = i === r ? "1" : "0.3"));
      });
      g.addEventListener("pointerleave", () => {
        rowGs.forEach((g2) => (g2.style.opacity = "1"));
        tooltip.classList.remove("on");
      });
    });

    /* the hovered bar itself: crisp outline, its ticks light up, value detail */
    for (const vb of valueBars) {
      vb.g.addEventListener("pointerenter", () => {
        const hi = vb.sign < 0 ? "hsl(215 55% 24%)" : "hsl(172 55% 24%)";
        vb.faces.top.setAttribute("stroke", hi);
        vb.faces.top.setAttribute("stroke-width", 1.3);
        vb.faces.left.setAttribute("stroke", hi);
        vb.faces.left.setAttribute("stroke-width", 0.9);
        vb.faces.right.setAttribute("stroke", hi);
        vb.faces.right.setAttribute("stroke-width", 0.9);
        for (const tk of vb.ticks) {
          tk.setAttribute("stroke-opacity", 0.85);
          tk.setAttribute("stroke-width", 1);
        }
        showMath(vb);
      });
      vb.g.addEventListener("pointerleave", () => {
        vb.faces.top.setAttribute("stroke", vb.sign < 0 ? "hsl(215 35% 48%)" : "hsl(172 30% 55%)");
        vb.faces.top.setAttribute("stroke-width", 0.5);
        vb.faces.left.removeAttribute("stroke");
        vb.faces.left.removeAttribute("stroke-width");
        vb.faces.right.removeAttribute("stroke");
        vb.faces.right.removeAttribute("stroke-width");
        for (const tk of vb.ticks) {
          tk.setAttribute("stroke-opacity", 0.18);
          tk.setAttribute("stroke-width", 0.6);
        }
      });
    }
    /* seed the strip with the loudest bar so the math is visible before any hover */
    showMath(valueBars.reduce((a, b2) =>
      rows[b2.r].codes[b2.c] * rows[b2.r].scale > rows[a.r].codes[a.c] * rows[a.r].scale ? b2 : a));

    /* --- memory panel: one 11x16 tile, BF16 vs NVFP4 --- */
    const MX = NARROW ? 202 : 720;
    const MY = NARROW ? 380 : 120;
    const MH = NARROW ? 190 : 244; /* wide: bar baseline at y=364, flush with the scene's front corner */
    const MW = 64;
    const bf16B = NR * NC * 2;  /* values x 2 bytes */
    const fp4B = (NR * NC) / 2; /* 176 x 4 bit = 88 */
    const scaleB = NR;          /* 11 rows x FP8 = 11 */
    const mScale = MH / bf16B;
    root.appendChild(svg("text", {
      x: MX + MW + 46, y: MY - 26, "text-anchor": "middle", "font-size": NARROW ? 14 : 10.5,
      "letter-spacing": "0.06em", fill: "#6d6860", "font-weight": 600,
      text: "TENSOR MEMORY USED",
    }));
    function memBar(x, segs, label) {
      let y = MY + MH;
      const g = svg("g", { style: "cursor:default" });
      for (const [bytes, color] of segs) {
        const hh = bytes * mScale;
        y -= hh;
        g.appendChild(svg("rect", { x, y, width: MW, height: hh, fill: color, stroke: "#fffdf8", "stroke-width": 1 }));
      }
      root.appendChild(g);
      root.appendChild(svg("text", {
        x: x + MW / 2, y: MY + MH + 18, "text-anchor": "middle", "font-size": NARROW ? 14 : 11, fill: "#3f3a33", "font-weight": 600, text: label,
      }));
      return g;
    }
    const gBf = memBar(MX, [[bf16B, "#b9b2a5"]], "BF16");
    const gFp = memBar(MX + MW + 28, [[fp4B, "hsl(172 32% 62%)"], [scaleB, "hsl(9 45% 62%)"]], "NVFP4");
    /* the amortized FP32 tensor scale: a gold hairline atop the NVFP4 bar */
    const fpTop = MY + MH - (fp4B + scaleB) * mScale;
    root.appendChild(svg("rect", {
      x: MX + MW + 28, y: fpTop - 4 * mScale, width: MW, height: 4 * mScale,
      fill: "hsl(42 55% 72%)", stroke: "#8a6d2f", "stroke-width": 0.8, "stroke-dasharray": "3 2",
    }));
    root.appendChild(svg("text", {
      x: MX + MW + 28 + MW / 2, y: MY + MH + (NARROW ? 36 : 32), "text-anchor": "middle", "font-size": NARROW ? 12 : 9.5, fill: "#8a6d2f",
      text: "+ FP32 / tensor",
    }));
    root.appendChild(svg("text", {
      x: MX + MW + 46, y: fpTop - 4 * mScale - 12, "text-anchor": "middle",
      "font-size": NARROW ? 16 : 12.5, "font-weight": 700, fill: TEAL,
      text: (bf16B / (fp4B + scaleB)).toFixed(1) + "\u00d7 smaller",
    }));
    function bindMem(g, html) {
      g.addEventListener("pointerenter", () => {
        tooltip.innerHTML = html;
        tooltip.classList.add("on");
        const rect = plotWrap.getBoundingClientRect();
        const r = g.getBoundingClientRect();
        placeTip(plotWrap, tooltip, r.left - rect.left + plotWrap.scrollLeft, r.top - rect.top + 24);
      });
      g.addEventListener("pointerleave", () => tooltip.classList.remove("on"));
    }
    bindMem(gBf, `<div class="tt-x">BF16 tile</div>${NR * NC} values \u00d7 16 bits = <b>${bf16B} bytes</b>`);
    bindMem(gFp,
      `<div class="tt-x">NVFP4 tile</div>` +
      `${NR * NC} values \u00d7 4 bits = <b>${fp4B} B</b> <span style="color:${TEAL}">(teal)</span><br>` +
      `${NR} block scales \u00d7 FP8 = <b>${scaleB} B</b> <span style="color:${RUST}">(rust)</span><br>` +
      `one FP32 tensor scale = 4 B <span style="color:#8a6d2f">(gold, dashed)</span> \u2014 per tensor, amortized over every tile`);

    /* view choice: NVFP4 reconstruction vs original BF16 weights (above the plot) */
    const controls = el("div", { class: "viz-controls", style: "justify-content:center;margin-bottom:6px" });
    function setMode(orig) {
      if (showOriginal === orig) return;
      showOriginal = orig;
      btnQ.setAttribute("aria-pressed", String(!orig));
      btnU.setAttribute("aria-pressed", String(orig));
      applyHeights();
    }
    const btnQ = el("button", {
      class: "viz-btn", text: "quantized weights", "aria-pressed": "true",
      onclick: () => setMode(false),
    });
    const btnU = el("button", {
      class: "viz-btn", text: "unquantized weights", "aria-pressed": "false",
      onclick: () => setMode(true),
    });
    controls.appendChild(btnQ);
    controls.appendChild(btnU);
    body.insertBefore(controls, plotWrap);

  };

  /* viz.js may have booted before these registrations ran (all scripts are
   * deferred, in order); boot again for any figures still waiting. */
  NV.boot();
})();
