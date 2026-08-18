/* Canaan-owned diagram specs for the shared humans& visualization runtime. */
(function () {
  "use strict";

  const NV = window.NVFP4_VIZ;

  function kimiK3Spec(narrow) {
    const nodes = {
      text: {
        x: 20, y: 110, w: 180, h: 64, kind: "plain", title: "Text tokens",
        sub: ["160K vocabulary"],
        info: "文本 token 直接进入共享的语言模型 embedding 空间。",
      },
      vision: {
        x: 20, y: 510, w: 180, h: 72, kind: "io", title: "Images · video",
        sub: ["native multimodal input"],
        info: "图像和视频共用同一套视觉参数；视频还会做时间维度的注意力与池化。",
      },
      moon: {
        x: 245, y: 500, w: 210, h: 92, kind: "rowcopy", title: "MoonViT-V2",
        sub: ["27 layers · 401M", "2×2 pixel shuffle"],
        info: "从头训练的视觉编码器。Pixel shuffle 将视觉 token 数量压到四分之一。",
      },
      projector: {
        x: 500, y: 510, w: 180, h: 72, kind: "rowcopy", title: "MLP projector",
        sub: ["visual → LLM space"],
        info: "轻量 MLP 把视觉特征投影到语言模型的共享 embedding 空间。",
      },
      embedding: {
        x: 245, y: 250, w: 210, h: 78, kind: "bf16", title: "Shared embedding",
        sub: ["7168 hidden dim", "up to 1M tokens"],
        info: "文本与视觉 token 在这里汇入同一条最长一百万 token 的序列。",
      },
      memory: {
        x: 245, y: 70, w: 210, h: 78, kind: "bf16", title: "Embedding + prior blocks",
        sub: ["depth-wise memory"],
        info: "AttnRes 可以读取初始 embedding 和先前 block 的输出，不必把全部历史压进一个残差状态。",
      },
      attnres: {
        x: 500, y: 225, w: 180, h: 92, kind: "dq", title: "Attention Residuals",
        sub: ["learned pseudo-query", "select across depth"],
        info: "每层用可学习的 pseudo-query 对 embedding 和先前 block 表示计算权重，再选择性聚合。",
      },
      kda: {
        x: 755, y: 70, w: 180, h: 82, kind: "nvfp4", title: "KDA × 3",
        sub: ["linear attention", "token mixing"],
        info: "每个 block 先连续放置三层 Kimi Delta Attention，用 channel-wise forget gate 处理长序列。",
      },
      moeKda: {
        x: 970, y: 70, w: 190, h: 82, kind: "gemm", title: "Stable LatentMoE × 3",
        sub: ["after every KDA"],
        info: "每个 KDA 后都跟一个稀疏 MoE 前馈层，负责 channel mixing。",
      },
      mla: {
        x: 755, y: 260, w: 180, h: 82, kind: "fprop", title: "Gated MLA × 1",
        sub: ["global attention", "96 heads"],
        info: "每三层 KDA 后插入一层 Gated MLA，周期性恢复全局 token 交互。",
      },
      moeMla: {
        x: 970, y: 260, w: 190, h: 82, kind: "gemm", title: "Stable LatentMoE × 1",
        sub: ["896 experts · top-16"],
        info: "每个 token 从 896 个 routed experts 中激活 16 个，并额外使用 2 个 shared experts。",
      },
      repeat: {
        x: 755, y: 430, w: 180, h: 72, kind: "plain", title: "Repeat × 23 blocks",
        sub: ["92 attention layers"],
        info: "3:1 的 KDA / MLA 模式重复 23 次，共组成 92 层。",
      },
      finalMla: {
        x: 970, y: 430, w: 190, h: 82, kind: "fprop", title: "Final Gated MLA",
        sub: ["layer 93 · global"],
        info: "骨干末尾再放一个 Gated MLA，使最后一层始终执行全局注意力。",
      },
      finalMoe: {
        x: 970, y: 570, w: 190, h: 82, kind: "gemm", title: "Stable LatentMoE",
        sub: ["104B active / 2.8T total"],
        info: "K3 总参数量 2.8T；稀疏路由让每个 token 实际激活约 104B 参数。",
      },
      output: {
        x: 755, y: 570, w: 180, h: 82, kind: "plain", title: "LM head · output",
        sub: ["reasoning · tools · pixels"],
        info: "共享骨干输出下一个 token，可继续推理、调用工具，或检查视觉反馈。",
      },
    };

    let w = 1180;
    let h = 680;
    let edges = [
      { id: "textEmbed", p: [[200, 142], [220, 142], [220, 289], [245, 289]] },
      { id: "visionMoon", p: [[200, 546], [245, 546]] },
      { id: "moonProjector", p: [[455, 546], [500, 546]] },
      { id: "projectorEmbed", p: [[590, 510], [590, 380], [350, 380], [350, 328]], label: "shared token stream", labelAt: [470, 369] },
      { id: "memoryRes", p: [[455, 109], [475, 109], [475, 255], [500, 255]], label: "α", labelAt: [482, 189], labelAnchor: "start" },
      { id: "embedRes", p: [[455, 289], [500, 289]], label: "α", labelAt: [477, 279] },
      { id: "resKda", p: [[680, 271], [710, 271], [710, 111], [755, 111]], label: "block input", labelAt: [718, 194], labelAnchor: "start" },
      { id: "kdaMoe", p: [[935, 111], [970, 111]] },
      { id: "moeKdaMla", p: [[1065, 152], [1065, 205], [720, 205], [720, 301], [755, 301]], label: "×3", labelAt: [732, 244], labelAnchor: "start" },
      { id: "mlaMoe", p: [[935, 301], [970, 301]] },
      { id: "moeRepeat", p: [[1065, 342], [1065, 382], [845, 382], [845, 430]], label: "one hybrid block", labelAt: [954, 372] },
      { id: "repeatFinal", p: [[935, 466], [970, 466]] },
      { id: "finalMlaMoe", p: [[1065, 512], [1065, 570]] },
      { id: "finalMoeOutput", p: [[970, 611], [935, 611]] },
    ];

    if (narrow) {
      Object.assign(nodes.text, { x: 25, y: 20, w: 170, h: 52 });
      Object.assign(nodes.vision, { x: 305, y: 20, w: 170, h: 52 });
      Object.assign(nodes.moon, { x: 285, y: 110, w: 190, h: 70 });
      Object.assign(nodes.projector, { x: 285, y: 215, w: 190, h: 60 });
      Object.assign(nodes.embedding, { x: 155, y: 320, w: 190, h: 64 });
      Object.assign(nodes.memory, { x: 25, y: 430, w: 190, h: 64 });
      Object.assign(nodes.attnres, { x: 270, y: 430, w: 205, h: 70 });
      Object.assign(nodes.kda, { x: 155, y: 555, w: 190, h: 70 });
      Object.assign(nodes.moeKda, { x: 155, y: 665, w: 190, h: 78 });
      Object.assign(nodes.mla, { x: 155, y: 790, w: 190, h: 70 });
      Object.assign(nodes.moeMla, { x: 155, y: 900, w: 190, h: 78 });
      Object.assign(nodes.repeat, { x: 155, y: 1025, w: 190, h: 64 });
      Object.assign(nodes.finalMla, { x: 155, y: 1135, w: 190, h: 78 });
      Object.assign(nodes.finalMoe, { x: 155, y: 1260, w: 190, h: 78 });
      Object.assign(nodes.output, { x: 155, y: 1385, w: 190, h: 70 });
      w = 500;
      h = 1480;
      edges = [
        { id: "textEmbed", p: [[110, 72], [110, 352], [155, 352]] },
        { id: "visionMoon", p: [[390, 72], [390, 110]] },
        { id: "moonProjector", p: [[380, 180], [380, 215]] },
        { id: "projectorEmbed", p: [[285, 245], [250, 245], [250, 320]], label: "shared tokens", labelAt: [240, 292], labelAnchor: "end" },
        { id: "memoryRes", p: [[215, 462], [270, 462]], label: "α", labelAt: [242, 452] },
        { id: "embedRes", p: [[250, 384], [250, 407], [372, 407], [372, 430]], label: "α", labelAt: [310, 397] },
        { id: "resKda", p: [[372, 500], [372, 525], [250, 525], [250, 555]], label: "block input", labelAt: [310, 515] },
        { id: "kdaMoe", p: [[250, 625], [250, 665]] },
        { id: "moeKdaMla", p: [[250, 743], [250, 790]], label: "×3", labelAt: [265, 770], labelAnchor: "start" },
        { id: "mlaMoe", p: [[250, 860], [250, 900]] },
        { id: "moeRepeat", p: [[250, 978], [250, 1025]], label: "one block", labelAt: [265, 1005], labelAnchor: "start" },
        { id: "repeatFinal", p: [[250, 1089], [250, 1135]] },
        { id: "finalMlaMoe", p: [[250, 1213], [250, 1260]] },
        { id: "finalMoeOutput", p: [[250, 1338], [250, 1385]] },
      ];
    }

    return {
      w,
      h,
      title: "Kimi K3 · Native Multimodal MoE",
      subtitle: "2.8T total · 104B active · 93 layers · 1M context",
      nodes,
      edges,
      flows: [
        { path: [{ edge: "textEmbed" }, { edge: "embedRes" }, { edge: "resKda" }, { edge: "kdaMoe" }, { edge: "moeKdaMla" }, { edge: "mlaMoe" }, { edge: "moeRepeat" }, { edge: "repeatFinal" }, { edge: "finalMlaMoe" }, { edge: "finalMoeOutput" }], start: "hp", count: 5 },
        { path: [{ edge: "visionMoon" }, { edge: "moonProjector" }, { edge: "projectorEmbed" }], start: "hp", count: 2 },
      ],
      modes: [
        { label: "whole model", note: "Follow the complete multimodal token path." },
        { label: "token mixing", note: "KDA handles efficient long context; Gated MLA restores periodic global attention.", dim: ["text", "vision", "moon", "projector", "embedding", "memory", "attnres", "moeKda", "moeMla", "repeat", "finalMoe", "output"] },
        { label: "depth mixing", note: "AttnRes selects from the embedding and preceding block representations.", dim: ["text", "vision", "moon", "projector", "kda", "moeKda", "mla", "moeMla", "finalMla", "finalMoe", "output"] },
        { label: "channel mixing", note: "Stable LatentMoE routes each token to 16 of 896 experts.", dim: ["text", "vision", "moon", "projector", "embedding", "memory", "attnres", "kda", "mla", "repeat", "finalMla", "output"] },
        { label: "vision path", note: "MoonViT-V2 and an MLP projector join vision to the shared token stream.", dim: ["text", "memory", "attnres", "kda", "moeKda", "mla", "moeMla", "repeat", "finalMla", "finalMoe", "output"] },
      ],
      flowNote: "● token flow · hover a module for detail",
      legend: [
        { color: "#e4eeec", label: "KDA · token mixing" },
        { color: "#e6f1e2", label: "Gated MLA · global attention" },
        { color: "#e9eef6", label: "AttnRes · depth mixing" },
        { color: "#f4ecdb", label: "Stable LatentMoE · channel mixing" },
        { color: "#dceafd", label: "native vision pathway" },
      ],
      caption: "One hybrid block contains three KDA layers and one Gated MLA layer, each followed by Stable LatentMoE. The block repeats 23 times; a final Gated MLA makes layer 93 global.",
    };
  }

  NV.registerDiagram("kimi-k3-architecture", kimiK3Spec);
  NV.boot();
})();
