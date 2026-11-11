<p align="center">
  <img src="assets/banner.svg" alt="defect-inspector" width="100%">
</p>

<p align="center"><em>Find surface defects from a few good samples, with a heatmap of where they are.</em></p>

<p align="center">
  <img alt="Status: in development" src="https://img.shields.io/badge/status-in%20development-8a3a24?style=flat-square">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1c1a17?style=flat-square">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-1c1a17?style=flat-square">
  <img alt="Backends: API or Hugging Face" src="https://img.shields.io/badge/backends-API%20%7C%20Hugging%20Face-1c1a17?style=flat-square">
</p>

---

## Features

- **Few-shot anomaly detection**: learns what "good" looks like from a handful of defect-free images, no defect labels needed.
- **Heatmap overlay**: shows where on the part the anomaly is, with an adjustable threshold.
- **Web demo**: upload an image, get a score and a heatmap.
- **Export**: heatmaps as PNG, scores and regions as JSON or CSV.

> **Data.** Demos will only use datasets whose license permits public demo use, or images we create ourselves. No dataset is bundled in this repository.

## Architecture

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","wrappingWidth":260},"themeVariables":{"fontFamily":"Georgia, Times New Roman, serif","fontSize":"14px","primaryColor":"#f3efe6","primaryTextColor":"#1c1a17","primaryBorderColor":"#1c1a17","lineColor":"#1c1a17","clusterBkg":"#faf7f0","clusterBorder":"#8a3a24","titleColor":"#8a3a24","edgeLabelBackground":"#f3efe6"}}}%%
flowchart TB
  G["A few good samples"]:::ink --> F["Feature extractor<br/>wide_resnet50_2 / DINOv2"]:::ink
  F --> M["Memory bank<br/>PatchCore via anomalib"]:::ink
  T["New image"]:::ink --> F
  M --> HM["Anomaly score<br/>+ heatmap"]:::ink
  HM --> DS["Describe and label<br/>the defect"]:::ink
  DS --> EX["Web demo · PNG · JSON / CSV"]:::ink
  subgraph BK["Backend · pick one"]
    direction LR
    BKA["<b>(a) Your API key</b><br/>OpenAI-compatible<br/>vision model"]:::rust
    BKH["<b>(b) Hugging Face</b><br/>Qwen2.5-VL-7B-Instruct<br/>run locally"]:::rust
  end
  BK -.-> DS

  classDef cl fill:#faf7f0,stroke:#8a3a24,stroke-dasharray:4 3,color:#8a3a24;
  classDef ink fill:#f3efe6,stroke:#1c1a17,color:#1c1a17,stroke-width:1px;
  classDef rust fill:#f3efe6,stroke:#8a3a24,color:#8a3a24,stroke-width:1.5px;
```

### Two interchangeable backends

Pick one with `MS_BACKEND` in `.env`. The rest of the pipeline does not change.

| | (a) Bring your own API key | (b) Open-source on Hugging Face |
|---|---|---|
| **What** | Any OpenAI-compatible endpoint | Open models, run locally or via Hugging Face Inference |
| **Set** | `MS_BACKEND=api`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `MS_BACKEND=hf`, `HF_MODE=local` or `inference` |
| **Good for** | Fastest start, strongest hosted models | Data stays on your machines, no per-call cost |

**Default open-source models**

| Model | Used for | License |
|---|---|---|
| [`timm/wide_resnet50_2.tv_in1k`](https://huggingface.co/timm/wide_resnet50_2.tv_in1k) | PatchCore feature backbone | BSD-3-Clause |
| [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | alternative feature backbone | Apache-2.0 |
| [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | describing and labelling the flagged region | Apache-2.0 |
| [anomalib](https://github.com/open-edge-platform/anomalib) | PatchCore implementation | Apache-2.0 |

## Quickstart

> **Coming soon.** The commands below show the planned interface.

```bash
git clone https://github.com/ModelShifts/defect-inspector
cd defect-inspector
cp .env.example .env   # choose a backend
pip install -e .
defect-inspector fit samples/good/
defect-inspector inspect part.jpg --heatmap out/part_heatmap.png
```

## Try it

> **Coming soon.** A hosted demo will be linked here and on [modelshifts.com/work](https://modelshifts.com/work).

## Roadmap

- [x] Repository scaffold and README
- [ ] Few-shot fit on good samples
- [ ] Anomaly score and heatmap
- [ ] Defect description via either backend
- [ ] Web demo
- [ ] Export to PNG, JSON and CSV

## License

[MIT](LICENSE) © Modelshifts Technology Pte. Ltd.

<p align="center"><sub>Part of the <a href="https://modelshifts.com/work">Modelshifts</a> showcase.</sub></p>
