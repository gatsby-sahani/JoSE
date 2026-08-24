# Joint-Space Empowerment

This repository contains the official implementation of

**Joint-Space Empowerment for Dexterous Coordination in Tendon-Driven Hands**  
*James Heald, Vittorio Caggiano, Vikash Kumar, Maneesh Sahani*  
**ICML 2026 (Spotlight)**  
[Paper](https://openreview.net/pdf?id=qI2eHwfNfh) | [Project Page](https://joint-space-empowerment.github.io/)

---

Searching for effective policies is notoriously challenging in overactuated tendon-driven systems, where each joint is actuated by many muscles or motorized cables. Although this redundancy complicates naive policy search, it also implies that effective control can be captured by a low-dimensional action manifold. To identify such a manifold, we introduce Joint-Space Empowerment (JoSE), a novel information-theoretic objective that quantifies how much control an agent has over its mechanical degrees of freedom. We frame manifold discovery as an optimal precoding problem—where a state-dependent precoder maps low-dimensional latent actions to high-dimensional actions—and derive its closed-form solution under learned control-affine Gaussian dynamics. Across both a musculoskeletal hand model and a tendon-driven robotic hand, we show that policies trained on this manifold achieve significantly enhanced dexterity, sample efficiency, and improved generalization. More broadly, these results present optimal precoding as a general information-theoretic paradigm for coordinating high-dimensional actuators to control low-dimensional features. 

---

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

```bash
uv sync
```

### Verify installation

```bash
uv run pytest tests
```

---

## Training

Run training with:

```bash
uv run python train.py env=<env_name>
```

### Available environments

| Environment key | Hand | Task |
|---|---|---|
| `CustomMyoChallengeBaodingP1-v1` | MyoHand | Baoding Balls |
| `CustomMyoChallengeDieReorientP2-v0` | MyoHand | Die Reorientation |
| `myoHandKeyTurnRandom-v0` | MyoHand | Key Turn |
| `myoHandPenTwirlRandom-v0` | MyoHand | Pen Twirl |
| `myoHandReorient8-v0` | MyoHand | Reorient 8 objects (sparse reward) |
| `myoHandReorient100-v0` | MyoHand | Reorient 100 objects |
| `CustomAdroitBaodingP1-v1` | Adroit | Baoding Balls |
| `CustomAdroitDieReorientP2-v0` | Adroit | Die Reorientation |
| `CustomAdroitKeyTurnRandom-v0` | Adroit | Key Turn |
| `CustomAdroitPenTwirlRandom-v0` | Adroit | Pen Twirl |

---

## Pretrained models

Pretrained models are available on [HuggingFace](https://huggingface.co/jamesheald/joint-space-empowerment). To load and run a policy:

```bash
uv run python play.py --hand <hand> --task <task> --seed <seed>
```

| Argument | Choices |
|---|---|
| `--hand` | `Adroit`, `MyoHand` |
| `--task` | `BaodingBalls`, `DieReorient`, `KeyTurn`, `PenTwirl`, `Reorient8-sparse`, `Reorient100/Training` |
| `--seed` | `0`, `1`, `2`, `3`, `4`, `5` |

Seeds vary by hand and task. See the HuggingFace model directory to see which seeds are available for each task-hand combination.


---

## Citation

```bibtex
@inproceedings{heald2026jointspaceempowerment,
  title     = {Joint-Space Empowerment for Dexterous Coordination in Tendon-Driven Hands},
  author    = {Heald, James and Caggiano, Vittorio and Kumar, Vikash and Sahani, Maneesh},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
