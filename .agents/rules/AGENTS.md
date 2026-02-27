---
trigger: always_on
---

# Project Governance & Strict Rules

## 1. Pre-training Architecture

- **Primary Trainer**: `scripts/trainer.py` is the **only** authorized implementation for pre-training. Any pre-training logic must be integrated into or called through this file.
- **Configuration**: `scripts/config.py` is the **only** authorized source for global configurations. All hyperparameters, system settings, and paths must originate from here.
- No alternate trainer entrypoints, wrappers, or parallel implementations are allowed.
- Any refactoring must consolidate logic into `scripts/` and eliminate legacy fragmentation.

---

## 2. Development Constraints

- **Strict Adherence**: All architectural and structural rules must be followed without exception.
- **No File Creation**: Strictly no new files may be created unless explicitly instructed to do so.
- **Class Creation**: No new classes or architectural components should be introduced unless explicitly requested.
- **No Dynamic Implementation**: Do not add CLI flags, runtime switches, dynamic dispatch, plugin systems, or conditional selection logic unless explicitly requested. Implement exactly what is specified — statically.
- **No Unwanted Comments**: Do not add redundant, decorative, or explanatory comments inside the codebase.
- **Best Practices Mandatory**: All implementations must follow production-grade engineering standards, including clean structure, deterministic behavior, reproducibility, and maintainability.
- **OOM Prevention Required**: All training code must be written defensively to prevent out-of-memory errors. This includes:
  - Controlled batch sizing
  - Gradient accumulation where required
  - Proper use of `torch.no_grad()` where applicable
  - Efficient tensor lifecycle management
  - Avoiding unnecessary tensor copies
  - Releasing unused references
  - Mixed precision where appropriate
- **Redundancy Removal**: Any fragmented scripts or legacy pre-training implementations (e.g., those importing from non-existent `pretrain/` or `src/` directories) must be refactored to align strictly with the consolidated `scripts/` structure.
- No duplicate training loops.
- No shadow configurations.
- No implicit defaults outside `scripts/config.py`.

---

## 3. Current Project State

The project is transitioning to a consolidated structure under the `scripts/` directory:

- `scripts/trainer.py`: Production-grade training wrapper optimized for low VRAM usage.
- `scripts/config.py`: Centralized `TrainingConfig` dataclass and the single source of truth for configuration.
- Existing scripts such as `train_pretrain.py` and `train_distill.py` must be refactored to remove legacy imports and point exclusively to the authorized `scripts/` modules.
- All execution paths must ultimately route through `scripts/trainer.py`.

---

## 4. Enforcement Policy

- Any deviation from the above structure is considered a violation.
- No temporary patches.
- No parallel implementations.
- No architectural experiments without explicit approval.
- All updates must strengthen consolidation, stability, and memory safety.

---
