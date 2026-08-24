# Adroit source files

These files are copied from the [Adroit](https://github.com/vikashplus/Adroit) repository at commit `c98c9c92fc9729e92a10c8db79588348897d8fe3`. They are bundled here so that no external clone is required.

At runtime, `adroit_utils.py` augments the model with an actuated pronation-supination wrist joint and creates self-contained MuJoCo environment XMLs with task elements transferred from the MyoHand model. The XMLs are written to the sibling `assets/` directory (git-ignored) and define the Adroit hand environments registered in `src/envs/__init__.py`.
