# james_mukunzi_rl_summative

## Setup and run

Requires only [uv](https://docs.astral.sh/uv/). uv installs Python 3.11 and all dependencies automatically.

```bash
uv sync
uv run main.py
```

## Project structure

```
project_root/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── environment/
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── dqn_training.py
│   └── pg_training.py
├── models/
│   ├── dqn/
│   └── pg/
├── logs/
├── assets/
└── tests/
```
