from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_BUILDS = "DERAIL_BUILDS_DIR"
ENV_OUT = "DERAIL_ANALYSIS_OUT"


def _resolve(cli_value: str | None, env_name: str, what: str) -> Path:
    raw = cli_value if cli_value else os.environ.get(env_name)
    if not raw:
        raise SystemExit(
            f"{what} is not configured. Pass it on the command line or export {env_name}."
        )
    return Path(raw).expanduser()


@dataclass(frozen=True)
class Paths:
    builds_dir: Path
    out_dir: Path

    @classmethod
    def resolve(cls, builds_dir: str | None = None, out_dir: str | None = None) -> "Paths":
        builds = _resolve(builds_dir, ENV_BUILDS, "The derail_builds directory")
        out = _resolve(out_dir, ENV_OUT, "The analysis output directory")
        if not builds.is_dir():
            raise SystemExit(f"{ENV_BUILDS} does not point at a directory: {builds}")
        return cls(builds_dir=builds.resolve(), out_dir=out.resolve())

    @property
    def human_labels(self) -> Path:
        return self.builds_dir / "human_labels"

    @property
    def rubric_scores(self) -> Path:
        return self.human_labels / "rubric_scores"

    @property
    def rollout_flags(self) -> Path:
        return self.human_labels / "rollout_flags"

    @property
    def cleaning_proposals(self) -> Path:
        return self.human_labels / "cleaning_proposals"

    @property
    def drafts(self) -> Path:
        return self.human_labels / "drafts"

    @property
    def taxonomy_file(self) -> Path:
        return self.human_labels / "taxonomy" / "open_coded_labels.json"

    @property
    def tables(self) -> Path:
        return self.out_dir / "tables"

    @property
    def figures(self) -> Path:
        return self.out_dir / "figures"

    def ensure_out(self) -> None:
        for d in (self.out_dir, self.tables, self.figures):
            d.mkdir(parents=True, exist_ok=True)
