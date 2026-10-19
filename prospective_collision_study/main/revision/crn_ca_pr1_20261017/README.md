# Collision-aware prospective study runner

This isolated package implements protocol `CRN-CA-PR1-20261017`. It does not
import or alter another prospective study's code or results. Run commands from
the repository root.

Pre-freeze review and smoke sequence:

```bash
uv run --frozen ruff check main/revision/crn_ca_pr1_20261017
uv run --frozen pytest -q main/revision/crn_ca_pr1_20261017
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.train --plan
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.evaluate --plan
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.train --all --smoke --workers 1
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.evaluate --smoke --workers 1
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.validation --smoke
```

Delete or relocate only the disposable smoke directory before repeating that
smoke sequence. Do not modify production output directories in place.

After scientific/code review, freeze once, then run production:

```bash
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.provenance init
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.provenance verify
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.train --all --workers 8
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.evaluate --workers 12
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.validation
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.analyze
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.plot
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.provenance seal-results
uv run --frozen python -m main.revision.crn_ca_pr1_20261017.provenance verify-seal
```

Production commands fail without a valid source freeze and refuse to overwrite
their final training/evaluation artifacts. The evaluation summary is compact:
per-slot/node pseudoreplication is avoided by aggregating within each run.
