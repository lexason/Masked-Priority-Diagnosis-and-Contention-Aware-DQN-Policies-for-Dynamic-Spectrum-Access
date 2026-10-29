# Masked-Priority Diagnosis and Contention-Aware Evaluation of Deep Q-Network Policies for Dynamic Spectrum Access

This repository is the MIT-licensed code companion for the standalone
original research article of the same title. It contains executable policy
audits, the source-frozen prospective collision-aware implementation, and the
article-specific logical-coordinate and inferential reanalysis code.

The inherited path `prospective_collision_study/main/revision/...` identifies
the frozen experiment source tree; it does not characterize this standalone
article as a revision or correction.

Aggregate research data, trained checkpoints, results, and validation
materials are distributed through the article-specific Zenodo record. The six
multi-gigabyte archival trace ZIPs are intentionally excluded from the minimum
deposit; native row-level revalidation of trace-derived audits requires that
separately sealed corpus. The article-specific
version DOI is https://doi.org/10.5281/zenodo.23766140;
consult that DOI landing page for the current versioned record. Historical
lineage remains in the package's machine-readable provenance rather than in
this current-release README.

The manuscript authoring path `analysis/recompute_inference.py` is released as
`article_reanalysis/recompute_inference.py`. The top-level `README.md` in the
versioned Zenodo archive gives the exact invocation from the extracted package
root.

The installable software package remains version
`0.1.0`; the article-specific dataset/release is
version `1.2.0`. These identifiers describe different artifact scopes.

The GitHub tree deliberately excludes manuscript files, supplementary files,
author portraits, editorial material, credentials, the workspace research
protocol, data, checkpoints, and generated results. See
`SOURCE_MANIFEST.json` for the exact committed-file identities.
Its text members receive a deterministic, semantics-preserving cleanup of
trailing spaces and trailing blank lines before their hashes are recorded.

## Installation

```sh
uv sync --frozen --extra dev
```

This source-only tree can run the 47 frozen prospective implementation tests.
The article reanalysis, claim/macro/figure, Gold-comparison, and release smoke
tests require the data and generated evidence that are intentionally absent
here. To run those evidence-dependent controls, obtain the article-specific
Zenodo reproducibility ZIP, extract it into a separate clean working directory,
and execute the commands from that extracted archive root as documented in its
top-level `README.md`. Do not treat missing-data failures from this MIT-only
source tree as reproduction results.

## Licence

Code and repository documentation are licensed under the MIT License. Data and
checkpoints are not distributed from this Git repository.
