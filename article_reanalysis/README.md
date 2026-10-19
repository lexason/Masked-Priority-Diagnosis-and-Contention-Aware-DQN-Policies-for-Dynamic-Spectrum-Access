# Article-specific reanalysis

These scripts read the sealed aggregate collision cube and do not import
the original analysis implementation. They distinguish logical from physical
channel coordinates and audit attempted actions separately from abstention.
They also recompute crossed-bootstrap intervals, fully enumerated sign-flip
statistics, deletion sensitivities, load bounds, and operating diagnostics.

Pinned workspace inputs copied into this archive:

- `analysis/reanalyse.py`: `4c19d29f992131e9c289c17ba4d0cf0c65d889c5a43df060c57f8f4d7594aba1`
- `analysis/recompute_inference.py`: `0d6416d94f270b772e2e711cab7aa1420ff7956b162c65db0ece25afde5df9f4`
- `analysis/test_reanalyse_smoke.py`: `c3118506def2fc9426cf05b7e4cdca14eb9d98bd21b2a80d57965d6ebce8a545`
- `analysis/test_reanalyse_smoke_contract.py`: `9ab72058187e1ff7efd4147d63efd991ef914a5aa4a33aef46b6c61b3515d600`
- `analysis/test_recompute_inference.py`: `a813e3890d5cc7f4fe09df3e5a09c05df57e693102b7a23ca49c8969307cdd1b`
- `evidence/gold/inference/recomputed.json`: `5199f52a44d119010207c2e0358217d2a14ed238f57bb9a743f76f7c189ccca3`
- `evidence/gold/reanalysis/attempted_only_logical_audit.csv`: `fe686729c5c65d6d1a59b0afb5016e8d9b80c24401d7a4abd7d5cb71c9d8d7f2`
- `evidence/gold/reanalysis/mechanism_audit_figure_data.csv`: `9e69eca00e3c5343d67e342e34f9cb816b311c137f493dc92d2fa2aa12c61e2a`
- `evidence/gold/reanalysis/reanalysis_output.json`: `ab6ccafc932ddca28a2aec40d6e8f46091ab17992955aed988a764649affe2e0`

The paths in this list are workspace source paths. Scripts are released
directly under `article_reanalysis/`, and outputs under
`article_reanalysis/outputs/`.
