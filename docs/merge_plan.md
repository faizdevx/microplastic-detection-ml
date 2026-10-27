# Merge plan

State of `origin/main` when this was written: `a83b314` (pull request 2 merged:
polymer identification and the scattering model). Dry runs were made in a throw-away worktree
(`git merge --no-commit --no-ff`), one branch after the other, and discarded. `phase5-robustness` was at
`93c1f61`.

| Order | Branch | Dry-run result | Change against the previous state | Conflicts |
|---|---|---|---|---|
| 1 | `phase3-scattering-sim` | clean | 3 files changed, 150 insertions(+), 15 deletions(-) | none |
| 2 | `phase4-release` | clean | 54 files changed, 2229 insertions(+), 327 deletions(-) | none |
| 3 | `phase5-robustness` | clean | 16 files changed, 1961 insertions(+), 2 deletions(-) | none |

Notes:

- `phase3-scattering-sim` adds one commit that `main` does not have (the README status table and
  gallery).
- `phase4-release` was cut from `phase3-scattering-sim`, so it contains that commit; merging it also
  merges `mp analyse`, the demo app, deployment tools, CI, the claims audit and the release files.
- `phase5-robustness` was cut from `phase4-release`. Merging it alone into `main` brings in all
  three, so the three merges can be one pull request.
- No branch has conflicts with `main` or with the others. Nothing was merged or pushed by this
  plan.
