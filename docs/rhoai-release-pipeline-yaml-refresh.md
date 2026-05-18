# RHOAI release branch: refresh managed `pipeline.yaml` artifacts

On release branches (for example `rhoai-3.5-ea.1`), compiled `pipeline.yaml` files are committed
in the repository. After an automatic merge from `main`, Konflux builds runtime images on the
release branch; those images are mirrored from Quay to Red Hat Registry with the **same digest**.

This runbook describes how to refresh managed pipeline YAMLs locally or how to implement
branch-scoped automation.

## Image sources

| Runtime | Quay (build output) | Red Hat Registry (use at compile time) |
|---------|---------------------|----------------------------------------|
| AutoML | `quay.io/rhoai/odh-automl-rhel9:<release-branch>` | `registry.redhat.io/rhoai/odh-automl-rhel9@<digest>` |
| AutoRAG | `quay.io/rhoai/odh-autorag-rhel9:<release-branch>` | `registry.redhat.io/rhoai/odh-autorag-rhel9@<digest>` |

Example (same digest on both registries):

```text
quay.io/rhoai/odh-automl-rhel9:rhoai-3.5-ea.1@sha256:b2b8a396cde5624cf9e501204783e1bd9246bf2f7207823632ce639690393b77
registry.redhat.io/rhoai/odh-automl-rhel9@sha256:b2b8a396cde5624cf9e501204783e1bd9246bf2f7207823632ce639690393b77
```

Resolve the digest from Quay using the **release branch name as the image tag**, after Konflux
builds for AutoML and AutoRAG have succeeded (and mirroring has completed).

## Environment variables

Component and pipeline compilation read runtime images from `utils/consts.py`:

| Variable | Purpose |
|----------|---------|
| `RELATED_IMAGE_ODH_AUTOML_IMAGE` | AutoML runtime image (digest-pinned `registry.redhat.io/...` URL recommended on release branches) |
| `RELATED_IMAGE_ODH_AUTORAG_IMAGE` | AutoRAG runtime image (same convention) |

If unset, defaults are `quay.io/opendatahub/odh-automl:odh-stable` and
`quay.io/opendatahub/odh-autorag:odh-stable` (typical for upstream `main`, not RHOAI release).

## Resolve digests with `skopeo`

Replace `<release-branch>` with your branch (for example `rhoai-3.5-ea.1`).

### Linux / CI (`linux/amd64` runner)

```bash
RELEASE_BRANCH=rhoai-3.5-ea.1

AUTOML_DIGEST=$(skopeo inspect --format '{{.Digest}}' \
  "docker://quay.io/rhoai/odh-automl-rhel9:${RELEASE_BRANCH}")

AUTORAG_DIGEST=$(skopeo inspect --format '{{.Digest}}' \
  "docker://quay.io/rhoai/odh-autorag-rhel9:${RELEASE_BRANCH}")
```

### macOS (Apple Silicon)

`skopeo` defaults to `darwin/arm64`, which is not in the RHOAI image manifest list. Force
`linux/amd64` (clusters use this architecture):

```bash
RELEASE_BRANCH=rhoai-3.5-ea.1

AUTOML_DIGEST=$(skopeo inspect --override-os linux --override-arch amd64 --format '{{.Digest}}' \
  "docker://quay.io/rhoai/odh-automl-rhel9:${RELEASE_BRANCH}")

AUTORAG_DIGEST=$(skopeo inspect --override-os linux --override-arch amd64 --format '{{.Digest}}' \
  "docker://quay.io/rhoai/odh-autorag-rhel9:${RELEASE_BRANCH}")
```

### Alternative: `crane`

```bash
RELEASE_BRANCH=rhoai-3.5-ea.1

AUTOML_DIGEST=$(crane digest "quay.io/rhoai/odh-automl-rhel9:${RELEASE_BRANCH}" --platform linux/amd64)
AUTORAG_DIGEST=$(crane digest "quay.io/rhoai/odh-autorag-rhel9:${RELEASE_BRANCH}" --platform linux/amd64)
```

## Compile all managed pipelines

From the repository root (after `uv sync`):

```bash
export RELATED_IMAGE_ODH_AUTOML_IMAGE="registry.redhat.io/rhoai/odh-automl-rhel9@${AUTOML_DIGEST}"
export RELATED_IMAGE_ODH_AUTORAG_IMAGE="registry.redhat.io/rhoai/odh-autorag-rhel9@${AUTORAG_DIGEST}"

uv run python -m scripts.generate_managed_pipelines.generate_managed_pipelines \
  -o /tmp/managed-pipelines.json
```

Use `-o` to write the catalog manifest outside the repository (the release branch does not track
it in git). `pipeline.yaml` files are still written next to each managed `pipeline.py`. The
manifest is produced again during `odh-pipelines-components` image build in Konflux. Only commit
`pipelines/**/pipeline.yaml`.

Review and commit the diff:

```bash
git diff --stat
git diff pipelines/
git add pipelines/**/pipeline.yaml
```

## Compile a single pipeline (manual)

Equivalent to the historical per-pipeline workflow; set **both** env vars if the pipeline uses
both runtimes, or only the one you need:

```bash
export RELATED_IMAGE_ODH_AUTOML_IMAGE="registry.redhat.io/rhoai/odh-automl-rhel9@${AUTOML_DIGEST}"

uv run python pipelines/training/automl/autogluon_timeseries_training_pipeline/pipeline.py
```

Each managed `pipeline.py` with an `if __name__ == "__main__"` block writes `pipeline.yaml` in the
same directory.

## Recommended order of operations

1. Automatic merge lands on the release branch (for example `rhoai-3.5-ea.1`).
2. Konflux completes builds for `odh-automl-rhel9` and `odh-autorag-rhel9`.
3. Images are available on Quay with tag `<release-branch>` and mirrored to `registry.redhat.io`.
4. Run digest resolution and `generate_managed_pipelines` (this runbook).
5. Commit refreshed YAML artifacts.

Do not refresh YAML before step 3, or compiled specs will pin stale digests.

## GitHub Actions (example)

[`.github/workflows/refresh-managed-pipelines-rhoai.yaml`](../.github/workflows/refresh-managed-pipelines-rhoai.yaml)
runs on push to `rhoai-3.5-ea.1` (source paths only; ignores `pipeline.yaml` to avoid commit loops),
writes the catalog manifest to `${RUNNER_TEMP}` via `-o` (not in the repo), and supports manual
`workflow_dispatch`. It retries
Quay digest resolution until runtime images exist after Konflux builds.

## Konflux: skip builds when only compiled YAML changes

**Do not edit** `.tekton/` in this repository directly — it is synced from
[`konflux-central`](https://github.com/red-hat-data-services/konflux-central) (see
[`.tekton/README.md`](../.tekton/README.md)).

Today’s push `on-cel-expression` still starts a build when *any* path outside `.tekton/` changes.
A bot commit that only refreshes `pipeline.yaml` therefore matches
`files.all.exists(p, !p.matches('^\\.tekton/'))` and **re-triggers** AutoML / AutoRAG /
`pipelines-components` image builds.

Add this guard to each **v3.5 ea.1 on-push** PipelineRun under
`pipelineruns/pipelines-components/.tekton/` in `konflux-central`:

- `odh-automl-v3-5-ea-1-push.yaml`
- `odh-autorag-v3-5-ea-1-push.yaml`
- `odh-pipelines-components-v3-5-ea-1-push.yaml`

**Meaning:** `!files.all.all(x, x.matches('pipeline\\.yaml$'))` — run the build only if **not**
every changed file is a `pipeline.yaml` (i.e. skip pushes that touch compiled YAML only).

Example for AutoML (same line after `target_branch` in the other two files):

```yaml
pipelinesascode.tekton.dev/on-cel-expression: |
  event == "push"
  && target_branch == "rhoai-3.5-ea.1"
  && !files.all.all(x, x.matches('pipeline\\.yaml$'))
  && ( files.all.exists(p, !p.matches('^\\.tekton/'))
    || ".tekton/odh-automl-v3-5-ea-1-push.yaml".pathChanged()
    || "Dockerfile.konflux.automl".pathChanged() )
```

After merging in `konflux-central`, wait for sync into `pipelines-components` before relying on the
new CEL on the release branch.

## Automation notes

- **Branch scope**: trigger only on the target release branch (for example `rhoai-3.5-ea.1`).
- **No extra image builds**: apply the Konflux CEL guard above in `konflux-central`; GitHub Actions
  `paths-ignore` alone does not stop Konflux on push.
- **Reproducibility**: always pin `registry.redhat.io/...@sha256:...` at compile time using
  digests resolved from Quay for the release branch tag.

## See also

- [`scripts/generate_managed_pipelines/README.md`](../scripts/generate_managed_pipelines/README.md)
- [`utils/consts.py`](../utils/consts.py)
