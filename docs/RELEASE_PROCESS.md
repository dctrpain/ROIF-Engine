# Release Process

## Overview

This document describes the release workflow for ROIF Engine.

The objective is to ensure that every published version is:

- reproducible;
- tested;
- documented;
- traceable.

Every release should represent a stable snapshot of the project at a specific point in its development.

---

# Release Philosophy

A release is more than a Git tag.

Each release should provide:

- working source code;
- passing test suite;
- updated documentation;
- version history;
- known limitations.

No release should knowingly include broken functionality without documenting it.

---

# Versioning

ROIF Engine follows Semantic Versioning.

```
MAJOR.MINOR.PATCH
```

Meaning:

| Version | Description |
|----------|-------------|
| MAJOR | Breaking API changes |
| MINOR | New functionality with backward compatibility |
| PATCH | Bug fixes and documentation improvements |

Examples:

```
0.1.0-alpha
0.1.0-beta
0.1.0
0.2.0
1.0.0
1.1.0
1.1.1
```

Before version `1.0.0`, breaking changes may occur as the architecture evolves.

---

# Release Types

## Alpha

Early development.

Characteristics:

- experimental features;
- incomplete functionality;
- changing API;
- active development.

---

## Beta

Feature-complete candidate.

Characteristics:

- stable architecture;
- extensive testing;
- documentation largely complete;
- bug fixing prioritized over new features.

---

## Stable

Production-quality release.

Requirements:

- public API considered stable;
- documentation complete;
- validation documented;
- known issues explicitly listed.

---

# Release Checklist

Before creating a release, confirm:

## Source Code

- [ ] No known critical defects.
- [ ] All files committed.
- [ ] Repository builds correctly.

---

## Testing

Run:

```bash
pytest
```

Confirm:

- [ ] All required tests pass.
- [ ] Expected failures are intentional.
- [ ] No new regressions.

---

## Documentation

Verify that the following documents are current:

- [ ] README.md
- [ ] PROJECT_STATUS.md
- [ ] CHANGELOG.md
- [ ] ROADMAP.md
- [ ] ARCHITECTURE.md
- [ ] MATERIALS.md
- [ ] SOLVER.md
- [ ] XPBD.md
- [ ] TESTING.md
- [ ] API.md
- [ ] THEORY.md
- [ ] VALIDATION.md
- [ ] REFERENCES.md
- [ ] CONTRIBUTING.md
- [ ] CODING_STANDARD.md
- [ ] RELEASE_PROCESS.md

---

## Version Number

Update version identifiers where appropriate.

Ensure consistency between:

- repository documentation;
- release notes;
- Git tag.

---

## Changelog

Record:

- new features;
- bug fixes;
- API changes;
- documentation updates;
- known limitations.

Every public release should have a corresponding `CHANGELOG.md` entry.

---

# Creating a Release

Commit all pending changes:

```bash
git add .
git commit -m "release: prepare v0.1.0-alpha"
```

Create an annotated tag:

```bash
git tag -a v0.1.0-alpha -m "ROIF Engine v0.1.0-alpha"
```

Push commits and tags:

```bash
git push
git push --tags
```

---

# GitHub Release

After pushing the release tag:

1. Create a new GitHub Release.
2. Select the corresponding tag.
3. Add release notes.
4. Mark the release as a pre-release if appropriate.
5. Publish the release.

Release notes should summarize the most important changes since the previous version.

---

# Regression Policy

Every bug fixed after a release should include a regression test whenever practical.

Regression tests help prevent previously solved issues from reappearing.

---

# Hotfix Releases

Critical defects may require a patch release.

Typical workflow:

```
1. Fix the defect
2. Add regression test
3. Update CHANGELOG
4. Increase PATCH version
5. Publish release
```

Example:

```
1.1.0

↓

1.1.1
```

---

# Documentation Policy

Documentation is considered part of the release.

Documentation updates should be version-controlled alongside source code.

A feature is not considered complete until its documentation has been updated.

---

# Validation Policy

New physical models should not be included in a stable release unless they have:

- implementation;
- tests;
- documented assumptions;
- documented limitations;
- validation status.

Experimental models may be included in alpha releases if clearly identified.

---

# Reproducibility

Each release should make it possible for another developer to:

1. Clone the repository.
2. Install dependencies.
3. Run the full test suite.
4. Obtain the documented results.

Reproducibility is a core project objective.

---

# Long-Term Compatibility

Public APIs should evolve conservatively.

Breaking changes should:

- be justified;
- be documented;
- be announced in release notes;
- update all affected documentation.

---

# End-of-Release Review

Before publishing, verify:

- [ ] Source code is complete.
- [ ] Tests pass.
- [ ] Documentation is synchronized.
- [ ] Version number is correct.
- [ ] CHANGELOG is updated.
- [ ] Git tag is created.
- [ ] Release notes are written.
- [ ] Known limitations are documented.

---

# Guiding Principle

Every release should represent a reliable, reproducible, and well-documented milestone in the evolution of ROIF Engine.