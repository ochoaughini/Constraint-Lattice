# Contributing to Constraint-Lattice 🧩

Thank you for your interest in improving Constraint-Lattice!  We embrace an
**open-core** model:

* **MIT Core** – All code under the `clattice/`, `constraints/`, `engine/`, and
  other *core* folders is licensed under the permissive MIT license.
* **BSL SaaS Layer** – Code inside `saas/`, `api/`, `billing/`, and related
  infrastructure directories is released under the Business Source License
  v1.1 (BSL-1.1).

By participating in this project you agree that:

1. Any pull request that modifies **core** paths is contributed under the MIT
   license.
2. Any contribution that touches **SaaS** paths is contributed under the
   BSL-1.1 terms **unless** you obtain explicit written permission from
   **Lexsight LCC** designating a different license.
3. You have the authority to contribute the code and it is your original work
   (Developer Certificate of Origin – see below).

## Developer Certificate of Origin (DCO)

We follow the [DCO 1.1](https://developercertificate.org/).  Simply add a
`Signed-off-by:` trailer to every commit message, e.g.

```text
Signed-off-by: Jane Doe <jane@example.com>
```

Git makes this easy:

```bash
git commit -s -m "feat(core): add new constraint"
```

## How to Contribute

1. **Fork** the repo and create a feature branch.
2. Write tests (`pytest -q` must stay green).
3. Run `pre-commit run --all-files`.
4. Open a Pull Request – fill in the template.

### Core Pull Requests (MIT)
* Keep public APIs stable.
* Add or update documentation.
* Include unit tests.

### SaaS Pull Requests (BSL)
* Follow the same checklist but note that SaaS code may be merged into a
  separate **cloud** branch.

## Code of Conduct

Be kind, inclusive, and respectful.  See `docs/CODE_OF_CONDUCT.md`.
