#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap '_code=$?; echo "!! unify aborted with code ${_code}"; mkdir -p "${BASE_DIR}/merge-artifacts"; git status --porcelain=v2 -z > "${BASE_DIR}/merge-artifacts/forensic_${REPO:-unknown}_$(date +%s).status" 2>/dev/null || true; exit "${_code}"' ERR

BASE_DIR="${BASE_DIR:-$HOME/code}"
REPOS=(${REPOS_OVERRIDE:-"Constraint-Lattice WildCore Varkiel"})
WORKTREE_TOKEN="$(date +%s%N)"          # unique per invocation; CI can override via CI_JOB_ID
CI_IDENTITY_NAME="${CI_NAME:-CI}"
CI_IDENTITY_EMAIL="${CI_EMAIL:-ci@your-org.com}"
GPG_KEY="${GPG_SIGNING_KEY:-}"

for REPO in "${REPOS[@]}"; do
  echo ">> processing ${REPO}"
  cd "${BASE_DIR}/${REPO}"

  # assert identity to avoid ghost commits
  git config user.name  "${CI_IDENTITY_NAME}"
  git config user.email "${CI_IDENTITY_EMAIL}"

  echo ">> remote sync"
  git remote update --prune
  git checkout main
  git reset --hard origin/main

  echo ">> staging worktree"
  WORKTREE="../staging-${REPO}-${WORKTREE_TOKEN}"
  rm -rf "${WORKTREE}"
  git worktree add "${WORKTREE}" main
  cd "${WORKTREE}"

  echo ">> merging feature refs"
  for BRANCH in $(git for-each-ref --format='%(refname:strip=3)' refs/remotes/origin | grep -vE '^(main|HEAD)$'); do
    if ! git merge --no-ff --no-edit "origin/${BRANCH}"; then
      echo "${BRANCH}" >> "${BASE_DIR}/merge-artifacts/${REPO}_failed.txt"
      git merge --abort
    fi
  done

  echo ">> submodule realignment (if any)"
  git submodule update --init --recursive --remote || true

  echo ">> change classification"
  CHANGED_PY=$(git diff --name-only --cached -- '*.py' || true)
  if [[ -n "${CHANGED_PY}" ]]; then
    echo ">> formatting python surface"
    python -m pip install --quiet --upgrade pip setuptools wheel
    python -m black . && python -m isort .
    echo ">> executing tests"
    pytest -q
  else
    echo ">> no python deltas detected; skipping format and tests"
  fi

  echo ">> commit & push"
  git add -u
  COMMIT_FLAGS=(--author="${CI_IDENTITY_NAME} <${CI_IDENTITY_EMAIL}>" \
                --date="$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
  [[ -n "${GPG_KEY}" ]] && COMMIT_FLAGS+=(--gpg-sign="${GPG_KEY}")
  git commit -m "refactor: merged active branches and harmonized ${REPO}" "${COMMIT_FLAGS[@]}"
  git push origin main

  echo ">> provenance emission"
  mkdir -p "${BASE_DIR}/merge-artifacts"
  sha=$(git rev-parse HEAD)
  echo "$sha" > "${BASE_DIR}/merge-artifacts/${REPO}_sha.txt"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "${REPO}_sha=${sha}" >> "$GITHUB_OUTPUT"
  fi

  echo ">> vacuum staging residue"
  cd ..
  git worktree remove --force "${WORKTREE}"
  git gc --prune=now --aggressive
done

echo "OK unified synthesis complete"
