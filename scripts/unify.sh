#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap '_code=$?; echo "\u2716 unify aborted with code ${_code}"; mkdir -p "${BASE_DIR}/merge-artifacts"; git status --porcelain=v2 -z > "${BASE_DIR}/merge-artifacts/forensic_${REPO:-unknown}_$(date +%s).status" 2>/dev/null || true; exit "${_code}"' ERR

BASE_DIR="${BASE_DIR:-$HOME/code}"
REPOS=(${REPOS_OVERRIDE:-"Constraint-Lattice WildCore Varkiel"})
WORKTREE_TOKEN="$(date +%s%N)"          # unique per invocation; CI can override via CI_JOB_ID
CI_IDENTITY_NAME="${CI_NAME:-CI}"
CI_IDENTITY_EMAIL="${CI_EMAIL:-ci@your-org.com}"
GPG_KEY="${GPG_SIGNING_KEY:-}"

for REPO in "${REPOS[@]}"; do
  echo "\u2192 processing ${REPO}"
  cd "${BASE_DIR}/${REPO}"

  # assert identity to avoid ghost commits
  git config user.name  "${CI_IDENTITY_NAME}"
  git config user.email "${CI_IDENTITY_EMAIL}"

  echo "\u2192 remote sync"
  git remote update --prune

  echo "\u2192 staging worktree"
  WORKTREE="../staging-${REPO}-${WORKTREE_TOKEN}"
  rm -rf "${WORKTREE}"
  git worktree add "${WORKTREE}" main
  cd "${WORKTREE}"

  echo "\u2192 merging feature refs"
  for BRANCH in $(git for-each-ref --format='%(refname:strip=3)' refs/remotes/origin | grep -vE '^(main|HEAD)$'); do
    git merge --no-ff --no-edit "origin/${BRANCH}"
  done

  echo "\u2192 submodule realignment (if any)"
  git submodule update --init --recursive --remote || true

  echo "\u2192 change classification"
  CHANGED_PY=$(git diff --name-only --cached -- '*.py' || true)
  if [[ -n "${CHANGED_PY}" ]]; then
    echo "\u2192 formatting python surface"
    python -m pip install --quiet --upgrade pip setuptools wheel
    python -m black . && python -m isort .
    echo "\u2192 executing tests"
    pytest -q
  else
    echo "\u2192 no python deltas detected; skipping format and tests"
  fi

  echo "\u2192 commit & push"
  git add .
  COMMIT_FLAGS=(--author="${CI_IDENTITY_NAME} <${CI_IDENTITY_EMAIL}>" \
                --date="$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
  [[ -n "${GPG_KEY}" ]] && COMMIT_FLAGS+=(--gpg-sign="${GPG_KEY}")
  git commit -m "refactor: merged active branches and harmonized ${REPO}" "${COMMIT_FLAGS[@]}"
  git push origin main

  echo "\u2192 provenance emission"
  mkdir -p "${BASE_DIR}/merge-artifacts"
  git rev-parse HEAD > "${BASE_DIR}/merge-artifacts/${REPO}_sha.txt"

  echo "\u2192 vacuum staging residue"
  cd ..
  git worktree remove --force "${WORKTREE}"
  git gc --prune=now --aggressive
done

echo "\u2714 unified synthesis complete"
