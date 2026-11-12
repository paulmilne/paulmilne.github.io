#!/usr/bin/env bash
# Check that every page Quarto should build has an up-to-date page in docs/.
#   MISSING  the page has never been rendered
#   STALE    the .qmd is newer than its page
#   LISTING  (warning only) a post is newer than the home or section page that lists it
# Which files count as pages comes from the project.render list in _quarto.yml.
# Exit status 1 if anything is MISSING or STALE.
cd "$(git rev-parse --show-toplevel)" || exit 2

# Pull the render patterns out of _quarto.yml (lines like:  - "posts/**")
patterns=$(awk '/^  render:/{on=1;next} on&&/^ *- /{gsub(/^ *- *"?|"? *(#.*)?$/,"");print;next} on{exit}' _quarto.yml)
includes=(); excludes=()
while IFS= read -r p; do
  [ -z "$p" ] && continue
  if [ "${p:0:1}" = "!" ]; then excludes+=("${p:1}"); else includes+=("$p"); fi
done <<< "$patterns"

wanted() {  # is $1 in the render set?
  local f=$1 p
  for p in "${excludes[@]}"; do
    case $p in */) [[ $f == "$p"* ]] && return 1 ;; *) [[ $f == "$p" ]] && return 1 ;; esac
  done
  for p in "${includes[@]}"; do
    case $p in
      "*.qmd") [[ $f != */* ]] && return 0 ;;
      */\*\*)  [[ $f == "${p%\*\*}"* ]] && return 0 ;;
      *)       [[ $f == "$p" ]] && return 0 ;;
    esac
  done
  return 1
}

fail=0
while IFS= read -r q; do
  wanted "$q" || continue
  h="docs/${q%.qmd}.html"
  if [ ! -f "$h" ]; then echo "MISSING  $q"; fail=1
  elif [ "$q" -nt "$h" ]; then echo "STALE    $q"; fail=1
  fi
  # Listing pages: the section index and the home page should be newer than each post
  case $q in */*/index.qmd)
    sec=${q%%/*}
    for l in "docs/$sec/index.html" docs/index.html; do
      [ -f "$l" ] && [ "$q" -nt "$l" ] && echo "LISTING  ${l#docs/} is older than $q — re-render ${l#docs/}, i.e. quarto render $(dirname ${l#docs/})/index.qmd"
    done ;;
  esac
done < <(git ls-files -co --exclude-standard '*.qmd')

if [ $fail = 1 ]; then
  echo; echo "Render these (or run 'quarto render' for the whole site), then commit again."
  echo "To commit anyway: git commit --no-verify"
fi
exit $fail
