#!/usr/bin/env bash
# Check that every page Quarto should build has an up-to-date page in docs/.
#   MISSING  the page has never been rendered
#   STALE    the .qmd is newer than its page
#   LISTING  (warning only) a post is newer than the home or section page that lists it
#   CONFIG   (warning only) _quarto.yml or styles.css changed since pages were rendered
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

# Site-wide settings (navbar, theme, render list) live in _quarto.yml and styles.css.
# If they are newer than a page, that page shows the old settings until a full render.
old=0
while IFS= read -r h; do { [ _quarto.yml -nt "$h" ] || [ styles.css -nt "$h" ]; } && old=$((old+1)); done < <(find docs -name '*.html' -not -path 'docs/site_libs/*')
[ $old -gt 0 ] && echo "CONFIG   _quarto.yml or styles.css is newer than $old page(s): run 'quarto render' for the whole site"

if [ $fail = 1 ]; then
  echo; echo "Render these (or run 'quarto render' for the whole site), then commit again."
  echo "To commit anyway: git commit --no-verify"
fi
exit $fail
