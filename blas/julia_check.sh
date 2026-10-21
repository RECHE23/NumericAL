#!/usr/bin/env bash
# Julia's LinearAlgebra tests with numerical_blas forwarded under libblastrampoline (its ILP64 names; the routines it
# lacks stay OpenBLAS's), the groups that call the routines it has, each in a process of its own. A failure fails the
# check unless it is a known divergence listed below, and a listed one that no longer fails fails it too, so the list
# stays true.
#   blas/julia_check.sh JULIA LIBRARY
set -euo pipefail
julia=$1
library=$2
groups=(blas matmul dense lapack cholesky lu qr svd eigen symmetriceigen pinv generic)
# svd's U and Vᵀ compared with gesvd's column by column: equal up to the columns' signs, which LAPACK leaves open (the
# reference's gesdd and gesvd agree on them through the same bidiagonal reduction).
known=("lapack.jl:188" "lapack.jl:190")
out=$(mktemp -d)
driver="$out/group.jl"
cat > "$driver" <<JULIA
using Test, LinearAlgebra
BLAS.lbt_forward("$library"; clear=false, suffix_hint="64_")
occursin("numerical_blas", string(BLAS.get_config())) || error("numerical_blas not forwarded: \$(BLAS.get_config())")
include(joinpath(Sys.STDLIB, "LinearAlgebra", "test", ARGS[1] * ".jl"))
JULIA
printf '%s\n' "${groups[@]}" | xargs -P "$(nproc)" -I{} sh -c "'$julia' '$driver' {} > '$out/{}.log' 2>&1; echo \"exit=\$?\" >> '$out/{}.log'"
failed=$(grep -h -o -E '(Test Failed|Error During Test) at [^ ]+' "$out"/*.log | sed -E 's/.*\/([^/]+:[0-9]+)$/\1/' | sort -u || true)
status=0
for f in $failed; do
  if ! printf '%s\n' "${known[@]}" | grep -qx "$f"; then echo "new failure: $f"; status=1; fi
done
for k in "${known[@]}"; do
  if ! grep -qx "$k" <<< "$failed"; then echo "known divergence no longer fails: $k (remove it from the list)"; status=1; fi
done
for g in "${groups[@]}"; do
  tail -1 "$out/$g.log" | grep -q '^exit=' || { echo "$g: no exit status"; status=1; }
  if grep -q '^exit=0$' "$out/$g.log"; then echo "$g: passed"; else echo "$g: failed ($(grep -c -E 'Test Failed|Error During Test' "$out/$g.log") failures)"; fi
done
[ "$status" -eq 0 ] || { for g in "${groups[@]}"; do grep -A12 -E 'Test Failed|Error During Test' "$out/$g.log" | head -40; done; }
exit "$status"
