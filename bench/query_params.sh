#!/bin/sh
# Prints GMP-ECM's stage 2 parameters for every benchmark residue (stops
# each ecm run as soon as they are printed). Usage: query_params.sh DATA ECM
D=$1; E=$2
for f in $D/res/*.save; do
  case $f in *_x64.save) continue;; esac
  b1=$(sed -n 's/.*B1=\([0-9]*\);.*/\1/p' $f | head -1)
  out=$(timeout 4 $E -v -maxmem 8000 -resume $f $b1 2>&1 | grep -m3 -E "B2=|dF=|Estimated memory" | tr '\n' ' ')
  echo "$(basename $f .save) $out"
done
