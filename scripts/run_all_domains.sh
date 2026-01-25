#!/bin/bash
# Run scaffold evolution across all domains sequentially.
# Individual domain failures do not interrupt subsequent domains.

cd "$(dirname "$0")/.."

export N_REPEAT="${N_REPEAT:-1}"

if [ "$N_REPEAT" -gt 1 ]; then
    echo "Evolution rounds: $N_REPEAT"
fi

FAILED_DOMAINS=""

echo "========== SWE-bench (django) =========="
if ! DOMAIN=swe SWE_SUBSET=django bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS swe_django"
    echo "Warning: SWE django failed, continuing..."
fi

echo "========== SWE-bench (sympy) =========="
if ! DOMAIN=swe SWE_SUBSET=sympy bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS swe_sympy"
    echo "Warning: SWE sympy failed, continuing..."
fi

echo "========== DA-Code (ml) =========="
if ! DOMAIN=dacode DACODE_SUBSET=ml bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS dacode_ml"
    echo "Warning: DA-Code ml failed, continuing..."
fi

echo "========== DA-Code (sa) =========="
if ! DOMAIN=dacode DACODE_SUBSET=sa bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS dacode_sa"
    echo "Warning: DA-Code sa failed, continuing..."
fi

echo "========== DA-Code (dm) =========="
if ! DOMAIN=dacode DACODE_SUBSET=dm bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS dacode_dm"
    echo "Warning: DA-Code dm failed, continuing..."
fi

echo "========== DS-1000 (Numpy) =========="
if ! DOMAIN=ds1000 DS1000_SUBSET=Numpy bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS ds1000_numpy"
    echo "Warning: DS-1000 Numpy failed, continuing..."
fi

echo "========== DS-1000 (Pandas) =========="
if ! DOMAIN=ds1000 DS1000_SUBSET=Pandas bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS ds1000_pandas"
    echo "Warning: DS-1000 Pandas failed, continuing..."
fi

echo "========== DS-1000 (Matplotlib) =========="
if ! DOMAIN=ds1000 DS1000_SUBSET=Matplotlib bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS ds1000_matplotlib"
    echo "Warning: DS-1000 Matplotlib failed, continuing..."
fi

echo "========== Math =========="
if ! DOMAIN=math bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS math"
    echo "Warning: Math failed, continuing..."
fi

echo "========== AppWorld =========="
if ! DOMAIN=appworld bash scripts/run_evolve.sh; then
    FAILED_DOMAINS="$FAILED_DOMAINS appworld"
    echo "Warning: AppWorld failed, continuing..."
fi

echo ""
echo "========== Complete =========="
if [ -n "$FAILED_DOMAINS" ]; then
    echo "Failed domains:$FAILED_DOMAINS"
    exit 1
else
    echo "All domains completed successfully"
fi
