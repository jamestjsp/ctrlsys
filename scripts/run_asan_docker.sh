#!/bin/bash
# Run ASAN tests in Docker container
# Uses pre-built image for fast execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="slicot-asan"
NO_BUILD=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--no-build)
            NO_BUILD=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Platform detection
PLATFORM="linux/arm64"
if [[ "$(uname -m)" == "x86_64" ]]; then
    PLATFORM="linux/amd64"
fi

# Build image if needed (skip with --no-build)
if [[ "$NO_BUILD" == "false" ]] && ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "=== Building ASAN Docker image (one-time) ==="
    docker build --platform "$PLATFORM" -t "$IMAGE_NAME" -f "$PROJECT_DIR/docker/Dockerfile.asan" "$PROJECT_DIR/docker"
elif [[ "$NO_BUILD" == "true" ]] && ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Error: Image '$IMAGE_NAME' not found. Run without --no-build first."
    exit 1
fi

# Default test path
TEST_PATH="${1:-tests/python/}"
shift 2>/dev/null || true
PYTEST_ARGS="$*"

echo "=== Running ASAN tests ($PLATFORM) ==="
echo "Test: $TEST_PATH $PYTEST_ARGS"

docker run --rm \
    --privileged \
    --platform "$PLATFORM" \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace \
    -e TEST_PATH="$TEST_PATH" \
    -e PYTEST_ARGS="$PYTEST_ARGS" \
    "$IMAGE_NAME" \
    bash -c '
# Build with sanitizers
meson setup build -Db_sanitize=address,undefined -Db_lundef=false --buildtype=debug --wipe 2>/dev/null || \
meson setup build -Db_sanitize=address,undefined -Db_lundef=false --buildtype=debug
meson compile -C build
uv pip install . --reinstall --no-deps

# Run tests with ASAN
ASAN_LIB=$(gcc -print-file-name=libasan.so)
export LD_PRELOAD="$ASAN_LIB"
# More verbose ASAN options for better crash diagnosis
export ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:print_legend=1:abort_on_error=1:verbosity=1"
export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
# Python crash handler for non-ASAN crashes
export PYTHONFAULTHANDLER=1

# Disable pipefail so tee continues even if pytest crashes
set +o pipefail

# Use unbuffered output + tee; capture exit code properly
stdbuf -oL -eL pytest $TEST_PATH $PYTEST_ARGS 2>&1 | tee asan_output.txt
exit_code=${PIPESTATUS[0]}

# Ensure all output is flushed to disk
sync
sleep 0.5

exit $exit_code
'

echo "=== Done ==="
