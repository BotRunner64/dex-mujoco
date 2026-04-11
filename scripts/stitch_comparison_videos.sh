#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDERS_ROOT="$ROOT_DIR/recordings/full_renders"
OUTPUT_ROOT="$RENDERS_ROOT/comparisons"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
FORCE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --renders-root PATH   Full render root. Default: $RENDERS_ROOT
  --output-root PATH    Comparison output root. Default: $OUTPUT_ROOT
  --ffmpeg BIN          ffmpeg executable. Default: $FFMPEG_BIN
  --ffprobe BIN         ffprobe executable. Default: $FFPROBE_BIN
  --force               Overwrite comparison videos if they already exist
  -h, --help            Show this help

This script stitches robot render videos with their matching landmarks video:
  - right/* + landmarks/right/pico_right_landmarks.mp4
  - left/* + landmarks/left/pico_left_landmarks.mp4
  - bihand/* + landmarks/bihand/pico_bihand_landmarks.mp4

Outputs are written under:
  - <output-root>/right/
  - <output-root>/left/
  - <output-root>/bihand/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --renders-root)
            RENDERS_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --ffmpeg)
            FFMPEG_BIN="$2"
            shift 2
            ;;
        --ffprobe)
            FFPROBE_BIN="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

RIGHT_LANDMARKS="$RENDERS_ROOT/landmarks/right/pico_right_landmarks.mp4"
LEFT_LANDMARKS="$RENDERS_ROOT/landmarks/left/pico_left_landmarks.mp4"
BIHAND_LANDMARKS="$RENDERS_ROOT/landmarks/bihand/pico_bihand_landmarks.mp4"

for required_path in \
    "$RENDERS_ROOT/right" \
    "$RENDERS_ROOT/left" \
    "$RENDERS_ROOT/bihand" \
    "$RIGHT_LANDMARKS" \
    "$LEFT_LANDMARKS" \
    "$BIHAND_LANDMARKS"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT/right" "$OUTPUT_ROOT/left" "$OUTPUT_ROOT/bihand"

rendered=()
skipped=()
failed=()

stitch_group() {
    local group="$1"
    local landmarks_path="$2"
    local video_path

    for video_path in "$RENDERS_ROOT/$group"/*.mp4; do
        local name
        local output_path
        name="$(basename "$video_path")"
        output_path="$OUTPUT_ROOT/$group/$name"

        if [[ -f "$output_path" && "$FORCE" -ne 1 ]]; then
            echo "[skip] $group/$name"
            skipped+=("$output_path")
            continue
        fi

        echo "[stitch] $group/$name"
        if "$FFMPEG_BIN" -y \
            -i "$video_path" \
            -i "$landmarks_path" \
            -filter_complex \
            "[0:v]setpts=PTS-STARTPTS,scale=-2:720:flags=lanczos[robot]; \
             [1:v]setpts=PTS-STARTPTS,scale=-2:720:flags=lanczos[landmarks]; \
             [robot][landmarks]hstack=inputs=2[v]" \
            -map "[v]" \
            -map 0:a? \
            -c:v libx264 \
            -crf 18 \
            -preset veryfast \
            -pix_fmt yuv420p \
            -movflags +faststart \
            -shortest \
            "$output_path"; then
            rendered+=("$output_path")
        else
            failed+=("$video_path")
        fi
    done
}

stitch_group "right" "$RIGHT_LANDMARKS"
stitch_group "left" "$LEFT_LANDMARKS"
stitch_group "bihand" "$BIHAND_LANDMARKS"

echo
echo "Rendered: ${#rendered[@]}"
echo "Skipped: ${#skipped[@]}"
echo "Failed: ${#failed[@]}"

if [[ ${#failed[@]} -gt 0 ]]; then
    printf 'Failed videos:\n'
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
