# -*- coding: utf-8 -*-
"""Interactive OpenGL visualization of a pi-digit trajectory with borderless GL point sprites.

The first digit initializes the direction. Every following digit adds
36 degrees per digit value and produces one unit-length movement.

All input digits are processed and every trajectory position is retained in
CPU memory. An overview sample stays visible while zooming at overview scale.
Viewport detail is uploaded only after camera movement has stopped.

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import math

import numpy as np

from pi_trajectory_mp import full_trajectory_parallel, physical_core_count


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DIGITS_FILE = BASE_DIR / "Pi_1M.txt"

# CPU processing: larger chunks reduce Python-loop overhead, at the cost
# of somewhat higher temporary memory use.
CHUNK_SIZE = 2_000_000

# For very large files, use one worker per physical core, targeting roughly
# 100 million digits per worker. Smaller files stay single-process because
# multiprocessing startup and shared-memory setup outweigh the benefit.
PARALLEL_ENABLED = True
PARALLEL_WORKERS = 8
PARALLEL_MIN_DIGITS = 200_000_000

# Initial overview point cloud. It stays visible while the user zooms and pans
# at overview scale; the trajectory is not rectangle-culled at this stage.
INITIAL_GPU_POINTS = 1_000_000

# Switch to viewport detail only when the larger visible world dimension is
# below this threshold.
DETAIL_ZOOM_THRESHOLD = 1_000.0

# Hard upload cap for Intel integrated graphics. Each point uploads 2 float32
# coordinates plus 4 float32 color channels (24 bytes total); 2M points are
# approximately 46 MiB, rather than hundreds of MiB at deeper zoom levels.
MAX_GPU_POINTS: int | None = 2_000_000

# Wait until the camera has stopped moving before selecting and uploading a
# replacement point cloud. The existing cloud remains rendered meanwhile.
VIEW_REFRESH_DEBOUNCE_SECONDS = 0.25
VIEW_REFRESH_POLL_SECONDS = 0.05

# Select a margin around the actual camera rectangle. This keeps points just
# outside the viewport resident during small camera-coordinate discrepancies
# and avoids a visible selection edge at the window boundary.
VIEW_CULL_PADDING = 0.25

# Spatial index granularity for fast viewport queries. Each index entry
# stores the bounding box of this many consecutive trajectory positions.
VIEW_INDEX_CHUNK_POINTS = 25_000

# Dark background retained. Point colors are deliberately softened so the
# trajectory does not look like neon dots against a near-black canvas.
CANVAS_SIZE = (1440, 960)
BACKGROUND = "#121820"
ARROW_COLOR = "#83909D"
POINT_SIZE = 4.0
COLOR_SATURATION = 0.90  # Keep the chronological hues vivid.
COLOR_BRIGHTNESS = 1.00  # Use the full RGB brightness range.
COLOR_GAMMA = 0.72       # Lift darker color channels for visibility.
POINT_ALPHA = 1.00       # Opaque points: avoid dimming against the dark canvas.

# "time": chronological color gradient; "digit": color by digit 0-9.
COLOR_MODE = "time"

# Repeat the complete chronological color spectrum after this many input
# digits. For 100 million digits and 10 repetitions, use 10 million.
SPECTRUM_CYCLE_POINTS = 200_000

# Direction arrows summarize the ANGULAR CHANGES encoded by the digits in a
# local window. Their length represents circular-mean strength (0 to 1).
SHOW_ARROWS = False
ARROW_MAX_COUNT = 500
ARROW_WINDOW_POINTS = 1_000
ARROW_MAX_LENGTH = 10.0
ARROW_MIN_STRENGTH = 0.05
ARROW_WIDTH = 1.0

# Ten possible directions: 0, 36, ..., 324 degrees.
_DIRECTION_ANGLES = np.arange(10, dtype=np.float64) * (2.0 * np.pi / 10.0)
_STEP_X = np.cos(_DIRECTION_ANGLES)
_STEP_Y = np.sin(_DIRECTION_ANGLES)


def load_digits(path: Path) -> np.ndarray:
    """Read ASCII digits in chunks to limit temporary memory use."""
    if not path.is_file():
        raise FileNotFoundError(f"Pi digit file not found: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("The input file is empty.")

    # A byte can contain at most one digit, so file size is a safe upper
    # bound for the output allocation. Read in chunks instead of loading
    # the complete text file and creating a full-size Boolean mask.
    digit_buffer = np.empty(file_size, dtype=np.uint8)
    written = 0
    read_chunk_size = 8 * 1024 * 1024

    with path.open("rb") as file:
        while raw_bytes := file.read(read_chunk_size):
            raw = np.frombuffer(raw_bytes, dtype=np.uint8)
            mask = (raw >= ord("0")) & (raw <= ord("9"))
            chunk_digits = raw[mask] - ord("0")
            count = int(chunk_digits.size)
            digit_buffer[written:written + count] = chunk_digits
            written += count

    if written < 2:
        raise ValueError("The input file must contain at least two digits.")

    # Return a view; no second 100-million-byte copy is needed.
    return digit_buffer[:written]


def full_trajectory_sequential(
    digits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate and retain every path position plus chunk bounding boxes.

    A float32 XY array uses 8 bytes per input digit (about 800 MB for
    100 million digits). Bounding boxes make later viewport selection much
    faster than scanning the complete trajectory after every zoom or pan.
    """
    digit_count = int(digits.size)
    if digit_count < 2:
        raise ValueError("At least two digits are required.")
    if VIEW_INDEX_CHUNK_POINTS < 1:
        raise ValueError("VIEW_INDEX_CHUNK_POINTS must be positive.")

    try:
        positions = np.empty((digit_count, 2), dtype=np.float32)
    except MemoryError as exc:
        required_gib = digit_count * 2 * np.dtype(np.float32).itemsize / 2**30
        raise MemoryError(
            f"The full trajectory needs approximately {required_gib:.2f} GiB "
            "of RAM for coordinates."
        ) from exc

    positions[0] = (0.0, 0.0)

    index_chunk_count = (
        digit_count + VIEW_INDEX_CHUNK_POINTS - 1
    ) // VIEW_INDEX_CHUNK_POINTS
    # Columns: min_x, max_x, min_y, max_y.
    chunk_bounds = np.empty((index_chunk_count, 4), dtype=np.float32)
    chunk_bounds[:, 0] = np.inf
    chunk_bounds[:, 1] = -np.inf
    chunk_bounds[:, 2] = np.inf
    chunk_bounds[:, 3] = -np.inf
    # The first spatial chunk also contains the origin.
    chunk_bounds[0] = (0.0, 0.0, 0.0, 0.0)

    angle_state = int(digits[0])
    x_state = 0.0
    y_state = 0.0

    total_chunks = (digit_count - 1 + CHUNK_SIZE - 1) // CHUNK_SIZE
    report_interval = max(1, total_chunks // 10)
    calculation_start = perf_counter()

    for chunk_number, start in enumerate(
        range(1, digit_count, CHUNK_SIZE), start=1
    ):
        stop = min(start + CHUNK_SIZE, digit_count)
        chunk_digits = digits[start:stop]

        # Direction after each digit: cumulative sum modulo 10.
        angle_units = np.cumsum(chunk_digits, dtype=np.uint32)
        angle_units += angle_state
        np.remainder(angle_units, 10, out=angle_units)

        # Calculate the unit movement vectors, then local cumulative path.
        step_x = _STEP_X[angle_units]
        step_y = _STEP_Y[angle_units]
        local_x = np.cumsum(step_x, dtype=np.float64)
        local_y = np.cumsum(step_y, dtype=np.float64)

        positions[start:stop, 0] = x_state + local_x
        positions[start:stop, 1] = y_state + local_y

        # Update spatial-index bounds for every index block touched by this
        # calculation block. The two block sizes need not align.
        first_index_chunk = start // VIEW_INDEX_CHUNK_POINTS
        last_index_chunk = (stop - 1) // VIEW_INDEX_CHUNK_POINTS
        for index_chunk in range(first_index_chunk, last_index_chunk + 1):
            box_start = max(
                start, index_chunk * VIEW_INDEX_CHUNK_POINTS
            )
            box_stop = min(
                stop, (index_chunk + 1) * VIEW_INDEX_CHUNK_POINTS
            )
            block = positions[box_start:box_stop]
            bounds = chunk_bounds[index_chunk]
            bounds[0] = min(bounds[0], float(block[:, 0].min()))
            bounds[1] = max(bounds[1], float(block[:, 0].max()))
            bounds[2] = min(bounds[2], float(block[:, 1].min()))
            bounds[3] = max(bounds[3], float(block[:, 1].max()))

        x_state += float(local_x[-1])
        y_state += float(local_y[-1])
        angle_state = int(angle_units[-1])

        if (
            chunk_number % report_interval == 0
            or chunk_number == total_chunks
        ):
            processed = stop - 1
            elapsed = perf_counter() - calculation_start
            percent = processed / (digit_count - 1) * 100.0
            print(
                f"  Trajectory: {percent:5.1f}% "
                f"({processed:,}/{digit_count - 1:,} digits), "
                f"{elapsed:.1f} s"
            )

    if not np.all(np.isfinite(chunk_bounds)):
        raise RuntimeError("Spatial index contains an uninitialized chunk.")

    return positions, chunk_bounds


def full_trajectory(
    digits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the sequential or shared-memory multiprocessing backend."""
    digit_count = int(digits.size)
    movement_count = max(0, digit_count - 1)

    if not PARALLEL_ENABLED or movement_count < PARALLEL_MIN_DIGITS:
        print("  Trajectory backend: single-process NumPy")
        return full_trajectory_sequential(digits)

    worker_count = min(PARALLEL_WORKERS, movement_count)
    detection_method = "manually configured"

    if worker_count < 2:
        print(
            f"  Trajectory backend: single-process NumPy "
            f"(detected {physical_cores} CPU core(s); {detection_method})"
        )
        return full_trajectory_sequential(digits)

    print(
        f"  Trajectory backend: multiprocessing with {worker_count} workers "
        f"({detection_method})"
    )
    try:
        return full_trajectory_parallel(
            digits,
            worker_count=worker_count,
            chunk_size=CHUNK_SIZE,
            index_chunk_points=VIEW_INDEX_CHUNK_POINTS,
            progress=True,
        )
    except (OSError, MemoryError) as exc:
        raise MemoryError(
            "Could not allocate the shared-memory input/output required for "
            f"{worker_count} trajectory workers. The parallel run needs about "
            f"{digits.nbytes / 2**30:.2f} GiB for shared digits plus "
            f"{digit_count * 8 / 2**30:.2f} GiB for shared XY positions, "
            "in addition to the original digit array and working memory. "
            "Free memory/pagefile space or set PARALLEL_ENABLED = False."
        ) from exc


def evenly_spaced_indices(
    point_count: int,
    max_points: int,
) -> np.ndarray:
    """Return up to max_points indices distributed over the full path."""
    count = min(int(point_count), int(max_points))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == point_count:
        return np.arange(point_count, dtype=np.int64)
    return np.linspace(0, point_count - 1, count, dtype=np.int64)


def dynamic_point_budget(
    visible_world_width: float,
    visible_world_height: float,
    total_points: int,
) -> int:
    """Choose detail from the visible world-space rectangle.

    Keep the initial sample density while the viewport is larger than
    DETAIL_ZOOM_THRESHOLD in either dimension. Once the viewport is smaller,
    grow the budget inversely with its area (a 2-D view), allowing more than
    the former fixed 12-million-point limit when the input and GPU permit it.
    """
    if (
        visible_world_width <= 0
        or visible_world_height <= 0
        or total_points <= 0
    ):
        return min(INITIAL_GPU_POINTS, max(0, int(total_points)))

    visible_span = max(visible_world_width, visible_world_height)
    if visible_span >= DETAIL_ZOOM_THRESHOLD:
        budget = INITIAL_GPU_POINTS
    else:
        zoom_ratio = DETAIL_ZOOM_THRESHOLD / visible_span
        # In a 2-D viewport, halving both dimensions reduces visible area by
        # four; increase the point budget by the inverse area ratio.
        budget = int(np.ceil(INITIAL_GPU_POINTS * zoom_ratio * zoom_ratio))

    if MAX_GPU_POINTS is not None:
        budget = min(budget, MAX_GPU_POINTS)
    return min(max(1, budget), int(total_points))


def visible_point_indices(
    positions: np.ndarray,
    chunk_bounds: np.ndarray,
    view_bounds: tuple[float, float, float, float],
    max_points: int,
) -> np.ndarray:
    """Find points in the viewport without building an oversized index list."""
    min_x, max_x, min_y, max_y = view_bounds
    if max_points < 1:
        return np.empty(0, dtype=np.int64)

    # If the complete path fits in the viewport, avoid scanning all points.
    path_bounds = (
        float(chunk_bounds[:, 0].min()),
        float(chunk_bounds[:, 1].max()),
        float(chunk_bounds[:, 2].min()),
        float(chunk_bounds[:, 3].max()),
    )
    if (
        min_x <= path_bounds[0]
        and max_x >= path_bounds[1]
        and min_y <= path_bounds[2]
        and max_y >= path_bounds[3]
    ):
        return evenly_spaced_indices(len(positions), max_points)

    candidate_chunks = np.flatnonzero(
        (chunk_bounds[:, 1] >= min_x)
        & (chunk_bounds[:, 0] <= max_x)
        & (chunk_bounds[:, 3] >= min_y)
        & (chunk_bounds[:, 2] <= max_y)
    )
    if candidate_chunks.size == 0:
        return np.empty(0, dtype=np.int64)

    def local_visible_mask(chunk_index: int) -> np.ndarray:
        start = chunk_index * VIEW_INDEX_CHUNK_POINTS
        stop = min(start + VIEW_INDEX_CHUNK_POINTS, len(positions))
        block = positions[start:stop]
        return (
            (block[:, 0] >= min_x)
            & (block[:, 0] <= max_x)
            & (block[:, 1] >= min_y)
            & (block[:, 1] <= max_y)
        )

    # First pass counts only. Do not allocate an int64 index array just to
    # throw it away; index arrays are materialized only in the selection pass.
    visible_count = sum(
        int(np.count_nonzero(local_visible_mask(int(chunk_index))))
        for chunk_index in candidate_chunks
    )
    if visible_count == 0:
        return np.empty(0, dtype=np.int64)

    if visible_count <= max_points:
        visible_parts: list[np.ndarray] = []
        for chunk_index in candidate_chunks:
            local_indices = np.flatnonzero(
                local_visible_mask(int(chunk_index))
            )
            if local_indices.size:
                start = int(chunk_index) * VIEW_INDEX_CHUNK_POINTS
                visible_parts.append(local_indices + start)
        return np.concatenate(visible_parts)

    # Pick exactly max_points ranks from the chronological list of visible
    # points, but materialize only the selected indices.
    target_ranks = np.linspace(
        0, visible_count - 1, int(max_points), dtype=np.int64
    )
    selected = np.empty(target_ranks.size, dtype=np.int64)
    target_cursor = 0
    visible_offset = 0

    for chunk_index in candidate_chunks:
        chunk_index = int(chunk_index)
        local_indices = np.flatnonzero(local_visible_mask(chunk_index))
        next_offset = visible_offset + local_indices.size
        next_cursor = int(
            np.searchsorted(target_ranks, next_offset, side="left")
        )
        if next_cursor > target_cursor:
            local_ranks = target_ranks[target_cursor:next_cursor] - visible_offset
            start = chunk_index * VIEW_INDEX_CHUNK_POINTS
            selected[target_cursor:next_cursor] = (
                local_indices[local_ranks] + start
            )
        target_cursor = next_cursor
        visible_offset = next_offset

    if target_cursor != target_ranks.size:
        raise RuntimeError("Failed to assemble all selected viewport points.")
    return selected


def hsv_to_rgb(hues: np.ndarray) -> np.ndarray:
    """Convert hue values in [0, 1] to RGB float32 colors."""
    h6 = np.remainder(hues, 1.0) * 6.0
    sectors = np.floor(h6).astype(np.int8)
    fraction = h6 - sectors

    p = np.zeros_like(fraction)
    q = 1.0 - fraction
    t = fraction

    rgb = np.empty((hues.size, 3), dtype=np.float32)
    sector = sectors % 6

    masks = (
        sector == 0,
        sector == 1,
        sector == 2,
        sector == 3,
        sector == 4,
        sector == 5,
    )
    values = (
        (1.0, t, p),
        (q, 1.0, p),
        (p, 1.0, t),
        (p, q, 1.0),
        (t, p, 1.0),
        (1.0, p, q),
    )

    for mask, channels in zip(masks, values):
        for channel_index, channel in enumerate(channels):
            if isinstance(channel, np.ndarray):
                rgb[mask, channel_index] = channel[mask]
            else:
                rgb[mask, channel_index] = channel

    # Keep the dark canvas, but make the points clearly visible:
    # retain strong hue separation, use full brightness, and lift darker
    # RGB channels with a gamma curve. Point opacity is set separately.
    rgb = (1.0 - COLOR_SATURATION) + COLOR_SATURATION * rgb
    rgb *= COLOR_BRIGHTNESS
    np.maximum(rgb, 0.0, out=rgb)
    np.power(rgb, COLOR_GAMMA, out=rgb)
    np.clip(rgb, 0.0, 1.0, out=rgb)

    return rgb


def make_colors_for_indices(
    point_indices: np.ndarray,
    digits: np.ndarray,
) -> np.ndarray:
    """Create colors for arbitrary original trajectory indices."""
    point_count = int(point_indices.size)

    if COLOR_MODE == "time":
        if SPECTRUM_CYCLE_POINTS < 1:
            raise ValueError("SPECTRUM_CYCLE_POINTS must be positive.")
        hues = np.remainder(
            point_indices.astype(np.float64), SPECTRUM_CYCLE_POINTS
        ) / SPECTRUM_CYCLE_POINTS
    elif COLOR_MODE == "digit":
        hues = digits[point_indices].astype(np.float32) / 10.0
    else:
        raise ValueError("COLOR_MODE must be 'time' or 'digit'.")

    rgb = hsv_to_rgb(hues)
    rgba = np.empty((point_count, 4), dtype=np.float32)
    rgba[:, :3] = rgb
    rgba[:, 3] = POINT_ALPHA
    return rgba


def create_filled_point_cloud(
    positions: np.ndarray,
    colors: np.ndarray,
    point_size: float,
    parent,
):
    """Create borderless circular GL points using a minimal custom shader.

    VisPy's Markers visual draws a separate edge layer. Some drivers continue
    to show a dark fringe even when edge_width is zero or edge alpha is zero.
    This visual has no edge-color or outline code at all: each GL point is a
    solid, hard-edged circle filled only with its per-point RGBA color.
    """
    from vispy import gloo
    from vispy.scene.visuals import create_visual_node
    from vispy.visuals import Visual

    vertex_shader = """
    attribute vec2 a_position;
    attribute vec4 a_color;
    uniform float u_point_size;
    varying vec4 v_color;

    void main() {
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
        gl_PointSize = u_point_size;
        v_color = a_color;
    }
    """

    fragment_shader = """
    varying vec4 v_color;

    void main() {
        vec2 p = gl_PointCoord - vec2(0.5, 0.5);
        if (dot(p, p) > 0.25) {
            discard;
        }
        // No edge layer, stroke, shading, or dark antialias ring.
        gl_FragColor = vec4(v_color.rgb, v_color.a);
    }
    """

    class FilledPointVisual(Visual):
        def __init__(self, point_positions, point_colors, size):
            Visual.__init__(self, vcode=vertex_shader, fcode=fragment_shader)
            # Visual.draw() requires an explicit primitive mode. The custom
            # visual renders one GL point per trajectory position.
            self._draw_mode = "points"
            self._position_buffer = gloo.VertexBuffer(point_positions)
            self._color_buffer = gloo.VertexBuffer(point_colors)
            self._point_size = float(size)
            self._point_positions = point_positions

        def set_data(self, point_positions, point_colors):
            """Replace GPU buffers after a camera zoom or pan."""
            self._position_buffer.set_data(point_positions)
            self._color_buffer.set_data(point_colors)
            self._point_positions = point_positions
            self.update()

        def _prepare_transforms(self, view):
            # $transform is a VisPy shader template variable. Bind it on the
            # view program using the transform API documented by VisPy.
            view.view_program.vert["transform"] = (
                view.transforms.get_transform()
            )

        def _prepare_draw(self, view):
            # Bind GLSL attributes through MultiProgram.__setitem__.
            # Using shared_program.vert["a_position"] incorrectly treats
            # the attribute name as a shader-template variable and raises
            # KeyError: Invalid template variable 'a_position'.
            self.shared_program["a_position"] = self._position_buffer
            self.shared_program["a_color"] = self._color_buffer
            self.shared_program["u_point_size"] = self._point_size
            return True

        def _compute_bounds(self, axis, view):
            if axis not in (0, 1):
                return None
            values = self._point_positions[:, axis]
            return float(values.min()), float(values.max())

        def _draw(self, transforms):
            self.shared_program.draw("points")

    FilledPointNode = create_visual_node(FilledPointVisual)
    return FilledPointNode(positions, colors, point_size, parent=parent)


def make_arrow_segments(
    digits: np.ndarray,
    positions: np.ndarray,
    stride: int,
) -> np.ndarray | None:
    """Create arrows from local circular means of the digit angles.

    Each input digit d represents an angular change of d * 36 degrees.
    For each local window, the circular mean gives the dominant direction;
    the mean resultant length gives directional agreement. Arrow length is
    ARROW_MAX_LENGTH * resultant_strength, so weak or cancelling directions
    produce short arrows (or no arrow below ARROW_MIN_STRENGTH).

    Arrow locations are spread across the complete displayed trajectory.
    Only a small fixed number of arrows is rendered, even for 100M digits.
    The first digit is excluded because it initializes the starting direction
    rather than representing a movement's angular change.
    """
    if ARROW_MAX_COUNT < 1:
        raise ValueError("ARROW_MAX_COUNT must be at least 1.")
    if ARROW_WINDOW_POINTS < 1:
        raise ValueError("ARROW_WINDOW_POINTS must be at least 1.")
    if ARROW_MAX_LENGTH <= 0:
        raise ValueError("ARROW_MAX_LENGTH must be positive.")
    if not 0.0 <= ARROW_MIN_STRENGTH <= 1.0:
        raise ValueError("ARROW_MIN_STRENGTH must be between 0 and 1.")

    digit_count = int(digits.size)
    if digit_count < 2 or len(positions) < 2:
        return None

    # Windows contain movement digits only (input indices 1..N-1).
    window_points = min(ARROW_WINDOW_POINTS, digit_count - 1)
    before = window_points // 2
    after = window_points - before

    # A sampled position at index j corresponds to original digit index
    # j * stride. Select only centers with a complete local window.
    min_center = 1 + before
    max_center = digit_count - after
    min_sample_index = (min_center + stride - 1) // stride
    max_sample_index = min(
        max_center // stride,
        len(positions) - 1,
    )

    if max_sample_index < min_sample_index:
        return None

    available_centers = max_sample_index - min_sample_index + 1
    arrow_count = min(ARROW_MAX_COUNT, available_centers)
    if arrow_count == 1:
        center_samples = np.array(
            [(min_sample_index + max_sample_index) // 2],
            dtype=np.int64,
        )
    else:
        center_samples = np.linspace(
            min_sample_index,
            max_sample_index,
            num=arrow_count,
            dtype=np.int64,
        )

    center_digits = center_samples * stride
    window_starts = center_digits - before
    window_indices = (
        window_starts[:, None]
        + np.arange(window_points, dtype=np.int64)[None, :]
    )
    local_digits = digits[window_indices]

    # The ten unit vectors correspond to digit angles 0, 36, ..., 324°.
    # Their average is the circular-mean resultant vector (C, S).
    mean_x = _STEP_X[local_digits].mean(axis=1)
    mean_y = _STEP_Y[local_digits].mean(axis=1)
    strength = np.hypot(mean_x, mean_y)

    # The resultant vector already combines direction and strength. Its
    # magnitude is R and its angle is atan2(mean_y, mean_x).
    valid = strength >= ARROW_MIN_STRENGTH
    if not np.any(valid):
        return None

    starts = positions[center_samples[valid]].astype(np.float64, copy=False)
    resultant = np.column_stack((mean_x[valid], mean_y[valid]))
    tips = starts + resultant * ARROW_MAX_LENGTH

    valid_strength = strength[valid]
    unit_directions = resultant / valid_strength[:, None]
    arrow_lengths = ARROW_MAX_LENGTH * valid_strength

    perpendicular = np.column_stack(
        (-unit_directions[:, 1], unit_directions[:, 0])
    )
    head_lengths = arrow_lengths * 0.30
    head_half_widths = arrow_lengths * 0.13
    head_base = tips - unit_directions * head_lengths[:, None]
    head_offset = perpendicular * head_half_widths[:, None]
    head_a = head_base + head_offset
    head_b = head_base - head_offset

    segments = np.empty((starts.shape[0] * 6, 2), dtype=np.float32)
    segments[0::6] = starts
    segments[1::6] = tips
    segments[2::6] = tips
    segments[3::6] = head_a
    segments[4::6] = tips
    segments[5::6] = head_b
    return segments



def _grid_coordinates(
    lower: float,
    upper: float,
    spacing: int,
    *,
    exclude_major: bool = False,
    major_spacing: int = 1_000,
) -> np.ndarray:
    """Return grid coordinates within the requested interval."""
    first = int(np.ceil(lower / spacing))
    last = int(np.floor(upper / spacing))
    if last < first:
        return np.empty(0, dtype=np.float32)

    values = np.arange(first, last + 1, dtype=np.int64) * spacing
    values = values[values != 0]
    if exclude_major:
        values = values[values % major_spacing != 0]
    return values.astype(np.float32)


def _make_grid_segments(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Build independent vertical and horizontal line segments."""
    min_x, max_x, min_y, max_y = bounds
    segment_count = int(x_coordinates.size + y_coordinates.size)
    segments = np.empty((segment_count * 2, 2), dtype=np.float32)

    split = int(x_coordinates.size * 2)
    if x_coordinates.size:
        segments[0:split:2, 0] = x_coordinates
        segments[0:split:2, 1] = min_y
        segments[1:split:2, 0] = x_coordinates
        segments[1:split:2, 1] = max_y

    if y_coordinates.size:
        segments[split::2, 0] = min_x
        segments[split::2, 1] = y_coordinates
        segments[split + 1::2, 0] = max_x
        segments[split + 1::2, 1] = y_coordinates

    return segments


def add_coordinate_grid(
    parent,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw the 100-unit minor grid and 1,000-unit major grid."""
    from vispy import scene as vispy_scene

    min_x, max_x, min_y, max_y = bounds

    minor_x = _grid_coordinates(
        min_x, max_x, 100, exclude_major=True
    )
    minor_y = _grid_coordinates(
        min_y, max_y, 100, exclude_major=True
    )
    minor_segments = _make_grid_segments(
        minor_x, minor_y, bounds
    )
    if minor_segments.size:
        vispy_scene.visuals.Line(
            pos=minor_segments,
            connect="segments",
            color="#27333D",
            width=0.6,
            parent=parent,
        )

    major_x = _grid_coordinates(min_x, max_x, 1_000)
    major_y = _grid_coordinates(min_y, max_y, 1_000)
    major_segments = _make_grid_segments(
        major_x, major_y, bounds
    )
    if major_segments.size:
        vispy_scene.visuals.Line(
            pos=major_segments,
            connect="segments",
            color="#465562",
            width=1.0,
            parent=parent,
        )


def add_coordinate_axes(
    parent,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw the X=0 and Y=0 axes above the trajectory."""
    from vispy import scene as vispy_scene

    min_x, max_x, min_y, max_y = bounds
    axes = np.array(
        [
            [0.0, min_y], [0.0, max_y],
            [min_x, 0.0], [max_x, 0.0],
        ],
        dtype=np.float32,
    )
    vispy_scene.visuals.Line(
        pos=axes,
        connect="segments",
        color="#C0CBD5",
        width=2.5,
        parent=parent,
    )


def square_view_bounds(
    positions: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return square camera bounds that include the origin and the path."""
    min_x = min(float(np.min(positions[:, 0])), 0.0)
    max_x = max(float(np.max(positions[:, 0])), 0.0)
    min_y = min(float(np.min(positions[:, 1])), 0.0)
    max_y = max(float(np.max(positions[:, 1])), 0.0)

    span_x = max_x - min_x
    span_y = max_y - min_y
    padding = max(span_x, span_y) * 0.02
    half_extent = max(span_x, span_y) / 2.0 + padding
    half_extent = max(half_extent, 150.0)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    return (
        center_x - half_extent,
        center_x + half_extent,
        center_y - half_extent,
        center_y + half_extent,
    )

def main() -> None:
    start_time = perf_counter()

    print(f"Loading digits: {DIGITS_FILE}")
    digits = load_digits(DIGITS_FILE)
    loaded_time = perf_counter()
    print(
        f"Loaded {digits.size:,} digits "
        f"in {loaded_time - start_time:.2f} s"
    )

    print("Calculating the full trajectory and spatial index...")
    positions, chunk_bounds = full_trajectory(digits)
    calculated_time = perf_counter()
    print(
        f"Prepared {len(positions):,} trajectory positions in "
        f"{calculated_time - loaded_time:.2f} s; "
        f"coordinate memory: {positions.nbytes / 2**30:.2f} GiB"
    )

    try:
        from vispy import app, scene
    except ImportError as exc:
        raise SystemExit(
            "VisPy is not installed. Run: pip install vispy PyQt6"
        ) from exc

    print("Opening OpenGL window...")
    canvas = scene.SceneCanvas(
        title=f"Pi digit trajectory — {digits.size:,} digits",
        keys="interactive",
        bgcolor=BACKGROUND,
        size=CANVAS_SIZE,
        show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)

    # Use equal X/Y world scales and include the origin in the initial view.
    view_bounds = square_view_bounds(positions)
    add_coordinate_grid(view.scene, view_bounds)
    min_x, max_x, min_y, max_y = view_bounds

    # Keep the overview arrays in CPU memory so zooming back out can restore
    # the original full-trajectory sample without recalculating its colors.
    overview_indices = evenly_spaced_indices(
        len(positions), INITIAL_GPU_POINTS
    )
    overview_positions = positions[overview_indices]
    overview_colors = make_colors_for_indices(overview_indices, digits)
    point_cloud = create_filled_point_cloud(
        overview_positions,
        overview_colors,
        point_size=POINT_SIZE,
        parent=view.scene,
    )
    del overview_indices

    if SHOW_ARROWS:
        arrow_segments = make_arrow_segments(
            digits, positions, stride=1
        )
        if arrow_segments is not None:
            scene.visuals.Line(
                pos=arrow_segments,
                connect="segments",
                color=ARROW_COLOR,
                width=ARROW_WIDTH,
                parent=view.scene,
            )

    # Render the zero axes last so the origin remains clearly identifiable.
    add_coordinate_axes(view.scene, view_bounds)

    view.camera.set_range(
        x=(min_x, max_x),
        y=(min_y, max_y),
    )

    last_observed_bounds: tuple[float, float, float, float] | None = None
    last_camera_change_time = perf_counter()
    last_uploaded_bounds: tuple[float, float, float, float] | None = None
    displayed_mode = "overview"

    def read_camera_bounds() -> tuple[float, float, float, float] | None:
        """Return the camera rectangle in scene/world coordinates."""
        camera_rect = view.camera.rect
        if camera_rect is None:
            return None

        rect_pos = np.asarray(camera_rect.pos, dtype=np.float64)
        rect_size = np.asarray(camera_rect.size, dtype=np.float64)
        min_x = float(rect_pos[0])
        min_y = float(rect_pos[1])
        max_x = min_x + float(rect_size[0])
        max_y = min_y + float(rect_size[1])
        if max_x <= min_x or max_y <= min_y:
            return None
        return min_x, max_x, min_y, max_y

    def refresh_visible_points() -> None:
        """Refresh detail only after the camera has been stationary."""
        nonlocal last_observed_bounds
        nonlocal last_camera_change_time
        nonlocal last_uploaded_bounds
        nonlocal displayed_mode

        now = perf_counter()
        camera_bounds = read_camera_bounds()
        if camera_bounds is None:
            return

        # Camera movement is detected by polling its actual scene-space rect.
        # No point selection or GPU upload occurs while it is changing.
        if last_observed_bounds is None or not np.allclose(
            camera_bounds, last_observed_bounds, rtol=1e-5, atol=1e-4
        ):
            last_observed_bounds = camera_bounds
            last_camera_change_time = now
            return

        if now - last_camera_change_time < VIEW_REFRESH_DEBOUNCE_SECONDS:
            return

        min_x, max_x, min_y, max_y = camera_bounds
        visible_width = max_x - min_x
        visible_height = max_y - min_y
        visible_span = max(visible_width, visible_height)

        # At overview scale, keep the original global sample. Filtering it to
        # the camera rectangle here caused the delayed rectangular blackout:
        # the complete overview was replaced by only points inside that box.
        if visible_span >= DETAIL_ZOOM_THRESHOLD:
            if displayed_mode != "overview":
                point_cloud.set_data(overview_positions, overview_colors)
                displayed_mode = "overview"
                last_uploaded_bounds = None
                print("\rOverview point cloud restored", end="", flush=True)
            return

        # Add a generous margin around the camera rect. It prevents the
        # selected point set from ending exactly at the visible window edge.
        pad_x = visible_width * VIEW_CULL_PADDING
        pad_y = visible_height * VIEW_CULL_PADDING
        selection_bounds = (
            min_x - pad_x,
            max_x + pad_x,
            min_y - pad_y,
            max_y + pad_y,
        )

        if last_uploaded_bounds is not None and np.allclose(
            selection_bounds, last_uploaded_bounds, rtol=1e-4, atol=1e-3
        ):
            return

        point_budget = dynamic_point_budget(
            visible_width,
            visible_height,
            len(positions),
        )
        indices = visible_point_indices(
            positions,
            chunk_bounds,
            selection_bounds,
            point_budget,
        )

        if indices.size:
            visible_positions = positions[indices]
            visible_colors = make_colors_for_indices(indices, digits)
            # Keep the old cloud until both replacement arrays are ready.
            # set_data then replaces the two GL buffers together in one update.
            point_cloud.set_data(visible_positions, visible_colors)
            displayed_mode = "detail"
            last_uploaded_bounds = selection_bounds
            upload_mib = (
                visible_positions.nbytes + visible_colors.nbytes
            ) / (1024.0 * 1024.0)
            print(
                f"\rDetail: {indices.size:,} points, "
                f"{upload_mib:.1f} MiB uploaded; "
                f"camera {visible_width:.1f} x {visible_height:.1f}",
                end="",
                flush=True,
            )

    # Poll at 20 Hz. The camera continues transforming the current GL cloud
    # immediately; expensive CPU selection and GPU uploads wait until 250 ms
    # after the last detected movement.
    update_timer = app.Timer(
        interval=VIEW_REFRESH_POLL_SECONDS,
        connect=lambda event: refresh_visible_points(),
        iterations=-1,
    )
    update_timer.start()

    print(f"Color mode: {COLOR_MODE}")
    if COLOR_MODE == "time":
        print(
            f"Spectrum repeats every {SPECTRUM_CYCLE_POINTS:,} input digits "
            f"({digits.size / SPECTRUM_CYCLE_POINTS:.2f} cycles for this file)"
        )
    if SHOW_ARROWS:
        print(
            f"Arrows: local circular mean over {ARROW_WINDOW_POINTS:,} digits; "
            f"maximum length {ARROW_MAX_LENGTH:g}, "
            f"minimum strength {ARROW_MIN_STRENGTH:g}"
        )
    print(
        f"Point budget: {INITIAL_GPU_POINTS:,} while viewport exceeds "
        f"{DETAIL_ZOOM_THRESHOLD:g} x {DETAIL_ZOOM_THRESHOLD:g} world units; "
        "then grows with zoom"
    )
    print(
        f"GPU upload cap: {MAX_GPU_POINTS:,} points "
        f"({MAX_GPU_POINTS * 24 / (1024 * 1024):.1f} MiB maximum per refresh)"
    )
    print(
        f"Viewport refresh: after {VIEW_REFRESH_DEBOUNCE_SECONDS:.2f} s idle; "
        f"selection margin: {VIEW_CULL_PADDING:.0%}"
    )
    print("Mouse wheel: zoom | Drag: pan | Close window: exit")
    print(f"Total preparation time: {perf_counter() - start_time:.2f} s")
    app.run()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()

