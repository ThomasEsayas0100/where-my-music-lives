"""
Sophisticated ASCII art spinning globe for terminal loading screen.
Pure terminal output with 3D sphere mathematics and continental outlines.
"""

import math
import time
import threading
import os
import sys
from typing import Optional, Tuple, List


# Simplified continent outlines as (lat, lon) coordinate lists
# Lat: -90 (south) to 90 (north), Lon: -180 (west) to 180 (east)
CONTINENTS = [
    # North America
    [(60, -165), (70, -140), (70, -100), (55, -80), (45, -75), (30, -80),
     (25, -100), (20, -105), (15, -110), (10, -90), (15, -80), (20, -75),
     (25, -80), (30, -85), (35, -90), (40, -95), (45, -100), (50, -120),
     (55, -130), (60, -140), (65, -150), (60, -165)],

    # South America
    [(10, -80), (0, -75), (-10, -70), (-20, -65), (-30, -70), (-40, -70),
     (-50, -70), (-55, -68), (-50, -65), (-40, -60), (-30, -50), (-20, -45),
     (-10, -50), (0, -60), (5, -70), (10, -80)],

    # Europe
    [(70, -10), (70, 20), (60, 30), (50, 40), (45, 35), (40, 20), (45, 10),
     (50, 0), (55, -5), (60, -10), (70, -10)],

    # Africa
    [(35, 10), (30, 30), (20, 40), (10, 45), (0, 40), (-10, 35), (-20, 30),
     (-30, 25), (-35, 20), (-30, 15), (-20, 15), (-10, 20), (0, 20), (10, 15),
     (20, 10), (30, 5), (35, 10)],

    # Asia
    [(70, 60), (70, 100), (70, 140), (60, 150), (50, 140), (45, 130), (40, 120),
     (35, 110), (30, 100), (25, 90), (20, 85), (15, 90), (10, 95), (10, 100),
     (20, 105), (30, 110), (35, 115), (40, 125), (45, 135), (50, 145), (55, 150),
     (50, 155), (45, 160), (50, 170), (45, 180), (50, 180), (55, 170), (60, 160),
     (65, 150), (70, 140), (70, 120), (65, 100), (60, 80), (55, 70), (60, 65),
     (70, 60)],

    # Australia
    [(-10, 115), (-15, 125), (-20, 135), (-30, 140), (-35, 145), (-40, 145),
     (-35, 150), (-30, 150), (-25, 145), (-20, 140), (-15, 135), (-10, 130),
     (-15, 120), (-10, 115)],

    # Antarctica (simplified ring)
    [(-70, -180), (-70, -120), (-70, -60), (-70, 0), (-70, 60), (-70, 120),
     (-70, 180), (-75, 180), (-75, -180), (-70, -180)],
]


# Character set for shading (light to dark)
SHADE_CHARS = ' .:-=+*#%@'


def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Check if a point (lat, lon) is inside a polygon using ray casting."""
    n = len(polygon)
    inside = False

    p1_lat, p1_lon = polygon[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon[i % n]

        # Handle longitude wrapping
        if abs(p2_lon - p1_lon) > 180:
            if p2_lon > p1_lon:
                p2_lon -= 360
            else:
                p2_lon += 360

        if lon > min(p1_lon, p2_lon):
            if lon <= max(p1_lon, p2_lon):
                if lat <= max(p1_lat, p2_lat):
                    if p1_lon != p2_lon:
                        x_inters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                    if p1_lat == p2_lat or lat <= x_inters:
                        inside = not inside

        p1_lat, p1_lon = p2_lat, p2_lon

    return inside


def is_land(lat: float, lon: float) -> bool:
    """Check if coordinates fall on land (within any continent polygon)."""
    # Normalize longitude to -180 to 180
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360

    for continent in CONTINENTS:
        if point_in_polygon(lat, lon, continent):
            return True

    return False


def render_globe(rotation: float, width: int = 46, height: int = 23) -> List[str]:
    """
    Render a 3D globe with continental outlines at given rotation angle.

    Args:
        rotation: Y-axis rotation angle in radians
        width: Character width of output
        height: Character height of output

    Returns:
        List of strings representing each line of the rendered globe
    """
    # Sphere radius
    radius = min(width, height * 2) / 2.2

    # Center of the output grid
    center_x = width / 2
    center_y = height / 2

    # Light direction (from top-right-front)
    light_dir = (0.5, 0.7, 1.0)
    light_mag = math.sqrt(sum(x*x for x in light_dir))
    light_dir = tuple(x / light_mag for x in light_dir)

    # Initialize output grid
    output = [[' ' for _ in range(width)] for _ in range(height)]

    for row in range(height):
        for col in range(width):
            # Map character position to sphere coordinates
            # Account for character aspect ratio (~2:1)
            x = (col - center_x) * 0.5
            y = (row - center_y)

            # Check if ray hits sphere
            z_squared = radius * radius - x * x - y * y

            if z_squared >= 0:
                # Point is on sphere
                z = math.sqrt(z_squared)

                # Calculate surface normal
                normal_mag = math.sqrt(x*x + y*y + z*z)
                normal = (x / normal_mag, y / normal_mag, z / normal_mag)

                # Lighting (Lambertian shading)
                light_intensity = max(0, sum(n * l for n, l in zip(normal, light_dir)))

                # Convert 3D point to lat/lon with rotation
                # Apply Y-axis rotation
                x_rot = x * math.cos(rotation) - z * math.sin(rotation)
                z_rot = x * math.sin(rotation) + z * math.cos(rotation)

                # Calculate latitude and longitude
                # Negate y so north is at top of screen
                lat = math.degrees(math.asin(-y / radius))
                lon = math.degrees(math.atan2(x_rot, z_rot))

                # Check if land or ocean
                on_land = is_land(lat, lon)

                # Determine character based on land/ocean and lighting
                if on_land:
                    # Land: use darker/denser characters
                    shade_index = 5 + int(light_intensity * 4)
                    shade_index = max(5, min(len(SHADE_CHARS) - 1, shade_index))
                else:
                    # Ocean: use lighter characters
                    shade_index = 1 + int(light_intensity * 2)
                    shade_index = max(1, min(3, shade_index))

                # Edge falloff for spherical appearance
                edge_factor = z / radius
                if edge_factor < 0.3:
                    shade_index = max(0, shade_index - 2)

                output[row][col] = SHADE_CHARS[shade_index]

    return [''.join(row) for row in output]


class LoadingController:
    """Controller for managing the loading screen animation."""

    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.status_message = "Initializing..."
        self.rotation = 0.0
        self._lock = threading.Lock()

    def update_status(self, message: str):
        """Update the status message displayed below the globe."""
        with self._lock:
            self.status_message = message

    def stop(self):
        """Stop the animation and clear the screen."""
        self.running = False
        if self.thread:
            self.thread.join()

        # Clear screen
        print("\033[2J\033[H", end='', flush=True)

    def _animate(self):
        """Main animation loop (runs in background thread)."""
        # Hide cursor
        print("\033[?25l", end='', flush=True)

        try:
            while self.running:
                # Get terminal size
                try:
                    terminal_size = os.get_terminal_size()
                    term_width = terminal_size.columns
                    term_height = terminal_size.lines
                except OSError:
                    term_width = 80
                    term_height = 40

                # Globe dimensions
                globe_width = 46
                globe_height = 23

                # Render globe
                globe_lines = render_globe(self.rotation, globe_width, globe_height)

                # Calculate vertical centering
                total_content_height = globe_height + 6  # Title + globe + spacing + status
                top_padding = max(0, (term_height - total_content_height) // 2)

                # Build output
                output = []

                # Add top padding
                output.extend([''] * top_padding)

                # Add title
                title = "W H E R E   Y O U R   M U S I C   L I V E S"
                title_pad = max(0, (term_width - len(title)) // 2)
                output.append(' ' * title_pad + title)
                output.append('')

                # Add centered globe
                for line in globe_lines:
                    padding = max(0, (term_width - len(line)) // 2)
                    output.append(' ' * padding + line)

                # Add spacing
                output.append('')

                # Add status message with spinner
                with self._lock:
                    status = self.status_message

                spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                spinner = spinner_chars[int(self.rotation * 10) % len(spinner_chars)]
                status_line = f"{spinner} {status}"

                padding = max(0, (term_width - len(status_line)) // 2)
                output.append(' ' * padding + status_line)

                # Move cursor to home and print
                print("\033[H", end='')
                print('\n'.join(output), end='', flush=True)

                # Update rotation
                self.rotation += 0.08
                if self.rotation >= 2 * math.pi:
                    self.rotation -= 2 * math.pi

                # Frame delay for ~30 FPS
                time.sleep(0.033)

        finally:
            # Show cursor
            print("\033[?25h", end='', flush=True)

    def start(self):
        """Start the animation in a background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()


def show_loading() -> LoadingController:
    """
    Start the loading screen animation.

    Returns:
        LoadingController instance for updating status and stopping animation

    Example:
        loader = show_loading()
        loader.update_status("Fetching scrobbles...")
        time.sleep(2)
        loader.update_status("Resolving Spotify tracks...")
        time.sleep(2)
        loader.stop()
    """
    controller = LoadingController()

    # Clear screen
    print("\033[2J\033[H", end='', flush=True)

    controller.start()
    return controller


def main():
    """Demo mode showing various loading states."""
    print("ASCII Globe Loading Screen Demo")
    print("Press Ctrl+C to exit\n")
    time.sleep(2)

    loader = show_loading()

    try:
        # Cycle through demo status messages
        messages = [
            "Fetching scrobbles...",
            "Resolving Spotify tracks...",
            "Building audio profile...",
            "Analyzing audio features...",
            "Matching cities...",
            "Computing recommendations...",
            "Finalizing results...",
        ]

        for msg in messages:
            loader.update_status(msg)
            time.sleep(3)

        # Loop back to start
        while True:
            for msg in messages:
                loader.update_status(msg)
                time.sleep(3)

    except KeyboardInterrupt:
        loader.stop()
        print("\nDemo stopped.")


if __name__ == "__main__":
    main()
