#!/usr/bin/env python3
"""
Video to ASCII Art Converter with Audio Support
Standalone version - all modules combined into a single file.
"""

import argparse
import sys
import time
import os
import subprocess
import threading
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Iterator, Tuple, List

try:
    import cv2
    import numpy as np
except ImportError:
    print("\nFirst run setup: Installing required Python dependencies (opencv-python, numpy)...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "numpy"])
        import cv2
        import numpy as np
        print("Dependencies installed successfully!\n")
    except Exception as e:
        print(f"Error installing dependencies automatically: {e}")
        print("Please install them manually using: pip install opencv-python numpy")
        sys.exit(1)





# Terminal output settings
DEFAULT_WIDTH = 120
DEFAULT_HEIGHT = 40
DEFAULT_CHARSET = "standard"
ENABLE_COLOR = False
DEFAULT_SPEED = 1.0

# Video processing settings
VIDEO_QUALITY_SCALE = 0.7  # Scale factor for video processing (0.0-1.0)
FRAME_SKIP = 0  # Skip frames (0 = process all frames)

# ASCII conversion settings
ASPECT_RATIO_CORRECTION = 0.55  # Correction factor for character aspect ratio
TERMINAL_ASPECT_RATIO = 0.5  # Typical terminal character aspect ratio

# Color settings
USE_TRUECOLOR = True  # Use 24-bit RGB colors if available
MAX_COLORS = 256  # Maximum colors to use

# Logging
VERBOSE = False
SHOW_PROGRESS = True

# Performance
MAX_FPS = 60  # Maximum frames per second
MIN_FRAME_DELAY = 0.016  # Minimum delay between frames in seconds

# Output settings
SAVE_FRAMES = False  # Save individual frames to files
SAVE_DIRECTORY = "output_frames/"
EXPORT_FORMAT = "txt"  # 'txt', 'html', or 'both'



# Standard ASCII art charset - good balance of detail and compatibility
STANDARD_CHARSET = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# Simple charset - minimal characters, quick processing
SIMPLE_CHARSET = "@%#*+=-:. "

# Detailed charset - more characters for better quality
DETAILED_CHARSET = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.,~-+_=*%@$#"

# Block characters - uses Unicode block elements
BLOCK_CHARSET = "\u2588\u2593\u2592\u2591 "  # █ ▓ ▒ ░ (space)

# Minimal - just a few characters
MINIMAL_CHARSET = "@#*+=-:. "

# Binary - only two characters for extreme contrast
BINARY_CHARSET = "@. "

# All available charsets
CHARSETS = {
    "standard": STANDARD_CHARSET,
    "simple": SIMPLE_CHARSET,
    "detailed": DETAILED_CHARSET,
    "block": BLOCK_CHARSET,
    "minimal": MINIMAL_CHARSET,
    "binary": BINARY_CHARSET,
}

def get_charset(name: str = "standard") -> str:
    """Get a charset by name."""
    if name.lower() not in CHARSETS:
        available = ", ".join(CHARSETS.keys())
        raise ValueError(f"Unknown charset '{name}'. Available: {available}")
    return CHARSETS[name.lower()]

def list_charsets() -> list:
    """Get list of all available charset names."""
    return list(CHARSETS.keys())



class ASCIIConverter:
    """Converts images/frames to ASCII art."""
    
    def __init__(self, width: int = 120, charset: str = "standard"):
        self.width = width
        self.charset = get_charset(charset)
        self.charset_len = len(self.charset)
        
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        height = frame.shape[0]
        width = frame.shape[1]
        aspect_ratio = height / width
        
        new_height = int(self.width * aspect_ratio * ASPECT_RATIO_CORRECTION)
        resized = cv2.resize(frame, (self.width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def frame_to_ascii(self, frame: np.ndarray) -> str:
        resized = self.resize_frame(frame)
        if len(resized.shape) != 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        normalized = resized.astype(np.float32) / 255.0
        indices = (normalized * (self.charset_len - 1)).astype(np.int32)
        
        ascii_chars = np.array([self.charset[idx] for idx in indices.flat])
        ascii_chars = ascii_chars.reshape(resized.shape)
        
        lines = []
        for row in ascii_chars:
            lines.append(''.join(row))
        return '\n'.join(lines)
    
    def get_ascii_array(self, frame: np.ndarray) -> np.ndarray:
        resized = self.resize_frame(frame)
        normalized = resized.astype(np.float32) / 255.0
        indices = (normalized * (self.charset_len - 1)).astype(np.int32)
        ascii_chars = np.array([self.charset[idx] for idx in indices.flat])
        ascii_chars = ascii_chars.reshape(resized.shape)
        return ascii_chars
    
    def get_dimensions(self, frame: np.ndarray) -> tuple:
        resized = self.resize_frame(frame)
        return (resized.shape[1], resized.shape[0])



class ColorProcessor:
    """Handles conversion of pixel colors to terminal ANSI colors."""
    
    ANSI_COLORS = {
        'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
        'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
    }
    
    def __init__(self, use_256_colors: bool = True, use_truecolor: bool = False):
        self.use_256_colors = use_256_colors
        self.use_truecolor = use_truecolor
    
    @staticmethod
    def rgb_to_ansi_16(r: int, g: int, b: int) -> int:
        brightness = (r + g + b) // 3
        if brightness < 50: color = 30
        elif brightness > 200: color = 37
        elif r > g and r > b: color = 31
        elif g > r and g > b: color = 32
        elif b > r and b > g: color = 34
        elif r > b: color = 33
        elif g > b: color = 32
        else: color = 34
        return color
    
    @staticmethod
    def rgb_to_256_color(r: int, g: int, b: int) -> int:
        r = round(r / 255 * 5)
        g = round(g / 255 * 5)
        b = round(b / 255 * 5)
        return 16 + 36 * r + 6 * g + b
    
    @staticmethod
    def rgb_to_truecolor(r: int, g: int, b: int) -> str:
        return f"\033[38;2;{r};{g};{b}m"
    
    def colorize_char(self, char: str, r: int, g: int, b: int) -> str:
        if self.use_truecolor:
            color_code = self.rgb_to_truecolor(r, g, b)
            return f"{color_code}{char}\033[0m"
        elif self.use_256_colors:
            color_code = self.rgb_to_256_color(r, g, b)
            return f"\033[38;5;{color_code}m{char}\033[0m"
        else:
            color_code = self.rgb_to_ansi_16(r, g, b)
            return f"\033[{color_code}m{char}\033[0m"
    
    def colorize_frame(self, frame: np.ndarray, ascii_array: np.ndarray) -> str:
        if len(frame.shape) == 2:
            frame_bgr = np.stack([frame, frame, frame], axis=2)
        else:
            frame_bgr = frame
        
        try:
            frame_resized = cv2.resize(frame_bgr, 
                                       (ascii_array.shape[1], ascii_array.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        except:
            return self._fallback_colorize(ascii_array)
        
        if len(frame_resized.shape) == 3:
            frame_rgb = frame_resized[:, :, ::-1]
        else:
            frame_rgb = frame_resized
        
        lines = []
        for y in range(ascii_array.shape[0]):
            line = ""
            for x in range(ascii_array.shape[1]):
                char = ascii_array[y, x]
                if len(frame_rgb.shape) == 3:
                    r, g, b = frame_rgb[y, x]
                    r, g, b = int(r), int(g), int(b)
                else:
                    r = g = b = int(frame_rgb[y, x])
                line += self.colorize_char(char, r, g, b)
            lines.append(line)
        return "\n".join(lines)
    
    def _fallback_colorize(self, ascii_array: np.ndarray) -> str:
        lines = []
        for row in ascii_array:
            lines.append(''.join(row))
        return '\n'.join(lines)



class VideoProcessor:
    """Handles video file reading and frame processing."""
    
    def __init__(self, video_path: str):
        try:
            camera_index = int(video_path)
            self.video_path = camera_index
        except ValueError:
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
            self.video_path = video_path
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._is_camera = isinstance(self.video_path, int)
        
    @property
    def fps(self) -> float: return self._fps
    
    @property
    def frame_count(self) -> int: return self._frame_count
    
    @property
    def width(self) -> int: return self._width
    
    @property
    def height(self) -> int: return self._height
    
    @property
    def is_camera(self) -> bool: return self._is_camera
    
    def get_frame(self, frame_num: Optional[int] = None) -> Optional[np.ndarray]:
        if frame_num is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret: return None
        return frame
    
    def get_frames(self, skip: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            if skip > 0 and frame_num % (skip + 1) != 0:
                frame_num += 1
                continue
            yield frame_num, frame
            frame_num += 1
    
    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self) -> None:
        if self.cap: self.cap.release()
    
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def __del__(self): self.close()



class AudioProcessor:
    """Handles audio extraction and playback."""
    
    def __init__(self, video_path: str, duration: float = 0):
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        self.video_path = video_path
        self.audio_file_path = os.path.abspath(video_path)
        self.is_playing = False
        self.play_thread = None
        self.playback_process = None
        self.playback_start_time = None
        self.audio_duration = duration
        self.ffplay_path = self.get_ffplay_path()
        
    @staticmethod
    def get_ffplay_path() -> Optional[str]:
        if sys.platform != 'win32':
            return 'ffplay'
            
        appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        player_dir = os.path.join(appdata, 'ASCIIVideoPlayer')
        ffplay_path = os.path.join(player_dir, 'ffplay.exe')
        
        if os.path.exists(ffplay_path):
            return ffplay_path
            
        try:
            print("\nFirst run setup: Downloading tiny portable audio player to system folder (~15MB)...")
            os.makedirs(player_dir, exist_ok=True)
            url = "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffplay-4.4.1-win-64.zip"
            zip_path = os.path.join(player_dir, "ffplay.zip")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("ffplay.exe", player_dir)
            os.remove(zip_path)
            print("Download complete!\n")
            return ffplay_path
        except Exception as e:
            print(f"Warning: Failed to download audio player: {e}")
            return None
        

    
    def get_audio_duration(self) -> float:
        return self.audio_duration
    
    def play_async(self, delay: float = 0, volume: float = 1.0) -> None:
        if not self.audio_file_path or not os.path.exists(self.audio_file_path):
            print("Warning: No audio file to play")
            return
        
        self.stop()
        
        def playback_thread():
            try:
                self.is_playing = True
                self.playback_start_time = time.time()
                if delay > 0: time.sleep(delay)
                
                if sys.platform == 'win32':
                    if self.ffplay_path:
                        self.playback_process = subprocess.Popen(
                            [self.ffplay_path, '-nodisp', '-autoexit', '-loglevel', 'quiet', self.audio_file_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW
                        )
                        self.playback_process.wait()
                    else:
                        print("Error: Audio player missing")
                elif sys.platform == 'darwin':
                    self.playback_process = subprocess.Popen(
                        ['afplay', self.audio_file_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    self.playback_process.wait()
                else:
                    players = ['ffplay', 'aplay', 'paplay', 'mpg123']
                    for player in players:
                        try:
                            cmd = [player]
                            if player == 'ffplay':
                                cmd.extend(['-nodisp', '-autoexit', '-loglevel', 'quiet'])
                            cmd.append(self.audio_file_path)
                            self.playback_process = subprocess.Popen(
                                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                            )
                            self.playback_process.wait()
                            break
                        except FileNotFoundError:
                            continue
                self.is_playing = False
            except Exception as e:
                print(f"Error during audio playback: {e}")
                self.is_playing = False
        
        self.play_thread = threading.Thread(target=playback_thread, daemon=True)
        self.play_thread.start()
    
    def stop(self) -> None:
        self.is_playing = False
        if self.playback_process:
            try:
                self.playback_process.terminate()
                self.playback_process.wait(timeout=1)
            except:
                try: self.playback_process.kill()
                except: pass
            self.playback_process = None
    
    def get_current_playback_time(self) -> float:
        if not self.is_playing or self.playback_start_time is None: return 0
        return time.time() - self.playback_start_time
    
    def is_audio_available(self) -> bool:
        return self.audio_file_path is not None and os.path.exists(self.audio_file_path)
    
    def cleanup(self) -> None:
        self.stop()


class AudioSynchronizer:
    """Synchronizes audio playback with frame display."""
    
    def __init__(self, fps: float, audio_processor: Optional[AudioProcessor] = None):
        self.fps = fps
        self.audio_processor = audio_processor
        self.frame_duration = 1.0 / fps if fps > 0 else 0.033
        self.start_time = None
        self.frame_count = 0
        
    def start(self, play_audio: bool = True) -> None:
        self.start_time = time.time()
        self.frame_count = 0
        if play_audio and self.audio_processor:
            self.audio_processor.play_async()
    
    def wait_for_frame(self, speed_multiplier: float = 1.0, processing_time: float = 0) -> None:
        if self.start_time is None: self.start_time = time.time()
        expected_time = self.frame_count * self.frame_duration / speed_multiplier
        if self.audio_processor and self.audio_processor.is_playing:
            actual_time = self.audio_processor.get_current_playback_time()
        else:
            actual_time = time.time() - self.start_time
        sleep_time = expected_time - actual_time - processing_time
        if sleep_time > 0: time.sleep(sleep_time)
        self.frame_count += 1
    
    def get_audio_duration(self) -> float:
        if self.audio_processor: return self.audio_processor.get_audio_duration()
        return 0
    
    def cleanup(self) -> None:
        if self.audio_processor: self.audio_processor.cleanup()



class ASCIIVideoPlayer:
    """Main video player that converts and displays video as ASCII art with optional audio."""
    
    def __init__(self, video_path: str, width: int = DEFAULT_WIDTH,
                 charset: str = DEFAULT_CHARSET, color: bool = ENABLE_COLOR,
                 speed: float = DEFAULT_SPEED, skip: int = 0, audio: bool = False):
        self.video_path = video_path
        self.width = width
        self.charset = charset
        self.color = color
        self.speed = speed
        self.skip = skip
        self.audio_enabled = audio
        
        try:
            self.processor = VideoProcessor(video_path)
            self.converter = ASCIIConverter(width, charset)
            self.color_processor = ColorProcessor(use_truecolor=True) if color else None
            self.audio_processor = None
            if audio:
                duration = self.processor.frame_count / self.processor.fps if self.processor.fps > 0 else 0
                self.audio_processor = AudioProcessor(video_path, duration)
            self.synchronizer = AudioSynchronizer(
                self.processor.fps, 
                self.audio_processor
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    def print_frame(self, frame_ascii: str) -> None:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(frame_ascii)
    
    def play(self) -> None:
        # Adjust terminal window size on Windows to fit the ASCII art
        if os.name == 'nt':
            expected_height = int(self.width * (self.processor.height / self.processor.width) * ASPECT_RATIO_CORRECTION) + 10
            os.system(f'mode con: cols={self.width} lines={expected_height}')
            
        print(f"\n=== ASCII Video Player with Audio ===")
        print(f"Video: {self.video_path}")
        print(f"Dimensions: {self.processor.width}x{self.processor.height}")
        print(f"FPS: {self.processor.fps:.2f}")
        print(f"Total Frames: {self.processor.frame_count}")
        print(f"Output Width: {self.width} characters")
        print(f"Charset: {self.charset}")
        print(f"Color: {'Enabled' if self.color else 'Disabled'}")
        print(f"Audio: {'Enabled' if self.audio_enabled else 'Disabled'}")
        print(f"Speed: {self.speed}x")
        print(f"\nStarting playback... Press Ctrl+C to stop.\n")
        time.sleep(2)
        
        self.synchronizer.start(play_audio=self.audio_enabled)
        
        try:
            frame_num = 0
            frame_count = 0
            total_frames = self.processor.frame_count
            
            for frame_idx, frame in self.processor.get_frames(skip=self.skip):
                start_time = time.time()
                if self.color and self.color_processor:
                    ascii_array = self.converter.get_ascii_array(frame)
                    frame_ascii = self.color_processor.colorize_frame(frame, ascii_array)
                else:
                    frame_ascii = self.converter.frame_to_ascii(frame)
                
                self.print_frame(frame_ascii)
                
                audio_time = ""
                if self.audio_processor and self.audio_processor.is_playing:
                    audio_time = f" | Audio: {self.synchronizer.get_audio_duration():.1f}s"
                
                if SHOW_PROGRESS and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"\n[{'='*30}] {progress:.1f}% - Frame {frame_count}/{total_frames}{audio_time}", 
                          end='', flush=True)
                else:
                    print(f"\nFrame: {frame_count}{audio_time}", end='', flush=True)
                
                processing_time = time.time() - start_time
                self.synchronizer.wait_for_frame(
                    speed_multiplier=self.speed,
                    processing_time=processing_time
                )
                frame_count += 1
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"\n=== Playback Complete ===")
            print(f"Total frames played: {frame_count}")
            print(f"Average FPS: {frame_count / (self.processor.fps / self.speed) if self.processor.fps > 0 else 0:.2f}")
            print("\nPress any key to exit...")
            
        except KeyboardInterrupt:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n\nPlayback interrupted by user.")
            print(f"Frames played: {frame_count}")
        finally:
            self.processor.close()
            if self.audio_processor: self.audio_processor.cleanup()
    
    def export_frames(self, output_dir: str = "output_frames/", include_color: bool = False) -> None:
        Path(output_dir).mkdir(exist_ok=True)
        print(f"Exporting frames to {output_dir}...")
        for frame_num, frame in self.processor.get_frames(skip=self.skip):
            if include_color and self.color_processor:
                ascii_array = self.converter.get_ascii_array(frame)
                frame_ascii = self.color_processor.colorize_frame(frame, ascii_array)
            else:
                frame_ascii = self.converter.frame_to_ascii(frame)
            filename = f"{output_dir}frame_{frame_num:06d}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(frame_ascii)
            if SHOW_PROGRESS and frame_num % 10 == 0:
                print(f"Exported {frame_num} frames...", end='\r')
        print(f"\nExport complete! {frame_num + 1} frames saved.")
        self.processor.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to ASCII art with optional audio synchronization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f video.mp4
  %(prog)s -f video.mp4 -w 100 --audio --color
  %(prog)s -f video.mp4 -w 80 -c simple --speed 2.0 --audio
  %(prog)s -f 0 -w 80  # Webcam input
        """
    )
    
    parser.add_argument('-f', '--file', dest='video_file', required=True,
                        help='Path to video file (or camera index like "0")')
    parser.add_argument('-w', '--width', dest='width', type=int, default=DEFAULT_WIDTH,
                        help=f'Output width in characters (default: {DEFAULT_WIDTH})')
    parser.add_argument('-c', '--charset', dest='charset', default=DEFAULT_CHARSET,
                        choices=list_charsets(),
                        help=f'Character set to use (default: {DEFAULT_CHARSET})')
    parser.add_argument('--audio', dest='audio', action='store_true',
                        help='Enable audio playback (standalone, no dependencies needed)')
    parser.add_argument('--color', dest='color', action='store_true',
                        help='Enable color output')
    parser.add_argument('--speed', dest='speed', type=float, default=DEFAULT_SPEED,
                        help=f'Playback speed multiplier (default: {DEFAULT_SPEED})')
    parser.add_argument('--skip', dest='skip', type=int, default=0,
                        help='Skip frames (0 = process all frames)')
    parser.add_argument('--export', dest='export', type=str, default=None,
                        help='Export frames to directory instead of playing')
    parser.add_argument('--export-color', dest='export_color', action='store_true',
                        help='Include ANSI color codes when exporting frames')
    
    args = parser.parse_args()
    
    if args.width < 20:
        print("Error: Width must be at least 20 characters", file=sys.stderr)
        sys.exit(1)
    if args.width > 300:
        print("Warning: Very large widths may be slow", file=sys.stderr)
    if args.speed <= 0:
        print("Error: Speed must be positive", file=sys.stderr)
        sys.exit(1)
    
    player = ASCIIVideoPlayer(
        video_path=args.video_file,
        width=args.width,
        charset=args.charset,
        color=args.color,
        speed=args.speed,
        skip=args.skip,
        audio=args.audio
    )
    
    if args.export:
        player.export_frames(args.export, include_color=args.export_color)
    else:
        player.play()

if __name__ == '__main__':
    main()
