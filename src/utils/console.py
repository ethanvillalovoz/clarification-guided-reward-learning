"""
Console output formatting utilities for research-grade terminal output.

This module provides utilities for creating structured, visually appealing terminal output
suitable for research presentations, demos, and debugging. Features include:

- Color-coded output based on message type
- Progress bars with customizable appearance
- Section headers with dividers
- Hierarchical indentation for nested information
- Timestamp formatting
- Multiple log levels with filtering
"""

import sys
import os
import time
import math
import shutil
from datetime import datetime
from enum import Enum
import textwrap
import re

class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    CRITICAL = 4

class ConsoleFormatter:
    """
    Utility class to format console output in a clean, research-grade manner.
    """
    # ANSI color codes
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Default verbosity level (can be changed at runtime)
    VERBOSITY = LogLevel.INFO
    
    @classmethod
    def set_verbosity(cls, level):
        """Set the global verbosity level."""
        if isinstance(level, str):
            level = level.upper()
            if hasattr(LogLevel, level):
                cls.VERBOSITY = getattr(LogLevel, level)
        elif isinstance(level, int):
            for lvl in LogLevel:
                if lvl.value == level:
                    cls.VERBOSITY = lvl
                    break
        elif isinstance(level, LogLevel):
            cls.VERBOSITY = level
            
    @classmethod
    def progress_bar(cls, iteration, total, prefix='', suffix='', decimals=1, length=50, 
                    fill='█', color='blue', print_end='\r'):
        """
        Call in a loop to create terminal progress bar
        
        Parameters
        ----------
        iteration  - Required  : current iteration (Int)
        total      - Required  : total iterations (Int)
        prefix     - Optional  : prefix string (Str)
        suffix     - Optional  : suffix string (Str)
        decimals   - Optional  : number of decimals in percent (Int)
        length     - Optional  : character length of bar (Int)
        fill       - Optional  : bar fill character (Str)
        color      - Optional  : color of the progress bar (Str)
        print_end  - Optional  : end character (e.g. "\\r", "\\n") (Str)
        """
        color_code = getattr(cls, color.upper(), cls.BLUE) if isinstance(color, str) else cls.BLUE
        
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = color_code + fill * filled_length + cls.END + '-' * (length - filled_length)
        
        # Get terminal width for dynamic formatting
        term_width = shutil.get_terminal_size().columns
        
        # Make sure our output fits in the terminal
        max_prefix_len = min(len(prefix), 30)  # Limit prefix length
        max_suffix_len = min(len(suffix), 30)  # Limit suffix length
        max_bar_len = term_width - max_prefix_len - max_suffix_len - 10  # 10 for percent and spacing
        
        if length > max_bar_len and max_bar_len > 10:
            length = max_bar_len
            filled_length = int(length * iteration // total)
            bar = color_code + fill * filled_length + cls.END + '-' * (length - filled_length)
        
        formatted_output = f"{prefix} |{bar}| {percent}% {suffix}"
        
        print('\r' + formatted_output, end=print_end)
        sys.stdout.flush()
        
        # Print New Line on Complete
        if iteration == total:
            print()
            
    @classmethod
    def elapsed_time(cls, start_time):
        """Format elapsed time in a human-readable format"""
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{elapsed:.2f} seconds"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @classmethod
    def should_log(cls, level):
        """Check if the given log level should be displayed."""
        if isinstance(level, str):
            level = getattr(LogLevel, level.upper(), LogLevel.INFO)
        return level.value >= cls.VERBOSITY.value
    
    @classmethod
    def section(cls, title, color="purple", bold=True, top_line=True, bottom_line=True):
        """
        Print a section header with customizable formatting.
        
        Parameters:
        -----------
        title : str
            The title text to display
        color : str
            Color name (purple, blue, green, yellow, red, cyan)
        bold : bool
            Whether to bold the text
        top_line : bool
            Whether to show a line above the title
        bottom_line : bool
            Whether to show a line below the title
        """
        width = os.get_terminal_size().columns - 4
        color_code = getattr(cls, color.upper(), cls.PURPLE) if isinstance(color, str) else cls.PURPLE
        
        style_start = f"{cls.BOLD if bold else ''}{color_code}"
        style_end = cls.END
        
        if top_line:
            print(f"\n{style_start}{'═' * width}{style_end}")
        else:
            print()
            
        print(f"{style_start}  {title.upper()}{style_end}")
        
        if bottom_line:
            print(f"{style_start}{'═' * width}{style_end}\n")
        else:
            print()
    
    @classmethod
    def subsection(cls, title, color="blue", bold=False, line=True):
        """
        Print a subsection header with customizable formatting.
        
        Parameters:
        -----------
        title : str
            The title text to display
        color : str
            Color name (purple, blue, green, yellow, red, cyan)
        bold : bool
            Whether to bold the text
        line : bool
            Whether to show a line below the title
        """
        width = os.get_terminal_size().columns - 4
        color_code = getattr(cls, color.upper(), cls.BLUE) if isinstance(color, str) else cls.BLUE
        
        style_start = f"{cls.BOLD if bold else ''}{color_code}"
        style_end = cls.END
        
        print(f"\n{style_start}  {title}{style_end}")
        
        if line:
            print(f"{color_code}{'-' * width}{style_end}\n")
    
    @classmethod
    def _format_message(cls, message, color=None, bold=False, indent=0):
        """Format message with color and indentation"""
        prefix = "  " * indent
        color_code = ""
        if color:
            color_code = getattr(cls, color.upper(), "") if isinstance(color, str) else ""
        
        if bold:
            color_code = f"{cls.BOLD}{color_code}"
            
        return f"{prefix}{color_code}{message}{cls.END}" if color_code else f"{prefix}{message}"
    
    @classmethod
    def info(cls, message, indent=0, color=None, bold=False):
        """
        Print an info message.
        
        Parameters:
        -----------
        message : str
            Message to display
        indent : int
            Indentation level (2 spaces per level)
        color : str
            Optional color for the message text
        bold : bool
            Whether to bold the message text
        """
        if cls.should_log(LogLevel.INFO):
            prefix = "  " * indent
            icon = cls._format_message("▶", "green", False)
            text = cls._format_message(message, color, bold)
            print(f"{prefix}{icon} {text}")
    
    @classmethod
    def debug(cls, message, indent=0, color=None, bold=False):
        """Print a debug message with optional formatting."""
        if cls.should_log(LogLevel.DEBUG):
            prefix = "  " * indent
            icon = cls._format_message("↳", "cyan", False)
            text = cls._format_message(message, color, bold)
            print(f"{prefix}{icon} {text}")
    
    @classmethod
    def warn(cls, message, indent=0, color=None, bold=False):
        """Print a warning message with optional formatting."""
        if cls.should_log(LogLevel.WARN):
            prefix = "  " * indent
            icon = cls._format_message("⚠", "yellow", False)
            text = cls._format_message(message, color or "yellow", bold)
            print(f"{prefix}{icon} {text}")
    
    @classmethod
    def error(cls, message, indent=0, color=None, bold=False):
        """Print an error message with optional formatting."""
        if cls.should_log(LogLevel.ERROR):
            prefix = "  " * indent
            icon = cls._format_message("✖", "red", False)
            text = cls._format_message(message, color or "red", bold)
            print(f"{prefix}{icon} {text}")
    
    @classmethod
    def critical(cls, message, indent=0, color=None, bold=True):
        """Print a critical error message with optional formatting."""
        if cls.should_log(LogLevel.CRITICAL):
            prefix = "  " * indent
            icon = cls._format_message("☢", "red", True)
            text = cls._format_message(message, color or "red", bold)
            print(f"{prefix}{icon} {text}")
    
    @classmethod
    def success(cls, message, indent=0, color=None, bold=True):
        """Print a success message with optional formatting."""
        prefix = "  " * indent
        icon = cls._format_message("✓", "green", True)
        text = cls._format_message(message, color or "green", bold)
        print(f"{prefix}{icon} {text}")
    
    @classmethod
    def result(cls, label, value, indent=0, label_color="blue", value_color=None, bold_label=True):
        """Print a labeled result with customizable formatting."""
        prefix = "  " * indent
        formatted_label = cls._format_message(f"{label}:", label_color, bold_label)
        formatted_value = cls._format_message(value, value_color, False)
        print(f"{prefix}{formatted_label} {formatted_value}")
    
    @classmethod
    def progress(cls, current, total, message="Processing", width=30, color="blue"):
        """
        Print a progress bar (simplified version).
        For more advanced progress bars, use progress_bar method.
        
        Parameters:
        -----------
        current : int
            Current progress value
        total : int
            Total target value
        message : str
            Message to display before the progress bar
        width : int
            Width of the progress bar in characters
        color : str
            Color of the progress bar
        """
        cls.progress_bar(current, total, prefix=message, length=width, color=color)
        sys.stdout.flush()
    
    @classmethod
    def timestamp(cls):
        """Print current timestamp."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{cls.CYAN}[{now}]{cls.END}")
    
    @classmethod
    def table(cls, headers, rows, indent=0, color=None, header_color="blue", header_bold=True):
        """
        Print a formatted table with customizable colors.
        
        Parameters:
        -----------
        headers : list
            List of column headers
        rows : list of lists
            List of rows, where each row is a list of cells
        indent : int
            Indentation level
        color : str
            Default color for the table data
        header_color : str
            Color for the header row
        header_bold : bool
            Whether to bold the header row
        """
        if not rows:
            return
        
        # Calculate column widths
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Get color codes
        header_style = getattr(cls, header_color.upper(), "") if header_color else ""
        if header_bold:
            header_style = f"{cls.BOLD}{header_style}"
        
        data_style = getattr(cls, color.upper(), "") if color else ""
        
        # Print headers
        prefix = "  " * indent
        header_row = " │ ".join(f"{str(h):{w}s}" for h, w in zip(headers, widths))
        print(f"{prefix}{header_style}{header_row}{cls.END}")
        
        # Print separator
        total_width = sum(widths) + (3 * (len(widths) - 1))
        separator = "─" * total_width
        print(f"{prefix}{header_style}{separator}{cls.END}")
        
        # Print rows
        for row in rows:
            row_str = " │ ".join(f"{str(cell):{w}s}" for cell, w in zip(row, widths))
            print(f"{prefix}{data_style}{row_str}{cls.END if data_style else ''}")
            
        print()
        
    @classmethod
    def beliefs_table(cls, beliefs, labels, indent=0, threshold=0.01, title=None):
        """
        Display beliefs in a nicely formatted table.
        
        Parameters:
        -----------
        beliefs : list or array
            List of belief probabilities
        labels : list
            List of model names corresponding to beliefs
        indent : int
            Indentation level
        threshold : float
            Only show beliefs above this threshold
        title : str
            Optional title for the table
        """
        if title:
            cls.info(title, indent=indent, color="magenta", bold=True)
            
        # Create a list of (label, belief) pairs and sort by belief
        belief_pairs = [(label, float(belief)) for label, belief in zip(labels, beliefs)]
        belief_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and prepare rows
        rows = []
        for label, belief in belief_pairs:
            if belief >= threshold:
                rows.append([label, f"{belief:.4f}", f"{belief*100:.1f}%"])
                
        if not rows:
            cls.info("(No beliefs above threshold)", indent=indent+1, color="gray")
            return
            
        # Display the table
        headers = ["Model", "Probability", "Percent"]
        cls.table(headers, rows, indent=indent+1, color="cyan", header_color="magenta")
    
    @classmethod
    def clear_line(cls):
        """Clear the current line in the terminal."""
        term_width = os.get_terminal_size().columns
        sys.stdout.write("\r" + " " * term_width + "\r")
        sys.stdout.flush()
    
    @classmethod
    def wrap_text(cls, text, indent=0, width=None):
        """Wrap text to fit the terminal width."""
        if width is None:
            width = os.get_terminal_size().columns - (indent * 2) - 2
        prefix = "  " * indent
        wrapped = textwrap.wrap(text, width=width)
        return "\n".join(f"{prefix}{line}" for line in wrapped)

# For convenience, create global aliases
log = ConsoleFormatter
