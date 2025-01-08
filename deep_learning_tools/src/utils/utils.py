from typing import Optional

class PrintManager:
    """Handles formatted console output with consistent styling."""
    
    @staticmethod
    def print_message(message: str, msg_type: Optional[str] = None, bold: bool = False) -> None:
        """
        Print messages with consistent formatting.

        Args:
            message (str): The message to print.
            msg_type (Optional[str]): Type of message ('success', 'progress', 'warning', 'info', 'error').
            bold (bool): Whether to print the message in bold.
        """
        color_codes = {
            # standard colors
            "green": "\033[38;5;40m",   # success messages
            "cyan": "\033[38;5;44m",    # progress messages
            "red": "\033[38;5;196m",    # error messages
            "orange": "\033[38;5;208m", # warning/caution messages
            
            # message type presets
            "success": "\033[38;5;40m",  # same as green
            "progress": "\033[38;5;44m", # same as cyan
            "warning": "\033[38;5;208m", # same as orange
            "error": "\033[38;5;196m",   # same as red
            "info": "\033[38;5;44m",     # same as cyan
            
            # formatting
            "bold": "\033[1m",
            "endc": "\033[0m"
        }

        formatted_message = ""
        
        if bold:
            formatted_message += color_codes["bold"]
        
        if msg_type and msg_type in color_codes:
            formatted_message += color_codes[msg_type]
        
        formatted_message += message + color_codes["endc"]
        
        print(formatted_message) 