"""
Jinja2 template filters
"""
import datetime
from typing import Any

def register_filters(app):
    """Register custom filters with Flask app"""
    
    @app.template_filter('timestamp_to_date')
    def timestamp_to_date(timestamp):
        """Convert Unix timestamp to formatted date string"""
        if not timestamp:
            return "Unknown"
        try:
            return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return "Invalid date"
            
    @app.template_filter('format_output')
    def format_output(value: Any) -> str:
        """Format output values, especially for lists
        
        Args:
            value: The value to format
            
        Returns:
            str: Formatted string representation of the value
        """
        if isinstance(value, list):
            # Format each list item on a new line
            return '<br>'.join([str(item) for item in value])
        return str(value)
        
    @app.template_filter('format_for_edit')
    def format_for_edit(value: Any) -> str:
        """Format output values for editing in a textarea
        
        Args:
            value: The value to format
            
        Returns:
            str: Formatted string representation suitable for textarea editing
        """
        if isinstance(value, list):
            # Format each list item on a new line
            return '\n'.join([str(item) for item in value])
        return str(value)