"""
Jinja2 template filters
"""
import datetime

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