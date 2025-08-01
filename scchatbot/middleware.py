"""
Custom middleware for development settings.
"""

from django.utils.cache import add_never_cache_headers


class DisableBrowserCacheMiddleware:
    """
    Middleware to disable browser caching in development.
    This ensures that static files are always fetched fresh.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Only apply no-cache headers to static files in development
        if request.path.startswith('/static/'):
            add_never_cache_headers(response)
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
        
        return response