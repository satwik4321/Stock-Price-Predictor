# settings.py
INSTALLED_APPS = [
    # Other apps
    "corsheaders",
    "channels",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    # Other middleware...
]

# Allow all origins or specify your frontend's domain
CORS_ALLOW_ALL_ORIGINS = True  # To allow all origins
# OR
CORS_ALLOWED_ORIGINS = [
    "http://your-frontend-domain.com",  # Replace with your actual frontend URL
]
# settings.py
# Configure CORS for WebSocket connections if necessary
CORS_ALLOW_ALL_ORIGINS = True
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}