#!/usr/bin/env bash
# DubbingStudio Stop Script

echo "Stopping DubbingStudio..."

# Stop Celery
if [ -f /tmp/dubbing-celery.pid ]; then
    kill -9 $(cat /tmp/dubbing-celery.pid) 2>/dev/null && echo "Celery stopped"
    rm -f /tmp/dubbing-celery.pid
else
    pkill -f "celery.*dubbing_studio" 2>/dev/null && echo "Celery stopped" || true
fi

# Stop Redis
redis-cli shutdown 2>/dev/null && echo "Redis stopped" || echo "Redis was not running"

echo "Done."
