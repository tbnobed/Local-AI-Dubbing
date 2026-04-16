#!/usr/bin/env bash
# DubbingStudio Stop Script

echo "Stopping DubbingStudio..."

# Stop Celery supervisor first so it doesn't respawn the worker
if [ -f /tmp/dubbing-celery-supervisor.pid ]; then
    SUP_PID=$(cat /tmp/dubbing-celery-supervisor.pid 2>/dev/null || true)
    if [ -n "$SUP_PID" ]; then
        kill -9 "$SUP_PID" 2>/dev/null && echo "Celery supervisor stopped" || true
    fi
    rm -f /tmp/dubbing-celery-supervisor.pid
fi

# Stop Celery worker(s)
if [ -f /tmp/dubbing-celery.pid ]; then
    kill -9 $(cat /tmp/dubbing-celery.pid) 2>/dev/null && echo "Celery stopped"
    rm -f /tmp/dubbing-celery.pid
fi
pkill -9 -f "celery.*-A app.core.celery_app.*worker" 2>/dev/null && echo "Celery worker(s) killed" || true

# Stop Redis
redis-cli shutdown 2>/dev/null && echo "Redis stopped" || echo "Redis was not running"

echo "Done."
