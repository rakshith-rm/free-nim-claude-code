"""Entry point - run the proxy server."""

import uvicorn

from app import app, settings


def main():
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
