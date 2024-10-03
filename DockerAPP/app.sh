docker pull ghcr.io/tabatsky/app:latest
docker run -it -v "$(pwd)/app":"/app" ghcr.io/tabatsky/app:latest /bin/bash
