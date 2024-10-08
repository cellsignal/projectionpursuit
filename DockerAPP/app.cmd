docker pull ghcr.io/tabatsky/app:latest
mkdir app
docker run -it -v ".\app":"/app" ghcr.io/tabatsky/app:latest /bin/bash
