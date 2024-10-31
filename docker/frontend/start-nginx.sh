#!/bin/sh
set -e
trap 'kill -TERM $PID' TERM INT
# Define configuration file location
CONFIG_FILE="/etc/nginx/conf.d/default.conf"

# Check if placeholders are present in default.conf
if grep -q "__BACKEND_URL__\|__FRONTEND_PORT__" "$CONFIG_FILE"; then
  echo "Detected placeholders in default.conf. Running replacement script."

  # Fallbacks for BACKEND_URL and FRONTEND_PORT if not set
  BACKEND_URL="${BACKEND_URL:-${1:-}}"
  FRONTEND_PORT="${FRONTEND_PORT:-${2:-80}}"

  if [ -z "$BACKEND_URL" ]; then
    echo "Error: BACKEND_URL must be set as an environment variable or as the first parameter. (e.g., http://localhost:7860)"
    exit 1
  fi

  # Substitute placeholders in default.conf
  sed -i "s|__BACKEND_URL__|$BACKEND_URL|g" "$CONFIG_FILE"
  sed -i "s|__FRONTEND_PORT__|$FRONTEND_PORT|g" "$CONFIG_FILE"

  echo "Updated default.conf with BACKEND_URL: $BACKEND_URL and FRONTEND_PORT: $FRONTEND_PORT"
else
  echo "No placeholders detected in default.conf. Skipping replacement."
fi


# Start nginx
exec nginx -g 'daemon off;'
