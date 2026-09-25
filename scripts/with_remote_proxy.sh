#!/usr/bin/env bash
# Apply the verified proxy only to this process and its children.
set -euo pipefail
AIC_DOWNLOAD_PROXY=${AIC_DOWNLOAD_PROXY:-http://127.0.0.1:7890}
export http_proxy="$AIC_DOWNLOAD_PROXY" https_proxy="$AIC_DOWNLOAD_PROXY"
export HTTP_PROXY="$AIC_DOWNLOAD_PROXY" HTTPS_PROXY="$AIC_DOWNLOAD_PROXY"
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
exec "$@"
