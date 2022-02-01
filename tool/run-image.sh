#!/usr/bin/env sh

set -e

docker run -v $PWD:/data -u "$(id -u):$(id -g)" shopty "$@"
