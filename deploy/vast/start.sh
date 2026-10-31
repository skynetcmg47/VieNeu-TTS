#!/bin/sh
set -eu
if [ "$#" -eq 0 ]; then
    set -- run
fi
exec /opt/model/bin/python /opt/vieneu/deploy/vast/runtime.py "$@"
