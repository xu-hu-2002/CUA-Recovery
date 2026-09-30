#!/usr/bin/env bash
# Put the guest into a quiescent state before an offline persona seed.
set -Eeuo pipefail

echo 'GUEST_SEEDER_STOPPING_APPS'
sudo -n timeout --signal=TERM --kill-after=30s 300 \
  systemctl stop mypcbench-apps.target || true
echo 'GUEST_SEEDER_APPS_STOPPED'

# Avoid matching this shell or the pkill command itself.
pkill -TERM -u user -f '[f]irefox-real/firefox' || true
sleep 3
pkill -KILL -u user -f '[f]irefox-real/firefox' || true

find /data /home/user -type f \( -name '*.sqlite-wal' -o -name '*.sqlite-shm' \) -delete

if pgrep -u user -f '[f]irefox-real/firefox' >/dev/null 2>&1; then
  echo 'guest seeder guard: Firefox is still running' >&2
  exit 1
fi
if sudo -n systemctl is-active --quiet mypcbench-apps.target; then
  echo 'guest seeder guard: mypcbench-apps.target is still active' >&2
  exit 1
fi

echo 'GUEST_SEEDER_QUIESCENT'
