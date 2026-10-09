#!/usr/bin/env bash
exec bwrap --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/bin /sbin \
    --symlink usr/lib /lib --symlink usr/lib /lib64 --ro-bind /etc /etc \
    --proc /proc --dev /dev --tmpfs /tmp \
	--bind "$(dirname "$0")/workspace" /workspace \
    --chdir /workspace --setenv HOME /workspace \
    --unshare-all --die-with-parent --new-session bash -c "$1"
