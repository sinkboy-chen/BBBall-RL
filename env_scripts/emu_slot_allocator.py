#!/usr/bin/env python3
import atexit
import getpass
import os
import time


def _default_lock_dir():
    user = getpass.getuser()
    return os.environ.get("BBBALL_LOCK_DIR", f"/tmp2/{user}/DRL_final_workspace/emu_locks")


def acquire_slot(slot_count=3, lock_dir=None, wait_s=1.0, timeout_s=300):
    lock_dir = lock_dir or _default_lock_dir()
    os.makedirs(lock_dir, exist_ok=True)

    start_time = time.time()
    while True:
        for slot in range(slot_count):
            lock_path = os.path.join(lock_dir, f"slot_{slot}.lock")
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                os.close(fd)
                atexit.register(release_slot, lock_path)
                return slot, lock_path
            except FileExistsError:
                continue

        if time.time() - start_time > timeout_s:
            raise RuntimeError("No emulator slots available on this host")
        time.sleep(wait_s)


def release_slot(lock_path):
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass
