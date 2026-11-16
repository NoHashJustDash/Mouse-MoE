# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard memory limits for training so the desktop always keeps RAM to work with.

Three layers, all failing loudly rather than slowing the machine down:

1. Windows Job Object: the training process and every data-loader worker it
   starts belong to one job whose host-side committed memory is capped at the RAM
   that was available at launch minus the reserve (the capped GPU allocation,
   which Windows also charges to commit, is added on top). An allocation beyond it fails
   (``MemoryError`` in Python) instead of pushing the system into swap.
2. GPU allocator cap: PyTorch may use at most ``gpu_memory_fraction`` of device
   memory. Exceeding it raises CUDA out-of-memory instead of letting the Windows
   driver spill GPU memory into system RAM.
3. Watchdog: if free system RAM falls below the reserve for any reason (other
   programs included), ``check()`` raises so training can checkpoint and stop.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass

import torch

GIB = 1024**3


class MemoryReserveError(RuntimeError):
    """Free system RAM fell below the configured reserve."""


class _MemoryStatus(ctypes.Structure):
    _fields_ = (
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    )


class _BasicLimits(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )


class _ExtendedLimits(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def no_idle_sleep() -> Iterator[None]:
    """Keep Windows from idle-sleeping while the block runs; released when it exits.

    A per-thread SetThreadExecutionState request, as video players make: no power
    setting changes. ab_kd_teacher stopped when the PC idle-slept for 7 hours mid-run
    and free RAM dipped under the reserve on wake (2026-10-22). A closed lid or a
    manual Sleep still sleeps.
    """
    if sys.platform != "win32":
        raise RuntimeError("no_idle_sleep implements the Windows request only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def available_ram() -> int:
    """Free physical RAM in bytes (Windows GlobalMemoryStatusEx)."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


@dataclass(frozen=True)
class MemoryLimits:
    reserve_bytes: int
    host_budget_bytes: int
    gpu_limit_bytes: int


class MemoryGuard:
    """Owns the job handle and watchdog for the lifetime of a training process."""

    def __init__(
        self, reserve_gib: float, gpu_memory_fraction: float, device: torch.device
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("MemoryGuard implements the Windows job limit only")
        if reserve_gib <= 0 or not 0 < gpu_memory_fraction < 1:
            raise ValueError("reserve must be positive and gpu_memory_fraction in (0, 1)")
        self._reserve = int(reserve_gib * GIB)
        self._low = threading.Event()
        self._stop = threading.Event()
        self._observed_free = 0
        free = available_ram()
        host_budget = free - self._reserve
        if host_budget < 4 * GIB:
            raise MemoryReserveError(
                f"only {free / GIB:.1f} GiB RAM free; training needs the {reserve_gib:g} GiB "
                "reserve plus at least 4 GiB. Close other programs and retry."
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, index)
        total_gpu = torch.cuda.get_device_properties(index).total_memory
        gpu_limit = int(total_gpu * gpu_memory_fraction)
        # Windows (WDDM) charges device allocations to the owning process's commit
        # so they can be paged; they do not occupy RAM while resident on the GPU.
        # The job limit therefore adds the capped GPU allocation to the host budget,
        # which keeps host-side commit (RAM actually used) below free RAM - reserve.
        job_limit = host_budget + gpu_limit
        self._job = self._create_job(job_limit)
        self.limits = MemoryLimits(self._reserve, host_budget, gpu_limit)
        self._thread = threading.Thread(target=self._watch, name="memory-watchdog", daemon=True)
        self._thread.start()

    @staticmethod
    def _create_job(limit: int) -> int:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = limit
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(job),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        # Data-loader workers started afterwards are created inside this job.
        if not kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(job), wintypes.HANDLE(kernel32.GetCurrentProcess())
        ):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        return int(job)

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            free = available_ram()
            if free < self._reserve:
                self._observed_free = free
                self._low.set()

    def check(self) -> None:
        """Raise when free system RAM has dropped below the reserve."""
        if self._low.is_set():
            raise MemoryReserveError(
                f"free system RAM fell to {self._observed_free / GIB:.1f} GiB, below the "
                f"{self._reserve / GIB:g} GiB reserve"
            )

    def describe(self) -> str:
        return (
            f"memory limits: host {self.limits.host_budget_bytes / GIB:.1f} GiB, "
            f"system reserve {self.limits.reserve_bytes / GIB:g} GiB, "
            f"GPU allocator {self.limits.gpu_limit_bytes / GIB:.1f} GiB"
        )

    def close(self) -> None:
        self._stop.set()
