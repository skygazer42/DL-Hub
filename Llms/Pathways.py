from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PathwaysConfig:
    interleave_quantum: int = 1


@dataclass(frozen=True)
class VirtualDevice:
    logical_id: str
    island: str
    physical_device: str


@dataclass(frozen=True)
class PathwaysProgram:
    name: str
    stages: tuple[str, ...]
    required_devices: int = 1
    compiled_functions: tuple[str, ...] = ()


@dataclass(frozen=True)
class GangScheduleEntry:
    program: PathwaysProgram
    devices: tuple[VirtualDevice, ...]
    timeslice: int
    islands: tuple[str, ...]


@dataclass(frozen=True)
class InterleavedStep:
    timeslice: int
    program_name: str
    stage: str


class PathwaysTracer:
    def __init__(self) -> None:
        self._compiled: list[str] = []

    def add_compiled(self, name: str) -> None:
        self._compiled.append(str(name))

    def fuse(self, name: str, *, required_devices: int = 1) -> PathwaysProgram:
        if not self._compiled:
            raise ValueError("at least one compiled function is required to build a program")
        compiled = tuple(self._compiled)
        return PathwaysProgram(
            name=str(name),
            stages=compiled,
            required_devices=int(required_devices),
            compiled_functions=compiled,
        )


class PathwaysRuntime:
    def __init__(self, config: PathwaysConfig, virtual_devices: list[VirtualDevice] | tuple[VirtualDevice, ...]) -> None:
        if int(config.interleave_quantum) <= 0:
            raise ValueError("interleave_quantum must be > 0")
        devices = sorted(list(virtual_devices), key=lambda device: device.logical_id)
        if not devices:
            raise ValueError("virtual_devices cannot be empty")
        self.config = config
        self.virtual_devices = tuple(devices)

    def _spread_virtual_devices(self, required_devices: int) -> tuple[VirtualDevice, ...]:
        if int(required_devices) <= 0:
            raise ValueError("required_devices must be > 0")
        if int(required_devices) > len(self.virtual_devices):
            raise ValueError("required_devices cannot exceed the number of virtual devices")

        by_island: dict[str, deque[VirtualDevice]] = {}
        for device in self.virtual_devices:
            by_island.setdefault(device.island, deque()).append(device)

        selected: list[VirtualDevice] = []
        active_islands = sorted(by_island)
        while active_islands and len(selected) < int(required_devices):
            remaining: list[str] = []
            for island in active_islands:
                queue = by_island[island]
                if queue and len(selected) < int(required_devices):
                    selected.append(queue.popleft())
                if queue:
                    remaining.append(island)
            active_islands = remaining
        return tuple(sorted(selected, key=lambda device: device.logical_id))

    def map_virtual_devices(self, program: PathwaysProgram) -> tuple[VirtualDevice, ...]:
        return self._spread_virtual_devices(int(program.required_devices))

    def gang_schedule(
        self,
        programs: list[PathwaysProgram] | tuple[PathwaysProgram, ...],
    ) -> tuple[GangScheduleEntry, ...]:
        schedule: list[GangScheduleEntry] = []
        for timeslice, program in enumerate(programs):
            devices = self.map_virtual_devices(program)
            schedule.append(
                GangScheduleEntry(
                    program=program,
                    devices=devices,
                    timeslice=timeslice,
                    islands=tuple(sorted({device.island for device in devices})),
                )
            )
        return tuple(schedule)

    def interleave(
        self,
        programs: list[PathwaysProgram] | tuple[PathwaysProgram, ...],
    ) -> tuple[InterleavedStep, ...]:
        queues = [deque(program.stages) for program in programs]
        quantum = int(self.config.interleave_quantum)
        timeline: list[InterleavedStep] = []
        timeslice = 0

        while any(queue for queue in queues):
            for program, queue in zip(programs, queues):
                for _ in range(quantum):
                    if not queue:
                        break
                    timeline.append(
                        InterleavedStep(
                            timeslice=timeslice,
                            program_name=program.name,
                            stage=queue.popleft(),
                        )
                    )
                    timeslice += 1
        return tuple(timeline)


__all__ = [
    "GangScheduleEntry",
    "InterleavedStep",
    "PathwaysConfig",
    "PathwaysProgram",
    "PathwaysRuntime",
    "PathwaysTracer",
    "VirtualDevice",
]
