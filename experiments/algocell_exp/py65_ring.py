"""Independent py65 NMOS oracle with 16-bit addresses mapped into a byte ring.

The primary run uses the installed, unmodified py65 MPU. DiagnosticMPU is used
only to explain differences; its narrow shims are recorded and never counted as
raw matches. No Zilion decode table or ALU implementation is used by this oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from py65.devices.mpu6502 import MPU

JAM = frozenset((0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x92, 0xB2, 0xD2, 0xF2))
IMPLEMENTED = frozenset(i for i, handler in enumerate(MPU.instruct) if handler != MPU.inst_not_implemented)
FIELDS = ("a", "x", "y", "s", "p", "pc")


class RingMemory:
    """Address width is applied BEFORE ring mapping, including py65's addr+1.

    Thus 0xffff+1 maps to position zero, rather than 65536 % ring_size. This
    matters for odd rings even though the production P=32/64 seam offset is zero.
    """

    def __init__(self, tape):
        self.data = bytearray(tape)
        if not self.data:
            raise ValueError("empty ring")

    def __getitem__(self, address: int) -> int:
        return self.data[(address & 0xFFFF) % len(self.data)]

    def __setitem__(self, address: int, value: int) -> None:
        self.data[(address & 0xFFFF) % len(self.data)] = value & 0xFF


def stack_pointer(ring_size: int, first_push_at: int) -> int:
    """Descriptor convention: largest S for which (0x100 | S) % P = t."""
    for s in range(255, -1, -1):
        if (0x100 | s) % ring_size == first_push_at:
            return s
    raise ValueError("no page-one stack alias")


def state(cpu: MPU) -> dict[str, int]:
    return dict(zip(FIELDS, (cpu.a, cpu.x, cpu.y, cpu.sp, cpu.p, cpu.pc)))


class DiagnosticMPU(MPU):
    """Two documented oracle differences, for attribution only.

    * B is a pushed-status marker, not a persistent NMOS flag (BRK/PLP/RTI).
    * JSR reads its low operand before its stack writes, and its high after.
    All other instructions, addressing and ALU operations remain py65's.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[dict] = []
        self.step_index = 0

    def event(self, kind: str, **evidence):
        self.events.append({"kind": kind, "step": self.step_index, **evidence})

    def step(self):
        opcode = self.memory[self.pc]
        if opcode == 0x20:
            pc = self.pc
            low = self.memory[pc + 1]
            self.stPushWord((pc + 2) & 0xFFFF)
            reread = self.memory[pc + 1]
            high = self.memory[pc + 2]
            self.pc = low | (high << 8)
            if low != reread:
                self.event("jsr-low-read-order", pc=pc, low_before_push=low, low_after_push=reread, high=high)
            return self
        super().step()
        if opcode in (0x00, 0x28, 0x40) and self.p & self.BREAK:
            self.event("status-b-bit", opcode=opcode, py65_p=self.p, nmos_p=self.p & ~self.BREAK)
            self.p &= ~self.BREAK
        return self

@dataclass
class OracleResult:
    memory: bytes
    state: dict[str, int]
    steps: int
    halted: bool
    unsupported: dict | None = None
    events: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


def run_oracle(tape, budget: int = 128, *, diagnostic: bool = False, trace: bool = False,
               initial_state: dict | None = None) -> OracleResult:
    memory = RingMemory(tape)
    cls = DiagnosticMPU if diagnostic else MPU
    cpu = cls(memory=memory, pc=0)
    cpu.a = cpu.x = cpu.y = 0
    cpu.p = 0x20
    cpu.sp = stack_pointer(len(memory.data), len(memory.data) - 1)
    if initial_state:
        for key, value in initial_state.items():
            setattr(cpu, "sp" if key == "s" else key, value)
    history = []
    halted, unsupported, steps = False, None, 0
    for index in range(1, budget + 1):
        opcode = memory[cpu.pc]
        entry = {"step": index, "opcode": opcode, "before": state(cpu)} if trace else None
        if opcode in JAM:
            steps, halted = index, True
        elif opcode not in IMPLEMENTED:
            unsupported = {"step": index, "pc": cpu.pc, "opcode": opcode}
            # Never execute py65's silent two-byte unsupported-opcode stub.
            break
        else:
            if diagnostic:
                cpu.step_index = index
            cpu.step()
            steps = index
        if trace:
            entry.update(after=state(cpu), memory=bytes(memory.data).hex())
            history.append(entry)
        if halted:
            break
    return OracleResult(bytes(memory.data), state(cpu), steps, halted, unsupported,
                        cpu.events if diagnostic else [], history)
