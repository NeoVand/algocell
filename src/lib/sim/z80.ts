// Compact Z80 CPU emulator for the primordial soup experiment
// Only needs 32-byte wrapped address space, no I/O ports
// Covers all unprefixed, CB-prefixed, and key ED-prefixed opcodes

export class Z80 {
	// Main registers
	a = 0;
	f = 0;
	b = 0;
	c = 0;
	d = 0;
	e = 0;
	h = 0;
	l = 0;

	// Shadow registers
	a2 = 0;
	f2 = 0;
	b2 = 0;
	c2 = 0;
	d2 = 0;
	e2 = 0;
	h2 = 0;
	l2 = 0;

	// Special registers
	sp = 0;
	pc = 0;
	ix = 0;
	iy = 0;
	i = 0;
	r = 0;
	iff1 = false;
	iff2 = false;

	halted = false;

	// Memory access (set externally)
	readByte: (addr: number) => number = () => 0;
	writeByte: (addr: number, val: number) => void = () => {};

	// Flag bit positions
	static readonly CF = 0x01;
	static readonly NF = 0x02;
	static readonly PF = 0x04;
	static readonly F3 = 0x08;
	static readonly HF = 0x10;
	static readonly F5 = 0x20;
	static readonly ZF = 0x40;
	static readonly SF = 0x80;

	reset(): void {
		this.a = this.f = this.b = this.c = this.d = this.e = this.h = this.l = 0;
		this.a2 = this.f2 = this.b2 = this.c2 = this.d2 = this.e2 = this.h2 = this.l2 = 0;
		this.sp = 0;
		this.pc = 0;
		this.ix = this.iy = 0;
		this.i = this.r = 0;
		this.iff1 = this.iff2 = false;
		this.halted = false;
	}

	// Register pair accessors
	get bc(): number {
		return (this.b << 8) | this.c;
	}
	set bc(v: number) {
		this.b = (v >> 8) & 0xff;
		this.c = v & 0xff;
	}
	get de(): number {
		return (this.d << 8) | this.e;
	}
	set de(v: number) {
		this.d = (v >> 8) & 0xff;
		this.e = v & 0xff;
	}
	get hl(): number {
		return (this.h << 8) | this.l;
	}
	set hl(v: number) {
		this.h = (v >> 8) & 0xff;
		this.l = v & 0xff;
	}
	get af(): number {
		return (this.a << 8) | this.f;
	}
	set af(v: number) {
		this.a = (v >> 8) & 0xff;
		this.f = v & 0xff;
	}

	private fetch(): number {
		const v = this.readByte(this.pc);
		this.pc = (this.pc + 1) & 0xffff;
		return v;
	}

	private fetchWord(): number {
		const lo = this.fetch();
		const hi = this.fetch();
		return (hi << 8) | lo;
	}

	private push16(val: number): void {
		this.sp = (this.sp - 1) & 0xffff;
		this.writeByte(this.sp, (val >> 8) & 0xff);
		this.sp = (this.sp - 1) & 0xffff;
		this.writeByte(this.sp, val & 0xff);
	}

	private pop16(): number {
		const lo = this.readByte(this.sp);
		this.sp = (this.sp + 1) & 0xffff;
		const hi = this.readByte(this.sp);
		this.sp = (this.sp + 1) & 0xffff;
		return (hi << 8) | lo;
	}

	// Get/set 8-bit register by index (0=B,1=C,2=D,3=E,4=H,5=L,6=(HL),7=A)
	private getReg(idx: number): number {
		switch (idx) {
			case 0:
				return this.b;
			case 1:
				return this.c;
			case 2:
				return this.d;
			case 3:
				return this.e;
			case 4:
				return this.h;
			case 5:
				return this.l;
			case 6:
				return this.readByte(this.hl);
			case 7:
				return this.a;
			default:
				return 0;
		}
	}

	private setReg(idx: number, val: number): void {
		val &= 0xff;
		switch (idx) {
			case 0:
				this.b = val;
				break;
			case 1:
				this.c = val;
				break;
			case 2:
				this.d = val;
				break;
			case 3:
				this.e = val;
				break;
			case 4:
				this.h = val;
				break;
			case 5:
				this.l = val;
				break;
			case 6:
				this.writeByte(this.hl, val);
				break;
			case 7:
				this.a = val;
				break;
		}
	}

	// Get/set 16-bit register pair by index (0=BC,1=DE,2=HL,3=SP)
	private getReg16(idx: number): number {
		switch (idx) {
			case 0:
				return this.bc;
			case 1:
				return this.de;
			case 2:
				return this.hl;
			case 3:
				return this.sp;
			default:
				return 0;
		}
	}

	private setReg16(idx: number, val: number): void {
		val &= 0xffff;
		switch (idx) {
			case 0:
				this.bc = val;
				break;
			case 1:
				this.de = val;
				break;
			case 2:
				this.hl = val;
				break;
			case 3:
				this.sp = val;
				break;
		}
	}

	// Get/set 16-bit register pair for PUSH/POP (0=BC,1=DE,2=HL,3=AF)
	private getReg16AF(idx: number): number {
		if (idx === 3) return this.af;
		return this.getReg16(idx);
	}

	private setReg16AF(idx: number, val: number): void {
		if (idx === 3) {
			this.af = val;
		} else {
			this.setReg16(idx, val);
		}
	}

	// Flag helpers
	private szFlags(val: number): number {
		return (val & Z80.SF) | (val === 0 ? Z80.ZF : 0) | (val & (Z80.F3 | Z80.F5));
	}

	private parity(val: number): boolean {
		let p = val;
		p ^= p >> 4;
		p ^= p >> 2;
		p ^= p >> 1;
		return (p & 1) === 0;
	}

	// ALU operations
	private aluOp(op: number, val: number): void {
		let result: number;
		const a = this.a;
		const c = this.f & Z80.CF;

		switch (op) {
			case 0: // ADD
				result = a + val;
				this.f =
					this.szFlags(result & 0xff) |
					(result > 0xff ? Z80.CF : 0) |
					((a ^ val ^ result) & Z80.HF) |
					((~(a ^ val) & (a ^ result) & 0x80) !== 0 ? Z80.PF : 0);
				this.a = result & 0xff;
				break;
			case 1: // ADC
				result = a + val + c;
				this.f =
					this.szFlags(result & 0xff) |
					(result > 0xff ? Z80.CF : 0) |
					((a ^ val ^ result) & Z80.HF) |
					((~(a ^ val) & (a ^ result) & 0x80) !== 0 ? Z80.PF : 0);
				this.a = result & 0xff;
				break;
			case 2: // SUB
				result = a - val;
				this.f =
					this.szFlags(result & 0xff) |
					Z80.NF |
					(result < 0 ? Z80.CF : 0) |
					((a ^ val ^ result) & Z80.HF) |
					(((a ^ val) & (a ^ result) & 0x80) !== 0 ? Z80.PF : 0);
				this.a = result & 0xff;
				break;
			case 3: // SBC
				result = a - val - c;
				this.f =
					this.szFlags(result & 0xff) |
					Z80.NF |
					(result < 0 ? Z80.CF : 0) |
					((a ^ val ^ result) & Z80.HF) |
					(((a ^ val) & (a ^ result) & 0x80) !== 0 ? Z80.PF : 0);
				this.a = result & 0xff;
				break;
			case 4: // AND
				this.a = a & val;
				this.f = this.szFlags(this.a) | Z80.HF | (this.parity(this.a) ? Z80.PF : 0);
				break;
			case 5: // XOR
				this.a = a ^ val;
				this.f = this.szFlags(this.a) | (this.parity(this.a) ? Z80.PF : 0);
				break;
			case 6: // OR
				this.a = a | val;
				this.f = this.szFlags(this.a) | (this.parity(this.a) ? Z80.PF : 0);
				break;
			case 7: // CP
				result = a - val;
				this.f =
					(result & 0xff & Z80.SF) |
					((result & 0xff) === 0 ? Z80.ZF : 0) |
					(val & (Z80.F3 | Z80.F5)) |
					Z80.NF |
					(result < 0 ? Z80.CF : 0) |
					((a ^ val ^ result) & Z80.HF) |
					(((a ^ val) & (a ^ result) & 0x80) !== 0 ? Z80.PF : 0);
				break;
		}
	}

	// Check condition code
	private checkCondition(cc: number): boolean {
		switch (cc) {
			case 0:
				return (this.f & Z80.ZF) === 0; // NZ
			case 1:
				return (this.f & Z80.ZF) !== 0; // Z
			case 2:
				return (this.f & Z80.CF) === 0; // NC
			case 3:
				return (this.f & Z80.CF) !== 0; // C
			case 4:
				return (this.f & Z80.PF) === 0; // PO
			case 5:
				return (this.f & Z80.PF) !== 0; // PE
			case 6:
				return (this.f & Z80.SF) === 0; // P
			case 7:
				return (this.f & Z80.SF) !== 0; // M
			default:
				return false;
		}
	}

	// INC 8-bit
	private inc8(val: number): number {
		const result = (val + 1) & 0xff;
		this.f =
			(this.f & Z80.CF) |
			this.szFlags(result) |
			(val === 0x7f ? Z80.PF : 0) |
			((result & 0x0f) === 0 ? Z80.HF : 0);
		return result;
	}

	// DEC 8-bit
	private dec8(val: number): number {
		const result = (val - 1) & 0xff;
		this.f =
			(this.f & Z80.CF) |
			this.szFlags(result) |
			Z80.NF |
			(val === 0x80 ? Z80.PF : 0) |
			((val & 0x0f) === 0 ? Z80.HF : 0);
		return result;
	}

	// ADD HL, rr
	private addHL(val: number): void {
		const hl = this.hl;
		const result = hl + val;
		this.f =
			(this.f & (Z80.SF | Z80.ZF | Z80.PF)) |
			(result > 0xffff ? Z80.CF : 0) |
			((hl ^ val ^ result) & 0x1000 ? Z80.HF : 0) |
			((result >> 8) & (Z80.F3 | Z80.F5));
		this.hl = result & 0xffff;
	}

	step(): void {
		if (this.halted) return;
		const op = this.fetch();
		this.executeMain(op);
	}

	private executeMain(op: number): void {
		const x = (op >> 6) & 3;
		const y = (op >> 3) & 7;
		const z = op & 7;
		const p = (y >> 1) & 3;
		const q = y & 1;

		switch (x) {
			case 0:
				this.execX0(y, z, p, q);
				break;
			case 1:
				if (y === 6 && z === 6) {
					this.halted = true; // HALT
				} else {
					this.setReg(y, this.getReg(z)); // LD r, r'
				}
				break;
			case 2:
				this.aluOp(y, this.getReg(z)); // ALU A, r
				break;
			case 3:
				this.execX3(y, z, p, q);
				break;
		}
	}

	private execX0(y: number, z: number, p: number, q: number): void {
		switch (z) {
			case 0:
				switch (y) {
					case 0:
						break; // NOP
					case 1: {
						// EX AF, AF'
						let tmp = this.a;
						this.a = this.a2;
						this.a2 = tmp;
						tmp = this.f;
						this.f = this.f2;
						this.f2 = tmp;
						break;
					}
					case 2: {
						// DJNZ
						const d = this.signedByte(this.fetch());
						this.b = (this.b - 1) & 0xff;
						if (this.b !== 0) this.pc = (this.pc + d) & 0xffff;
						break;
					}
					case 3: {
						// JR
						const d = this.signedByte(this.fetch());
						this.pc = (this.pc + d) & 0xffff;
						break;
					}
					default: {
						// JR cc (y-4)
						const d = this.signedByte(this.fetch());
						if (this.checkCondition(y - 4)) this.pc = (this.pc + d) & 0xffff;
						break;
					}
				}
				break;
			case 1:
				if (q === 0) {
					this.setReg16(p, this.fetchWord()); // LD rr, nn
				} else {
					this.addHL(this.getReg16(p)); // ADD HL, rr
				}
				break;
			case 2:
				if (q === 0) {
					switch (p) {
						case 0:
							this.writeByte(this.bc, this.a);
							break; // LD (BC), A
						case 1:
							this.writeByte(this.de, this.a);
							break; // LD (DE), A
						case 2: {
							const nn = this.fetchWord();
							this.writeByte(nn, this.l);
							this.writeByte((nn + 1) & 0xffff, this.h);
							break;
						} // LD (nn), HL
						case 3:
							this.writeByte(this.fetchWord(), this.a);
							break; // LD (nn), A
					}
				} else {
					switch (p) {
						case 0:
							this.a = this.readByte(this.bc);
							break; // LD A, (BC)
						case 1:
							this.a = this.readByte(this.de);
							break; // LD A, (DE)
						case 2: {
							const nn = this.fetchWord();
							this.l = this.readByte(nn);
							this.h = this.readByte((nn + 1) & 0xffff);
							break;
						} // LD HL, (nn)
						case 3:
							this.a = this.readByte(this.fetchWord());
							break; // LD A, (nn)
					}
				}
				break;
			case 3:
				if (q === 0) {
					this.setReg16(p, (this.getReg16(p) + 1) & 0xffff); // INC rr
				} else {
					this.setReg16(p, (this.getReg16(p) - 1) & 0xffff); // DEC rr
				}
				break;
			case 4:
				this.setReg(y, this.inc8(this.getReg(y))); // INC r
				break;
			case 5:
				this.setReg(y, this.dec8(this.getReg(y))); // DEC r
				break;
			case 6:
				this.setReg(y, this.fetch()); // LD r, n
				break;
			case 7:
				this.execRotAccum(y);
				break;
		}
	}

	private execRotAccum(y: number): void {
		const a = this.a;
		const c = this.f & Z80.CF;
		switch (y) {
			case 0: // RLCA
				this.a = ((a << 1) | (a >> 7)) & 0xff;
				this.f = (this.f & (Z80.SF | Z80.ZF | Z80.PF)) | (a >> 7) | (this.a & (Z80.F3 | Z80.F5));
				break;
			case 1: // RRCA
				this.a = ((a >> 1) | (a << 7)) & 0xff;
				this.f = (this.f & (Z80.SF | Z80.ZF | Z80.PF)) | (a & 1) | (this.a & (Z80.F3 | Z80.F5));
				break;
			case 2: // RLA
				this.a = ((a << 1) | c) & 0xff;
				this.f =
					(this.f & (Z80.SF | Z80.ZF | Z80.PF)) | (a >> 7) | (this.a & (Z80.F3 | Z80.F5));
				break;
			case 3: // RRA
				this.a = ((a >> 1) | (c << 7)) & 0xff;
				this.f = (this.f & (Z80.SF | Z80.ZF | Z80.PF)) | (a & 1) | (this.a & (Z80.F3 | Z80.F5));
				break;
			case 4: {
				// DAA
				let correction = 0;
				let carry = c;
				if ((this.f & Z80.HF) !== 0 || (a & 0x0f) > 9) correction |= 0x06;
				if (c !== 0 || a > 0x99) {
					correction |= 0x60;
					carry = 1;
				}
				if ((this.f & Z80.NF) !== 0) {
					this.a = (a - correction) & 0xff;
				} else {
					this.a = (a + correction) & 0xff;
				}
				this.f =
					(this.f & Z80.NF) |
					this.szFlags(this.a) |
					carry |
					((a ^ this.a) & Z80.HF) |
					(this.parity(this.a) ? Z80.PF : 0);
				break;
			}
			case 5: // CPL
				this.a = ~a & 0xff;
				this.f = (this.f & (Z80.SF | Z80.ZF | Z80.PF | Z80.CF)) | Z80.HF | Z80.NF | (this.a & (Z80.F3 | Z80.F5));
				break;
			case 6: // SCF
				this.f = (this.f & (Z80.SF | Z80.ZF | Z80.PF)) | Z80.CF | (this.a & (Z80.F3 | Z80.F5));
				break;
			case 7: // CCF
				this.f =
					(this.f & (Z80.SF | Z80.ZF | Z80.PF)) |
					((this.f & Z80.CF) !== 0 ? Z80.HF : 0) |
					(c !== 0 ? 0 : Z80.CF) |
					(this.a & (Z80.F3 | Z80.F5));
				break;
		}
	}

	private execX3(y: number, z: number, p: number, q: number): void {
		switch (z) {
			case 0: // RET cc
				if (this.checkCondition(y)) this.pc = this.pop16();
				break;
			case 1:
				if (q === 0) {
					this.setReg16AF(p, this.pop16()); // POP rr
				} else {
					switch (p) {
						case 0:
							this.pc = this.pop16();
							break; // RET
						case 1: {
							// EXX
							let tmp = this.b;
							this.b = this.b2;
							this.b2 = tmp;
							tmp = this.c;
							this.c = this.c2;
							this.c2 = tmp;
							tmp = this.d;
							this.d = this.d2;
							this.d2 = tmp;
							tmp = this.e;
							this.e = this.e2;
							this.e2 = tmp;
							tmp = this.h;
							this.h = this.h2;
							this.h2 = tmp;
							tmp = this.l;
							this.l = this.l2;
							this.l2 = tmp;
							break;
						}
						case 2:
							this.pc = this.hl;
							break; // JP (HL)
						case 3:
							this.sp = this.hl;
							break; // LD SP, HL
					}
				}
				break;
			case 2: {
				// JP cc, nn
				const nn = this.fetchWord();
				if (this.checkCondition(y)) this.pc = nn;
				break;
			}
			case 3:
				switch (y) {
					case 0:
						this.pc = this.fetchWord();
						break; // JP nn
					case 1:
						this.executeCB();
						break; // CB prefix
					case 2:
						this.fetch();
						break; // OUT (n), A - consume byte, NOP
					case 3:
						this.a = this.fetch() & 0xff;
						break; // IN A, (n) - read port byte as immediate, store in A (simplified)
					case 4: {
						// EX (SP), HL
						const lo = this.readByte(this.sp);
						const hi = this.readByte((this.sp + 1) & 0xffff);
						this.writeByte(this.sp, this.l);
						this.writeByte((this.sp + 1) & 0xffff, this.h);
						this.h = hi;
						this.l = lo;
						break;
					}
					case 5: {
						// EX DE, HL
						const tmp = this.de;
						this.de = this.hl;
						this.hl = tmp;
						break;
					}
					case 6:
						this.iff1 = false;
						this.iff2 = false;
						break; // DI
					case 7:
						this.iff1 = true;
						this.iff2 = true;
						break; // EI
				}
				break;
			case 4: {
				// CALL cc, nn
				const nn = this.fetchWord();
				if (this.checkCondition(y)) {
					this.push16(this.pc);
					this.pc = nn;
				}
				break;
			}
			case 5:
				if (q === 0) {
					this.push16(this.getReg16AF(p)); // PUSH rr
				} else {
					switch (p) {
						case 0: {
							// CALL nn
							const nn = this.fetchWord();
							this.push16(this.pc);
							this.pc = nn;
							break;
						}
						case 1:
							this.executeDD();
							break; // DD prefix (IX)
						case 2:
							this.executeED();
							break; // ED prefix
						case 3:
							this.executeFD();
							break; // FD prefix (IY)
					}
				}
				break;
			case 6:
				this.aluOp(y, this.fetch()); // ALU A, n
				break;
			case 7:
				this.push16(this.pc); // RST
				this.pc = y * 8;
				break;
		}
	}

	// CB prefix: bit operations, rotates, shifts
	private executeCB(): void {
		const op = this.fetch();
		const x = (op >> 6) & 3;
		const y = (op >> 3) & 7;
		const z = op & 7;
		const val = this.getReg(z);

		switch (x) {
			case 0: {
				// Rotation/shift group
				const result = this.cbRotShift(y, val);
				this.setReg(z, result);
				break;
			}
			case 1:
				// BIT y, r
				this.f =
					(this.f & Z80.CF) |
					Z80.HF |
					((val & (1 << y)) === 0 ? Z80.ZF | Z80.PF : 0) |
					(y === 7 && (val & 0x80) !== 0 ? Z80.SF : 0) |
					(val & (Z80.F3 | Z80.F5));
				break;
			case 2:
				this.setReg(z, val & ~(1 << y)); // RES y, r
				break;
			case 3:
				this.setReg(z, val | (1 << y)); // SET y, r
				break;
		}
	}

	private cbRotShift(op: number, val: number): number {
		let result: number;
		const c = this.f & Z80.CF;

		switch (op) {
			case 0: // RLC
				result = ((val << 1) | (val >> 7)) & 0xff;
				this.f = this.szFlags(result) | (val >> 7) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 1: // RRC
				result = ((val >> 1) | (val << 7)) & 0xff;
				this.f =
					this.szFlags(result) | (val & 1) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 2: // RL
				result = ((val << 1) | c) & 0xff;
				this.f = this.szFlags(result) | (val >> 7) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 3: // RR
				result = ((val >> 1) | (c << 7)) & 0xff;
				this.f =
					this.szFlags(result) | (val & 1) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 4: // SLA
				result = (val << 1) & 0xff;
				this.f = this.szFlags(result) | (val >> 7) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 5: // SRA
				result = ((val >> 1) | (val & 0x80)) & 0xff;
				this.f =
					this.szFlags(result) | (val & 1) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 6: // SLL (undocumented, sets bit 0)
				result = ((val << 1) | 1) & 0xff;
				this.f = this.szFlags(result) | (val >> 7) | (this.parity(result) ? Z80.PF : 0);
				return result;
			case 7: // SRL
				result = (val >> 1) & 0xff;
				this.f =
					this.szFlags(result) | (val & 1) | (this.parity(result) ? Z80.PF : 0);
				return result;
			default:
				return val;
		}
	}

	// ED prefix: block transfers and misc
	private executeED(): void {
		const op = this.fetch();

		switch (op) {
			// Block transfer operations
			case 0xa0:
				this.ldi();
				break;
			case 0xa8:
				this.ldd();
				break;
			case 0xb0:
				this.ldir();
				break;
			case 0xb8:
				this.lddr();
				break;

			// Block compare
			case 0xa1:
				this.cpi();
				break;
			case 0xa9:
				this.cpd();
				break;
			case 0xb1:
				this.cpir();
				break;
			case 0xb9:
				this.cpdr();
				break;

			// NEG
			case 0x44:
			case 0x4c:
			case 0x54:
			case 0x5c:
			case 0x64:
			case 0x6c:
			case 0x74:
			case 0x7c: {
				const a = this.a;
				this.a = 0;
				this.aluOp(2, a); // SUB a
				break;
			}

			// RETN/RETI
			case 0x45:
			case 0x4d:
			case 0x55:
			case 0x5d:
			case 0x65:
			case 0x6d:
			case 0x75:
			case 0x7d:
				this.iff1 = this.iff2;
				this.pc = this.pop16();
				break;

			// LD I,A / LD R,A / LD A,I / LD A,R
			case 0x47:
				this.i = this.a;
				break;
			case 0x4f:
				this.r = this.a;
				break;
			case 0x57:
				this.a = this.i;
				this.f =
					(this.f & Z80.CF) |
					this.szFlags(this.a) |
					(this.iff2 ? Z80.PF : 0);
				break;
			case 0x5f:
				this.a = this.r;
				this.f =
					(this.f & Z80.CF) |
					this.szFlags(this.a) |
					(this.iff2 ? Z80.PF : 0);
				break;

			// 16-bit load from/to memory
			case 0x43:
			case 0x53:
			case 0x63:
			case 0x73: {
				// LD (nn), rr
				const nn = this.fetchWord();
				const rp = (op >> 4) & 3;
				const val = this.getReg16(rp);
				this.writeByte(nn, val & 0xff);
				this.writeByte((nn + 1) & 0xffff, (val >> 8) & 0xff);
				break;
			}
			case 0x4b:
			case 0x5b:
			case 0x6b:
			case 0x7b: {
				// LD rr, (nn)
				const nn = this.fetchWord();
				const rp = (op >> 4) & 3;
				const lo = this.readByte(nn);
				const hi = this.readByte((nn + 1) & 0xffff);
				this.setReg16(rp, (hi << 8) | lo);
				break;
			}

			// ADC HL, rr / SBC HL, rr
			case 0x4a:
			case 0x5a:
			case 0x6a:
			case 0x7a: {
				// ADC HL, rr
				const rp = (op >> 4) & 3;
				const hl = this.hl;
				const val = this.getReg16(rp);
				const c = this.f & Z80.CF;
				const result = hl + val + c;
				this.f =
					((result >> 8) & Z80.SF) |
					((result & 0xffff) === 0 ? Z80.ZF : 0) |
					((hl ^ val ^ result) & 0x1000 ? Z80.HF : 0) |
					((~(hl ^ val) & (hl ^ result) & 0x8000) !== 0 ? Z80.PF : 0) |
					(result > 0xffff ? Z80.CF : 0) |
					((result >> 8) & (Z80.F3 | Z80.F5));
				this.hl = result & 0xffff;
				break;
			}
			case 0x42:
			case 0x52:
			case 0x62:
			case 0x72: {
				// SBC HL, rr
				const rp = (op >> 4) & 3;
				const hl = this.hl;
				const val = this.getReg16(rp);
				const c = this.f & Z80.CF;
				const result = hl - val - c;
				this.f =
					((result >> 8) & Z80.SF) |
					((result & 0xffff) === 0 ? Z80.ZF : 0) |
					Z80.NF |
					((hl ^ val ^ result) & 0x1000 ? Z80.HF : 0) |
					(((hl ^ val) & (hl ^ result) & 0x8000) !== 0 ? Z80.PF : 0) |
					(result < 0 ? Z80.CF : 0) |
					((result >> 8) & (Z80.F3 | Z80.F5));
				this.hl = result & 0xffff;
				break;
			}

			// IM 0/1/2 - no effect in our sim
			case 0x46:
			case 0x56:
			case 0x5e:
			case 0x4e:
			case 0x66:
			case 0x6e:
			case 0x76:
			case 0x7e:
				break;

			// IN/OUT - NOP in our sim
			case 0x40:
			case 0x48:
			case 0x50:
			case 0x58:
			case 0x60:
			case 0x68:
			case 0x70:
			case 0x78:
				// IN r, (C) - set register to 0
				this.setReg((op >> 3) & 7, 0);
				break;
			case 0x41:
			case 0x49:
			case 0x51:
			case 0x59:
			case 0x61:
			case 0x69:
			case 0x71:
			case 0x79:
				break; // OUT (C), r - NOP

			// RRD / RLD
			case 0x67: {
				// RRD
				const m = this.readByte(this.hl);
				this.writeByte(this.hl, ((this.a << 4) | (m >> 4)) & 0xff);
				this.a = (this.a & 0xf0) | (m & 0x0f);
				this.f =
					(this.f & Z80.CF) | this.szFlags(this.a) | (this.parity(this.a) ? Z80.PF : 0);
				break;
			}
			case 0x6f: {
				// RLD
				const m = this.readByte(this.hl);
				this.writeByte(this.hl, ((m << 4) | (this.a & 0x0f)) & 0xff);
				this.a = (this.a & 0xf0) | (m >> 4);
				this.f =
					(this.f & Z80.CF) | this.szFlags(this.a) | (this.parity(this.a) ? Z80.PF : 0);
				break;
			}

			// Block I/O - NOP in our sim
			case 0xa2:
			case 0xaa:
			case 0xb2:
			case 0xba:
			case 0xa3:
			case 0xab:
			case 0xb3:
			case 0xbb:
				break;

			default:
				break; // Unknown ED opcode - NOP
		}
	}

	// Block transfer: LDI
	private ldi(): void {
		const val = this.readByte(this.hl);
		this.writeByte(this.de, val);
		this.hl = (this.hl + 1) & 0xffff;
		this.de = (this.de + 1) & 0xffff;
		this.bc = (this.bc - 1) & 0xffff;
		const n = (val + this.a) & 0xff;
		this.f =
			(this.f & (Z80.SF | Z80.ZF | Z80.CF)) |
			(this.bc !== 0 ? Z80.PF : 0) |
			(n & Z80.F3) |
			((n & 0x02) !== 0 ? Z80.F5 : 0);
	}

	// Block transfer: LDD
	private ldd(): void {
		const val = this.readByte(this.hl);
		this.writeByte(this.de, val);
		this.hl = (this.hl - 1) & 0xffff;
		this.de = (this.de - 1) & 0xffff;
		this.bc = (this.bc - 1) & 0xffff;
		const n = (val + this.a) & 0xff;
		this.f =
			(this.f & (Z80.SF | Z80.ZF | Z80.CF)) |
			(this.bc !== 0 ? Z80.PF : 0) |
			(n & Z80.F3) |
			((n & 0x02) !== 0 ? Z80.F5 : 0);
	}

	// Block transfer: LDIR (LDI repeated)
	private ldir(): void {
		this.ldi();
		if (this.bc !== 0) {
			this.pc = (this.pc - 2) & 0xffff; // repeat
		}
	}

	// Block transfer: LDDR (LDD repeated)
	private lddr(): void {
		this.ldd();
		if (this.bc !== 0) {
			this.pc = (this.pc - 2) & 0xffff; // repeat
		}
	}

	// Block compare: CPI
	private cpi(): void {
		const val = this.readByte(this.hl);
		const result = (this.a - val) & 0xff;
		this.hl = (this.hl + 1) & 0xffff;
		this.bc = (this.bc - 1) & 0xffff;
		const n = (result - ((this.a ^ val ^ result) & Z80.HF ? 1 : 0)) & 0xff;
		this.f =
			(this.f & Z80.CF) |
			this.szFlags(result) |
			Z80.NF |
			((this.a ^ val ^ result) & Z80.HF) |
			(this.bc !== 0 ? Z80.PF : 0) |
			(n & Z80.F3) |
			((n & 0x02) !== 0 ? Z80.F5 : 0);
	}

	// Block compare: CPD
	private cpd(): void {
		const val = this.readByte(this.hl);
		const result = (this.a - val) & 0xff;
		this.hl = (this.hl - 1) & 0xffff;
		this.bc = (this.bc - 1) & 0xffff;
		const n = (result - ((this.a ^ val ^ result) & Z80.HF ? 1 : 0)) & 0xff;
		this.f =
			(this.f & Z80.CF) |
			this.szFlags(result) |
			Z80.NF |
			((this.a ^ val ^ result) & Z80.HF) |
			(this.bc !== 0 ? Z80.PF : 0) |
			(n & Z80.F3) |
			((n & 0x02) !== 0 ? Z80.F5 : 0);
	}

	// Block compare: CPIR
	private cpir(): void {
		this.cpi();
		if (this.bc !== 0 && (this.f & Z80.ZF) === 0) {
			this.pc = (this.pc - 2) & 0xffff;
		}
	}

	// Block compare: CPDR
	private cpdr(): void {
		this.cpd();
		if (this.bc !== 0 && (this.f & Z80.ZF) === 0) {
			this.pc = (this.pc - 2) & 0xffff;
		}
	}

	// DD prefix: IX instructions - simplified, skip prefix and use HL
	private executeDD(): void {
		const op = this.fetch();
		if (op === 0xcb) {
			// DD CB: IX bit operations - skip displacement, exec CB
			this.fetch(); // displacement (skip)
			this.executeCB();
		} else if (op === 0xdd || op === 0xfd || op === 0xed) {
			// Nested prefix - re-dispatch
			this.executeMain(op);
		} else {
			// Execute as normal opcode (IX treated as HL)
			this.executeMain(op);
		}
	}

	// FD prefix: IY instructions - simplified, skip prefix and use HL
	private executeFD(): void {
		const op = this.fetch();
		if (op === 0xcb) {
			this.fetch(); // displacement
			this.executeCB();
		} else if (op === 0xdd || op === 0xfd || op === 0xed) {
			this.executeMain(op);
		} else {
			this.executeMain(op);
		}
	}

	private signedByte(b: number): number {
		return b > 127 ? b - 256 : b;
	}
}
