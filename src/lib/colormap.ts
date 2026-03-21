// Z80 opcode-aware colormap
// Maps each of the 256 byte values to an RGBA color
// Special colors for key Z80 instructions that appear in self-replicators

export function createDefaultColormap(): Uint32Array {
	const cmap = new Uint32Array(256);

	// Base: grayscale gradient
	for (let i = 0; i < 256; i++) {
		const g = 64 + Math.floor(i / 8);
		cmap[i] = packRGBA(g, g, g, 255);
	}

	// Special opcodes - colors matching the original
	cmap[0x00] = packRGBA(0, 0, 0, 255); // NOP - black
	cmap[0x76] = packRGBA(40, 40, 40, 255); // HALT - dark gray

	// 16-bit immediate loads
	cmap[0x01] = packRGBA(220, 60, 60, 255); // LD BC,nn - red
	cmap[0x11] = packRGBA(60, 200, 60, 255); // LD DE,nn - green
	cmap[0x21] = packRGBA(60, 100, 220, 255); // LD HL,nn - blue
	cmap[0x31] = packRGBA(200, 200, 60, 255); // LD SP,nn - yellow

	// Memory loads
	cmap[0x2a] = packRGBA(255, 255, 255, 255); // LD HL,(nn) - white
	cmap[0x3a] = packRGBA(200, 200, 200, 255); // LD A,(nn) - light gray
	cmap[0x22] = packRGBA(180, 180, 255, 255); // LD (nn),HL - light blue
	cmap[0x32] = packRGBA(180, 255, 180, 255); // LD (nn),A - light green

	// Block transfer (ED prefix marker)
	cmap[0xed] = packRGBA(255, 50, 50, 255); // ED prefix - bright red
	cmap[0xb0] = packRGBA(255, 80, 80, 255); // LDIR (after ED) - red
	cmap[0xb8] = packRGBA(255, 100, 100, 255); // LDDR (after ED) - lighter red
	cmap[0xa0] = packRGBA(220, 80, 80, 255); // LDI (after ED) - red variant
	cmap[0xa8] = packRGBA(220, 100, 100, 255); // LDD (after ED) - red variant

	// Push/Pop
	cmap[0xc5] = packRGBA(180, 130, 200, 255); // PUSH BC - purple
	cmap[0xd5] = packRGBA(130, 200, 180, 255); // PUSH DE - teal
	cmap[0xe5] = packRGBA(200, 180, 130, 255); // PUSH HL - tan
	cmap[0xf5] = packRGBA(200, 130, 180, 255); // PUSH AF - pink
	cmap[0xc1] = packRGBA(160, 110, 180, 255); // POP BC
	cmap[0xd1] = packRGBA(110, 180, 160, 255); // POP DE
	cmap[0xe1] = packRGBA(180, 160, 110, 255); // POP HL
	cmap[0xf1] = packRGBA(180, 110, 160, 255); // POP AF

	// Jumps
	cmap[0xc3] = packRGBA(255, 180, 0, 255); // JP nn - orange
	cmap[0xcd] = packRGBA(255, 220, 0, 255); // CALL nn - gold
	cmap[0xc9] = packRGBA(200, 150, 0, 255); // RET - amber
	cmap[0x18] = packRGBA(220, 160, 40, 255); // JR - light orange
	cmap[0x10] = packRGBA(200, 140, 40, 255); // DJNZ - darker orange

	// CB prefix
	cmap[0xcb] = packRGBA(100, 180, 255, 255); // CB prefix - sky blue

	return cmap;
}

function packRGBA(r: number, g: number, b: number, a: number): number {
	return (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | ((a & 0xff) << 24);
}

export function unpackRGBA(packed: number): [number, number, number, number] {
	return [packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff, (packed >> 24) & 0xff];
}
