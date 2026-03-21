// Z80 opcode-aware colormap
// Maps each of the 256 byte values to an RGBA color
// Uses HSL-based palette for visual harmony

function hsl(h: number, s: number, l: number): number {
	// Convert HSL to RGB
	h = ((h % 360) + 360) % 360;
	s = Math.max(0, Math.min(1, s));
	l = Math.max(0, Math.min(1, l));
	const c = (1 - Math.abs(2 * l - 1)) * s;
	const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
	const m = l - c / 2;
	let r = 0,
		g = 0,
		b = 0;
	if (h < 60) {
		r = c;
		g = x;
	} else if (h < 120) {
		r = x;
		g = c;
	} else if (h < 180) {
		g = c;
		b = x;
	} else if (h < 240) {
		g = x;
		b = c;
	} else if (h < 300) {
		r = x;
		b = c;
	} else {
		r = c;
		b = x;
	}
	return packRGBA(
		Math.round((r + m) * 255),
		Math.round((g + m) * 255),
		Math.round((b + m) * 255),
		255
	);
}

export function createDefaultColormap(): Uint32Array {
	const cmap = new Uint32Array(256);

	// Base: subtle hue-shifted gradient based on byte value
	// Each byte gets a unique color from a continuous HSL sweep
	for (let i = 0; i < 256; i++) {
		const hue = (i * 1.41) % 360; // Golden-angle-ish sweep
		cmap[i] = hsl(hue, 0.15, 0.28); // Muted dark tones
	}

	// NOP - very dark (near black)
	cmap[0x00] = hsl(0, 0, 0.06);
	// HALT - dark charcoal
	cmap[0x76] = hsl(0, 0, 0.14);

	// 16-bit immediate loads - vivid saturated
	cmap[0x01] = hsl(350, 0.75, 0.55); // LD BC,nn - rose
	cmap[0x11] = hsl(150, 0.7, 0.48); // LD DE,nn - emerald
	cmap[0x21] = hsl(220, 0.75, 0.58); // LD HL,nn - royal blue
	cmap[0x31] = hsl(50, 0.8, 0.55); // LD SP,nn - amber

	// Memory loads - lighter pastel variants
	cmap[0x2a] = hsl(210, 0.5, 0.75); // LD HL,(nn) - light periwinkle
	cmap[0x3a] = hsl(0, 0, 0.78); // LD A,(nn) - silver
	cmap[0x22] = hsl(230, 0.45, 0.68); // LD (nn),HL - soft blue
	cmap[0x32] = hsl(140, 0.4, 0.62); // LD (nn),A - soft green

	// Block transfer - warm reds/corals
	cmap[0xed] = hsl(5, 0.85, 0.58); // ED prefix - coral red
	cmap[0xb0] = hsl(0, 0.8, 0.6); // LDIR - bright red
	cmap[0xb8] = hsl(10, 0.75, 0.62); // LDDR - warm red
	cmap[0xa0] = hsl(355, 0.7, 0.55); // LDI - crimson
	cmap[0xa8] = hsl(15, 0.65, 0.58); // LDD - salmon

	// Push/Pop - jewel tones
	cmap[0xc5] = hsl(280, 0.55, 0.58); // PUSH BC - amethyst
	cmap[0xd5] = hsl(170, 0.55, 0.5); // PUSH DE - teal
	cmap[0xe5] = hsl(35, 0.6, 0.55); // PUSH HL - bronze
	cmap[0xf5] = hsl(320, 0.5, 0.58); // PUSH AF - orchid
	cmap[0xc1] = hsl(275, 0.45, 0.5); // POP BC - purple
	cmap[0xd1] = hsl(165, 0.45, 0.45); // POP DE - dark teal
	cmap[0xe1] = hsl(30, 0.5, 0.5); // POP HL - dark bronze
	cmap[0xf1] = hsl(315, 0.4, 0.5); // POP AF - mauve

	// Jumps/Calls - warm gold/orange family
	cmap[0xc3] = hsl(30, 0.9, 0.55); // JP nn - tangerine
	cmap[0xcd] = hsl(45, 0.85, 0.55); // CALL nn - gold
	cmap[0xc9] = hsl(38, 0.7, 0.5); // RET - dark gold
	cmap[0x18] = hsl(25, 0.75, 0.52); // JR - burnt orange
	cmap[0x10] = hsl(20, 0.65, 0.48); // DJNZ - russet

	// CB prefix - electric blue
	cmap[0xcb] = hsl(200, 0.8, 0.6); // CB prefix - cyan-blue

	// EX instructions - distinctive
	cmap[0xe3] = hsl(60, 0.6, 0.55); // EX (SP),HL - chartreuse
	cmap[0xeb] = hsl(90, 0.5, 0.5); // EX DE,HL - lime
	cmap[0x08] = hsl(55, 0.5, 0.5); // EX AF,AF' - olive

	// DD/FD prefixes
	cmap[0xdd] = hsl(190, 0.6, 0.5); // DD - steel blue
	cmap[0xfd] = hsl(260, 0.5, 0.5); // FD - indigo

	// 8-bit loads between registers (x=1 group, 0x40-0x7F minus HALT)
	// Give them subtle distinct hues based on destination register
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			if (y === 6 && z === 6) continue; // HALT
			const op = 0x40 + y * 8 + z;
			cmap[op] = hsl(200 + y * 18, 0.2 + z * 0.03, 0.32 + y * 0.02);
		}
	}

	// ALU group (x=2, 0x80-0xBF) - warm shifted
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			const op = 0x80 + y * 8 + z;
			cmap[op] = hsl(340 + y * 12, 0.18 + z * 0.02, 0.3 + y * 0.02);
		}
	}

	return cmap;
}

function packRGBA(r: number, g: number, b: number, a: number): number {
	return (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | ((a & 0xff) << 24);
}

export function unpackRGBA(packed: number): [number, number, number, number] {
	return [packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff, (packed >> 24) & 0xff];
}

// Colormap: ocean theme - cool blues and teals
export function createOceanColormap(): Uint32Array {
	const cmap = new Uint32Array(256);
	for (let i = 0; i < 256; i++) {
		cmap[i] = hsl(200 + (i * 0.3), 0.2, 0.22 + (i / 256) * 0.08);
	}
	cmap[0x00] = hsl(210, 0.1, 0.05);
	cmap[0x76] = hsl(210, 0.1, 0.12);
	cmap[0x01] = hsl(340, 0.65, 0.55);
	cmap[0x11] = hsl(160, 0.7, 0.45);
	cmap[0x21] = hsl(210, 0.85, 0.55);
	cmap[0x31] = hsl(180, 0.7, 0.5);
	cmap[0xed] = hsl(0, 0.75, 0.55);
	cmap[0xb0] = hsl(355, 0.7, 0.55);
	cmap[0xb8] = hsl(5, 0.65, 0.55);
	cmap[0xa0] = hsl(350, 0.6, 0.5);
	cmap[0xa8] = hsl(10, 0.55, 0.52);
	cmap[0xc5] = hsl(260, 0.5, 0.55);
	cmap[0xd5] = hsl(180, 0.6, 0.48);
	cmap[0xe5] = hsl(195, 0.55, 0.5);
	cmap[0xf5] = hsl(290, 0.45, 0.52);
	cmap[0xc1] = hsl(255, 0.4, 0.48);
	cmap[0xd1] = hsl(175, 0.5, 0.42);
	cmap[0xe1] = hsl(190, 0.45, 0.45);
	cmap[0xf1] = hsl(285, 0.35, 0.48);
	cmap[0xc3] = hsl(40, 0.8, 0.55);
	cmap[0xcd] = hsl(50, 0.75, 0.55);
	cmap[0xc9] = hsl(35, 0.6, 0.48);
	cmap[0x18] = hsl(30, 0.65, 0.5);
	cmap[0x10] = hsl(25, 0.55, 0.45);
	cmap[0xcb] = hsl(190, 0.75, 0.58);
	cmap[0xe3] = hsl(170, 0.6, 0.52);
	cmap[0xeb] = hsl(150, 0.5, 0.48);
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			if (y === 6 && z === 6) continue;
			cmap[0x40 + y * 8 + z] = hsl(195 + y * 10, 0.18 + z * 0.02, 0.28 + y * 0.015);
		}
	}
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			cmap[0x80 + y * 8 + z] = hsl(220 + y * 8, 0.15 + z * 0.02, 0.26 + y * 0.015);
		}
	}
	return cmap;
}

// Colormap: thermal - black → deep magenta → red → orange → yellow → white
export function createThermalColormap(): Uint32Array {
	const cmap = new Uint32Array(256);
	// Base: true thermal gradient mapped by byte value
	for (let i = 0; i < 256; i++) {
		const t = i / 255;
		// Black → deep purple → red → orange → yellow
		const h = 300 - t * 300; // 300(magenta) → 0(red) for low-mid, wraps to yellow
		const hue = t < 0.5 ? 300 - t * 240 : 60 - (t - 0.5) * 120; // magenta→red→orange→yellow
		const sat = 0.6 + t * 0.2;
		const lit = 0.1 + t * 0.22;
		cmap[i] = hsl(hue, sat, lit);
	}

	// NOP - pure black
	cmap[0x00] = packRGBA(0, 0, 0, 255);
	// HALT - very dark
	cmap[0x76] = packRGBA(15, 5, 10, 255);

	// 16-bit immediate loads - hot bright colors
	cmap[0x01] = hsl(50, 0.95, 0.6);   // LD BC,nn - bright yellow
	cmap[0x11] = hsl(30, 0.9, 0.55);   // LD DE,nn - hot orange
	cmap[0x21] = hsl(10, 0.9, 0.55);   // LD HL,nn - bright red
	cmap[0x31] = hsl(60, 0.85, 0.65);  // LD SP,nn - pale yellow

	// Memory loads - warm mid tones
	cmap[0x2a] = hsl(25, 0.7, 0.5);
	cmap[0x3a] = hsl(35, 0.65, 0.52);
	cmap[0x22] = hsl(15, 0.65, 0.48);
	cmap[0x32] = hsl(40, 0.6, 0.5);

	// Block transfer - intense reds
	cmap[0xed] = hsl(355, 0.9, 0.5);
	cmap[0xb0] = hsl(0, 0.85, 0.52);
	cmap[0xb8] = hsl(350, 0.8, 0.48);
	cmap[0xa0] = hsl(345, 0.75, 0.45);
	cmap[0xa8] = hsl(5, 0.7, 0.5);

	// Push/Pop - warm oranges and deep reds
	cmap[0xc5] = hsl(20, 0.7, 0.52);   // PUSH BC
	cmap[0xd5] = hsl(35, 0.65, 0.5);   // PUSH DE
	cmap[0xe5] = hsl(45, 0.75, 0.55);  // PUSH HL - gold
	cmap[0xf5] = hsl(340, 0.6, 0.48);  // PUSH AF - deep rose
	cmap[0xc1] = hsl(15, 0.6, 0.45);   // POP BC
	cmap[0xd1] = hsl(30, 0.55, 0.42);  // POP DE
	cmap[0xe1] = hsl(42, 0.65, 0.5);   // POP HL - amber
	cmap[0xf1] = hsl(335, 0.5, 0.42);  // POP AF

	// Jumps/Calls - brightest yellows/whites (hottest)
	cmap[0xc3] = hsl(55, 0.95, 0.65);  // JP nn - bright yellow
	cmap[0xcd] = hsl(48, 0.9, 0.62);   // CALL nn - gold
	cmap[0xc9] = hsl(42, 0.8, 0.55);   // RET
	cmap[0x18] = hsl(38, 0.75, 0.5);   // JR
	cmap[0x10] = hsl(32, 0.7, 0.48);   // DJNZ

	// CB prefix - deep magenta (cool end of thermal)
	cmap[0xcb] = hsl(310, 0.7, 0.5);

	// EX instructions - hot orange-yellow
	cmap[0xe3] = hsl(40, 0.85, 0.58);  // EX (SP),HL
	cmap[0xeb] = hsl(50, 0.7, 0.52);   // EX DE,HL
	cmap[0x08] = hsl(35, 0.6, 0.48);   // EX AF,AF'

	// DD/FD prefixes - deep warm tones
	cmap[0xdd] = hsl(320, 0.55, 0.45);
	cmap[0xfd] = hsl(290, 0.5, 0.42);

	// 8-bit register loads - warm gradient (dark red → orange)
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			if (y === 6 && z === 6) continue;
			cmap[0x40 + y * 8 + z] = hsl(350 + y * 8, 0.35 + z * 0.04, 0.2 + y * 0.03);
		}
	}
	// ALU group - deep magenta → red
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			cmap[0x80 + y * 8 + z] = hsl(310 + y * 10, 0.3 + z * 0.04, 0.18 + y * 0.025);
		}
	}
	return cmap;
}

// Colormap: rainbow - full spectrum cycle
export function createRainbowColormap(): Uint32Array {
	const cmap = new Uint32Array(256);
	for (let i = 0; i < 256; i++) {
		const hue = (i / 256) * 360;
		cmap[i] = hsl(hue, 0.55, 0.38);
	}
	cmap[0x00] = hsl(0, 0, 0.04);
	cmap[0x76] = hsl(0, 0, 0.12);
	// Key opcodes get brighter saturated versions of their hue position
	cmap[0x01] = hsl(350, 0.85, 0.55);
	cmap[0x11] = hsl(150, 0.8, 0.48);
	cmap[0x21] = hsl(220, 0.85, 0.58);
	cmap[0x31] = hsl(50, 0.85, 0.55);
	cmap[0xed] = hsl(0, 0.85, 0.55);
	cmap[0xb0] = hsl(5, 0.8, 0.52);
	cmap[0xb8] = hsl(10, 0.75, 0.54);
	cmap[0xa0] = hsl(355, 0.7, 0.5);
	cmap[0xa8] = hsl(15, 0.65, 0.52);
	cmap[0xc5] = hsl(280, 0.65, 0.55);
	cmap[0xd5] = hsl(170, 0.6, 0.48);
	cmap[0xe5] = hsl(35, 0.7, 0.55);
	cmap[0xf5] = hsl(320, 0.55, 0.55);
	cmap[0xc1] = hsl(275, 0.55, 0.48);
	cmap[0xd1] = hsl(165, 0.5, 0.42);
	cmap[0xe1] = hsl(30, 0.6, 0.5);
	cmap[0xf1] = hsl(315, 0.45, 0.48);
	cmap[0xc3] = hsl(35, 0.9, 0.55);
	cmap[0xcd] = hsl(50, 0.85, 0.58);
	cmap[0xc9] = hsl(40, 0.7, 0.5);
	cmap[0x18] = hsl(28, 0.75, 0.5);
	cmap[0x10] = hsl(20, 0.65, 0.45);
	cmap[0xcb] = hsl(200, 0.8, 0.58);
	cmap[0xe3] = hsl(60, 0.7, 0.55);
	cmap[0xeb] = hsl(90, 0.55, 0.48);
	cmap[0xdd] = hsl(190, 0.65, 0.52);
	cmap[0xfd] = hsl(260, 0.55, 0.52);
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			if (y === 6 && z === 6) continue;
			const op = 0x40 + y * 8 + z;
			cmap[op] = hsl((op / 256) * 360, 0.35 + z * 0.03, 0.35 + y * 0.015);
		}
	}
	for (let y = 0; y < 8; y++) {
		for (let z = 0; z < 8; z++) {
			const op = 0x80 + y * 8 + z;
			cmap[op] = hsl((op / 256) * 360, 0.3 + z * 0.03, 0.33 + y * 0.015);
		}
	}
	return cmap;
}

export type ColormapName = 'rainbow' | 'ocean' | 'thermal';

export const COLORMAP_NAMES: ColormapName[] = ['rainbow', 'ocean', 'thermal'];

export function createColormap(name: ColormapName): Uint32Array {
	switch (name) {
		case 'ocean': return createOceanColormap();
		case 'thermal': return createThermalColormap();
		case 'rainbow': return createRainbowColormap();
		default: return createRainbowColormap();
	}
}
