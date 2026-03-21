<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Algocell — Artificial Life via Z80 Machine Code</title>
	<meta name="description" content="Watch self-replicating programs emerge from random bytes. A WebGPU-accelerated artificial life simulation running Z80 machine code on a 200x200 grid." />
	<meta property="og:title" content="Algocell — Artificial Life via Z80 Machine Code" />
	<meta property="og:description" content="Watch self-replicating programs emerge from random bytes. WebGPU-accelerated Z80 simulation — over 50B ops/sec on a MacBook Air M3." />
	<meta property="og:image" content="/og-image.png" />
	<meta property="og:type" content="website" />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="Algocell — Artificial Life via Z80 Machine Code" />
	<meta name="twitter:description" content="Watch self-replicating programs emerge from random bytes. WebGPU-accelerated Z80 simulation." />
	<meta name="twitter:image" content="/og-image.png" />
</svelte:head>
{@render children()}
