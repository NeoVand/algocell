<script lang="ts">
	import { resolve } from '$app/paths';
	import LegacyInstrument from '$lib/components/LegacyInstrument.svelte';
</script>

<LegacyInstrument />
<a class="legacy-return" href={resolve('/')} aria-label="Return to the research desk"
	>Algocell · Research desk</a
>

<style>
	.legacy-return {
		position: fixed;
		top: 12px;
		left: 12px;
		z-index: 100;
		color: #fff;
		background: #25313a;
		padding: 10px 14px;
		border-radius: 6px;
		font-size: 14px;
		text-decoration: none;
	}
	.legacy-return:focus-visible {
		outline: 3px solid #168a8d;
		outline-offset: 3px;
	}
	@media (max-width: 768px) {
		.legacy-return {
			left: auto;
			right: 8px;
			top: 8px;
			max-width: 140px;
			font-size: 12px;
		}
	}
</style>
