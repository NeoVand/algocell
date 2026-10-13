<script lang="ts">
	let {
		options,
		value,
		onchange,
		label,
		id,
		disabled = false,
		placeholder = 'Choose…'
	} = $props<{
		options: { value: string; label: string; group?: string }[];
		value: string;
		onchange: (value: string) => void;
		label: string;
		id?: string;
		disabled?: boolean;
		placeholder?: string;
	}>();
	const uid = $props.id();
	let open = $state(false);
	let highlighted = $state(0);
	let wrapper: HTMLDivElement;
	let trigger: HTMLButtonElement;
	const selected = $derived(
		options.find(
			(option: { value: string; label: string; group?: string }) => option.value === value
		)
	);
	function show() {
		if (disabled) return;
		highlighted = Math.max(
			0,
			options.findIndex(
				(option: { value: string; label: string; group?: string }) => option.value === value
			)
		);
		open = !open;
	}
	function choose(index: number) {
		const option = options[index];
		if (!option) return;
		onchange(option.value);
		open = false;
		trigger?.focus();
	}
	function keyboard(event: KeyboardEvent) {
		if (event.key === 'Escape' && open) {
			event.preventDefault();
			event.stopPropagation();
			open = false;
		} else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
			event.preventDefault();
			event.stopPropagation();
			if (!open) {
				highlighted = Math.max(
					0,
					options.findIndex(
						(option: { value: string; label: string; group?: string }) => option.value === value
					)
				);
				open = true;
			} else
				highlighted =
					event.key === 'Home'
						? 0
						: event.key === 'End'
							? options.length - 1
							: (highlighted + (event.key === 'ArrowDown' ? 1 : -1) + options.length) %
								options.length;
			requestAnimationFrame(() =>
				document
					.getElementById(`${uid}-option-${highlighted}`)
					?.scrollIntoView({ block: 'nearest' })
			);
		} else if ((event.key === 'Enter' || event.key === ' ') && open) {
			event.preventDefault();
			event.stopPropagation();
			choose(highlighted);
		} else if (event.key === 'Tab') open = false;
	}
</script>

<svelte:window
	onpointerdown={(event) => {
		if (open && event.target instanceof Node && !wrapper?.contains(event.target)) open = false;
	}}
/>
<div
	class="instrument-select"
	{@attach (element) => {
		wrapper = element;
	}}
>
	<button
		{@attach (element) => {
			trigger = element;
		}}
		id={id ?? uid}
		type="button"
		role="combobox"
		aria-label={label}
		aria-expanded={open}
		aria-controls={`${uid}-list`}
		aria-haspopup="listbox"
		aria-activedescendant={open ? `${uid}-option-${highlighted}` : undefined}
		data-value={value}
		{disabled}
		onclick={show}
		onkeydown={keyboard}
	>
		<span>{selected?.label ?? placeholder}</span><svg
			width="12"
			height="12"
			viewBox="0 0 12 12"
			fill="none"
			aria-hidden="true"><path d="m3 4.5 3 3 3-3" stroke="currentColor" stroke-width="1.2" /></svg
		>
	</button>
	{#if open}
		<div class="choices" id={`${uid}-list`} role="listbox" aria-label={label}>
			{#each options as option, index (option.value)}
				{#if option.group && (index === 0 || options[index - 1].group !== option.group)}<div
						class="group-label"
						role="presentation"
					>
						{option.group}
					</div>{/if}
				<button
					type="button"
					id={`${uid}-option-${index}`}
					role="option"
					data-value={option.value}
					aria-selected={value === option.value}
					class:highlighted={highlighted === index}
					tabindex="-1"
					onpointermove={() => (highlighted = index)}
					onclick={() => choose(index)}
					><span>{option.label}</span><span class="check" aria-hidden="true"
						>{value === option.value ? '✓' : ''}</span
					></button
				>
			{/each}
		</div>
	{/if}
</div>

<style>
	.instrument-select {
		position: relative;
		min-width: 0;
		width: 100%;
	}
	button {
		font: inherit;
		font-size: 12px;
		color: var(--text-primary);
		cursor: pointer;
		border-radius: 5px;
	}
	button[role='combobox'] {
		display: flex;
		width: 100%;
		min-height: 32px;
		gap: 12px;
		align-items: center;
		justify-content: space-between;
		border: 1px solid var(--border-muted);
		background: #21211f;
		padding: 5px 9px;
		text-align: left;
		transition: background 160ms ease;
	}
	button[role='combobox']:hover {
		background: #282824;
	}
	button span:first-child {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	button svg {
		flex: 0 0 auto;
		color: var(--text-muted);
	}
	button:disabled {
		opacity: 0.45;
		cursor: default;
	}
	.choices {
		position: absolute;
		top: calc(100% + 4px);
		inset-inline: 0;
		min-width: 160px;
		max-height: min(340px, 45vh);
		overflow: auto;
		z-index: 100;
		padding: 4px;
		border: 1px solid #414039;
		border-radius: 6px;
		background: #20201e;
		box-shadow: 0 10px 24px #0008;
	}
	.choices button {
		display: flex;
		width: 100%;
		gap: 8px;
		justify-content: space-between;
		align-items: center;
		min-height: 32px;
		padding: 6px 8px;
		text-align: left;
		border: 0;
		background: transparent;
	}
	.choices button.highlighted {
		background: #ffffff0b;
	}
	.choices button[aria-selected='true'] {
		color: var(--accent-warm);
	}
	.check {
		width: 12px;
		flex-shrink: 0;
	}
	.group-label {
		padding: 8px 8px 4px;
		font-size: 10px;
		color: var(--text-muted);
		text-transform: capitalize;
	}
</style>
