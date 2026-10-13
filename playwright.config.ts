import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	webServer: {
		command: 'npm run dev -- --host 127.0.0.1 --port 4183 --strictPort',
		url: 'http://127.0.0.1:4183',
		reuseExistingServer: false,
		timeout: 30_000
	},
	testMatch: '**/*.e2e.{ts,js}',
	fullyParallel: false,
	workers: 1,
	timeout: 60_000,
	expect: { timeout: 20_000 },
	use: {
		baseURL: 'http://127.0.0.1:4183',
		...devices['Desktop Chrome'],
		channel: 'chrome',
		viewport: { width: 1440, height: 1000 },
		launchOptions: { args: ['--enable-unsafe-webgpu'] },
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure'
	},
	reporter: [['list']]
});
