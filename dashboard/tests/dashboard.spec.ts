import { test, expect } from '@playwright/test'

test('creates a local run draft and preserves it after reload', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Demo mode', { exact: false })).toBeVisible()
  await page.getByRole('button', { name: '＋ New run' }).click()
  await page.getByLabel('Task and acceptance criteria').fill('Fix retries and verify regression tests')
  await page.getByRole('button', { name: 'Save run draft' }).click()
  await expect(page.getByRole('status')).toContainText('Connect a backend')
  await page.reload()
  await page.getByLabel('Filter runs').selectOption('Queued')
  await expect(page.getByRole('table')).toContainText('Fix retries and verify regression tests')
  await expect(page.getByRole('table')).not.toContainText('Review sandbox')
})

test('persists model preferences without implying a live connection', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Models', exact: true }).click()
  await page.getByLabel('Model identifier').fill('my-local-model')
  await page.getByRole('button', { name: 'Save preferences' }).click()
  await page.reload()
  await expect(page.getByRole('heading', { name: 'my-local-model' })).toBeVisible()
  await expect(page.getByText('Not connected', { exact: true })).toBeVisible()
})

test('navigation and run details work on a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'View Add retry handling' }).click()
  await expect(page.getByRole('dialog')).toContainText('Illustrative demo result')
  await page.getByRole('button', { name: 'Back to workspace' }).click()
  await page.getByRole('button', { name: 'Memory', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'A little context goes a long way.' })).toBeVisible()
})
