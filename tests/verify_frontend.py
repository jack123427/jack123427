
import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Listen for all console events and print them
        page.on("console", lambda msg: print(f"BROWSER LOG: {msg.text()}"))

        try:
            await page.goto("http://127.0.0.1:8080")

            # Wait for the page to load and elements to be ready
            await page.wait_for_selector("#stockSelector")
            await page.wait_for_selector("#modelSelector")

            # Select the Prophet model
            await page.select_option("#modelSelector", "prophet")

            # Select a stock
            await page.select_option("#stockSelector", "2330")

            # Click the predict button
            await page.click("#predictBtn")

            # Wait for the prediction chart to be visible
            chart_locator = page.locator("#predictionChart")
            await expect(chart_locator).to_be_visible(timeout=60000)

            # Take a screenshot
            await page.screenshot(path="verification/prophet_prediction.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    # Create the verification directory if it doesn't exist
    import os
    os.makedirs("verification", exist_ok=True)

    asyncio.run(main())
