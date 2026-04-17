"""
utils/pdf_export.py — Convert an HTML forensic report to PDF using playwright (headless Chromium).

Usage:
    from utils.pdf_export import html_to_pdf
    pdf_path = html_to_pdf("reports/Report_sample.html")

Requirements:
    pip install playwright
    playwright install chromium --with-deps
"""
import os


def html_to_pdf(html_path: str, pdf_path: str = None) -> str | None:
    """
    Render *html_path* with headless Chromium and save the result as PDF.

    Parameters
    ----------
    html_path : str
        Absolute or relative path to the HTML file to render.
    pdf_path : str, optional
        Where to write the PDF. Defaults to the same basename with .pdf extension.

    Returns
    -------
    str or None
        The path to the generated PDF, or None if playwright is not installed.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    import sys
    import asyncio
    if sys.platform == 'win32':
        # Playwright requires ProactorEventLoop on Windows to support subprocesses
        try:
            from asyncio import WindowsProactorEventLoopPolicy
        except ImportError:
            pass
        else:
            if not isinstance(asyncio.get_event_loop_policy(), WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(WindowsProactorEventLoopPolicy())

    if pdf_path is None:
        pdf_path = os.path.splitext(html_path)[0] + ".pdf"

    abs_html = os.path.abspath(html_path)
    file_url = f"file:///{abs_html.replace(os.sep, '/')}"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(file_url, wait_until="networkidle")
        # Allow Google Fonts to finish loading
        page.wait_for_timeout(1500)
        page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "8mm", "right": "8mm"},
        )
        browser.close()

    return pdf_path
