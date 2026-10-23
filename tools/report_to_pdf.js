// Render doc/report/report.html (built by pandoc in tools/build_docs.sh) to the
// PDF submitted with the repository. Uses Playwright's Chromium, which renders
// the MathML equations that a Word-to-PDF conversion in LibreOffice loses.
//
// Usage (from the repository root): node tools/report_to_pdf.js
// Needs: npm install playwright (and `npx playwright install chromium` once).
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const html = path.resolve("doc/report/report.html");
  const pdf = path.resolve("doc/report/MMA3001_Project_Report.pdf");
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + html, { waitUntil: "networkidle" });
  await page.pdf({
    path: pdf,
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true, // margins come from @page in report.css
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate:
      '<div style="width:100%;font-size:8pt;text-align:center;color:#555;">' +
      '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log("Report PDF -> " + path.relative(process.cwd(), pdf));
})();
