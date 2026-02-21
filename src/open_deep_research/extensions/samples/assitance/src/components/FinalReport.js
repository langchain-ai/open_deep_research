import React, { useEffect, useRef, useState, useMemo } from 'react';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import html2pdf from 'html2pdf.js';
import { Document, Packer, Paragraph, TextRun } from 'docx';
import { saveAs } from 'file-saver';
import LoadingIndicator from './LoadingIndicator';
import Tippy from '@tippyjs/react';
import BenchmarkResultCard from './BenchmarkResultCard';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/shift-away.css';

// Helper to convert heading text to URL-safe slug (must match Python _slugify in graph.py)
const slugify = (text) =>
  String(text)
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');

function FinalReport({ reportContent, isFocusedView, analysisData }) {
  // Function to detect if content is HTML (like database reports)
  const isHtmlContent = (content) => {
    if (!content) return false;
    // Check if content contains HTML structure with inline styles (database reports)
    return content.includes('<div style=') ||
      content.includes('<table style=') ||
      (content.includes('Database Analysis Results') && content.includes('<h3')) ||
      (content.includes('Database Query Results') && content.includes('<h1'));
  };

  // Function to detect if content is from benchmark mode
  const isBenchmarkContent = (content) => {
    if (!content) return false;

    // More flexible patterns to detect benchmark content headers
    // Checks for common benchmark section headers, possibly prefixed
    const benchmarkHeaderPatterns = [
      /((\*\*|#+\s*|(\d+\.\s*)?))?Direct Answer:/i,
      /((\*\*|#+\s*|(\d+\.\s*)?))?(Overall\s*)?Confidence:/i,
      /((\*\*|#+\s*|(\d+\.\s*)?))?(Key\s*)?Evidence:/i,
      /((\*\*|#+\s*|(\d+\.\s*)?))?Sources:/i,
    ];

    // Check if at least two or three of these characteristic headers are present
    let matchCount = 0;
    for (const pattern of benchmarkHeaderPatterns) {
      if (pattern.test(content)) {
        matchCount++;
      }
    }

    // Content is likely from benchmark mode if it has several characteristic headers
    // and is relatively short (benchmark answers are typically concise).
    const hasEnoughPatterns = matchCount >= 2;
    const isShortEnough = content.length < 3000; // Increased length slightly for flexibility

    return hasEnoughPatterns && isShortEnough;
  };

  // Function to parse benchmark content into structured data
  const parseBenchmarkContent = (content) => {
    const lines = content.split('\n');
    const result = {
      answer: '',
      confidenceLevel: '',
      confidence: 0.5, // Default
      evidence: '',
      sources: [],
      reasoning: '',
      limitations: '',
      expectedAnswer: ''
    };
    let currentSection = null;
    let sectionContent = [];

    const finalizeSection = () => {
      if (currentSection && sectionContent.length > 0) {
        const text = sectionContent.join(' ').trim();
        if (currentSection === 'answer') result.answer = text;
        else if (currentSection === 'confidence') {
          result.confidenceLevel = text;
          if (text.toUpperCase() === 'HIGH') result.confidence = 0.9;
          else if (text.toUpperCase() === 'MEDIUM') result.confidence = 0.6;
          else if (text.toUpperCase() === 'LOW') result.confidence = 0.3;
          else { // Try to parse numeric confidence if present
            const numericMatch = text.match(/(\d+(\.\d+)?)/);
            if (numericMatch) result.confidence = parseFloat(numericMatch[1]);
          }
        }
        else if (currentSection === 'evidence') result.evidence = text;
        else if (currentSection === 'sources') result.sources = sectionContent.map(s => s.replace(/^(\d+\.|-|\*)\s*/, '').trim()).filter(s => s);
        else if (currentSection === 'reasoning') result.reasoning = text;
        else if (currentSection === 'limitations') result.limitations = text;
        else if (currentSection === 'expectedAnswer') result.expectedAnswer = text;
      }
      sectionContent = [];
    };

    for (const line of lines) {
      const trimmed = line.trim();
      // Skip empty lines or lines that are just markdown separators like '---'
      if (!trimmed || trimmed === '---') continue;

      let newSection = null;
      let lineValue = '';

      // Regex to match various header styles: **, #, ##, ###, 1., etc.
      const headerRegex = /^(\*\*|#+\s*|(\d+\.\s*)?)(Direct Answer|Overall Confidence|Confidence|Key Evidence|Evidence|Sources|Reasoning|Limitations|Expected Answer):?/i;
      const match = trimmed.match(headerRegex);

      if (match) {
        const headerText = match[3].toLowerCase(); // Get the core header text (e.g., "direct answer")
        lineValue = trimmed.substring(match[0].length).trim(); // Get content after the header

        if (headerText.includes('direct answer')) newSection = 'answer';
        else if (headerText.includes('overall confidence')) newSection = 'confidence';
        else if (headerText.includes('confidence')) newSection = 'confidence'; // Broader match
        else if (headerText.includes('key evidence')) newSection = 'evidence';
        else if (headerText.includes('evidence')) newSection = 'evidence'; // Broader match
        else if (headerText.includes('sources')) newSection = 'sources';
        else if (headerText.includes('reasoning')) newSection = 'reasoning';
        else if (headerText.includes('limitations')) newSection = 'limitations';
        else if (headerText.includes('expected answer')) newSection = 'expectedAnswer';
      }

      if (newSection) {
        finalizeSection();
        currentSection = newSection;
        if (lineValue) sectionContent.push(lineValue);
      } else if (currentSection) {
        // If it's not a header, append to current section's content
        // For 'sources', each line is a new source item
        if (currentSection === 'sources') {
          // Remove list markers like "1. ", "- ", "* "
          const cleanedLine = trimmed.replace(/^(\d+\.|-|\*)\s*/, '');
          if (cleanedLine) sectionContent.push(cleanedLine);
        } else {
          sectionContent.push(trimmed);
        }
      } else if (!result.answer && trimmed) {
        // Fallback: if no section yet and line is not empty, assume it's part of the answer
        // This helps catch cases where the answer starts immediately without a "Direct Answer:" header
        currentSection = 'answer';
        sectionContent.push(trimmed);
      }
    }
    finalizeSection(); // Finalize the last section

    // Fallback if answer is still empty but content exists (e.g. LLM just gave the answer directly)
    if (!result.answer && content) {
      const firstMeaningfulLine = lines.find(l => {
        const t = l.trim();
        return t && !t.match(/^(\*\*|#+\s*|(\d+\.\s*)?)(Direct Answer|Overall Confidence|Confidence|Key Evidence|Evidence|Sources|Reasoning|Limitations|Expected Answer):?/i); // Not empty and not a header
      });
      if (firstMeaningfulLine) {
        result.answer = firstMeaningfulLine.trim();
      } else if (lines.length > 0 && lines[0].trim()) {
        result.answer = lines[0].trim();
      }
    }

    // If evidence is empty but key_evidence was parsed into it, it's fine.
    // If both are empty, and there's a reasoning field, sometimes evidence is in reasoning.
    if (!result.evidence && result.reasoning) {
      // A simple heuristic: if reasoning contains bullet points or "Source:", it might be evidence.
      if (result.reasoning.includes('- ') || result.reasoning.includes('* ') || /Source\s*\d*:/i.test(result.reasoning)) {
        // This is a basic assumption; might need refinement if it causes issues.
        // For now, we'll keep evidence separate unless explicitly told to merge.
      }
    }

    return result;
  };

  // Check if this is benchmark content
  const isBenchmark = isBenchmarkContent(reportContent);
  const benchmarkData = isBenchmark ? parseBenchmarkContent(reportContent) : null;

  // Extract report title from the content
  const extractReportTitle = (content) => {
    if (!content) return 'Research Report';

    // For benchmark mode, use a specific title
    if (isBenchmark) return 'Benchmark Question Result';

    // First, check for HTML h1 title (highest priority)
    const h1Match = content.match(/<h1>(.*?)<\/h1>/i);
    if (h1Match && h1Match[1]) {
      return h1Match[1].trim();
    }

    // Try to find a title in the first few lines of the report
    const lines = content.split('\n').slice(0, 20); // Look in first 20 lines
    const seenTitles = new Set(); // Track seen titles to prevent duplicates

    // Look for markdown headings first
    for (const line of lines) {
      const cleanLine = line.trim();
      if (cleanLine.startsWith('# ')) {
        const title = cleanLine.replace(/^# /, '');
        if (!seenTitles.has(title)) {
          seenTitles.add(title);
          return title;
        }
      }
    }

    // Then look for specific title patterns
    for (const line of lines) {
      const cleanLine = line.trim();
      if (cleanLine.match(/^(?:Profile of|Who is|Analysis of|Research on|State-of-the-Art)/i)) {
        if (!seenTitles.has(cleanLine)) {
          seenTitles.add(cleanLine);
          return cleanLine;
        }
      }
    }

    // Look for title in the research topic
    const researchTopicMatch = content.match(/Research report for[:\s]+(.*?)(?:\n|$)/i);
    if (researchTopicMatch && researchTopicMatch[1]) {
      const title = researchTopicMatch[1].trim();
      if (!seenTitles.has(title)) {
        seenTitles.add(title);
        return title;
      }
    }

    return 'Research Report';
  };
  const reportContainerRef = useRef(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [pdfGenerating, setPdfGenerating] = useState(false);
  const [wordGenerating, setWordGenerating] = useState(false);
  const [docxGenerating, setDocxGenerating] = useState(false);
  const [codeBlockStates, setCodeBlockStates] = useState({});
  const [citationCache, setCitationCache] = useState({});
  const [processCount, setProcessCount] = useState(0);

  // Function to copy report content to clipboard
  const copyToClipboard = () => {
    // Get text content without HTML tags
    const textContent = reportContent || '';

    // Use Clipboard API
    navigator.clipboard.writeText(textContent)
      .then(() => {
        // Show success message
        setCopySuccess(true);

        // Process citations immediately
        processCitations();

        // Set a single timer to hide the success message and ensure citations are processed
        const timer = setTimeout(() => {
          processCitations();
          setCopySuccess(false);

          // Force a small state update to trigger React to re-render
          // This ensures the citations remain active
          setProcessCount(count => count + 1);
        }, 2000);

        return () => clearTimeout(timer);
      })
      .catch(err => {
        console.error('Failed to copy: ', err);
        setCopySuccess(false);
      });
  };

  // Function to generate and download PDF of the report
  const downloadPDF = () => {
    if (!reportContainerRef.current || !reportContent) return;

    // Set generating state
    setPdfGenerating(true);

    // Clone the report container to modify it for PDF
    const reportClone = reportContainerRef.current.cloneNode(true);

    // Add simplified citation styles for the PDF
    const citationStyles = document.createElement('style');
    citationStyles.textContent = `
      /* Simple PDF-friendly citation styles */
      .citation-number {
        display: inline-flex !important;
        background-color: transparent !important;
        color: #1a5fb4 !important;
        margin: 0 2px !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        box-sizing: border-box !important;
      }
      
      /* Use padding to push content up rather than down */
      .citation-number > span {
        display: block !important;
        padding: 0 !important;
        margin-top: -2px !important; /* Move slightly up */
      }
    `;
    reportClone.appendChild(citationStyles);

    // Add special PDF-specific styling to the clone
    const pdfStyle = document.createElement('style');
    pdfStyle.textContent = `
      /* PDF-specific styles */
      body, div, p, li, h1, h2, h3, h4, h5, h6 {
        text-align: left !important;
        text-justify: none !important;
        line-height: 1.5 !important;
        letter-spacing: normal !important;
        word-spacing: normal !important;
        font-kerning: normal !important;
        font-feature-settings: normal !important;
        text-rendering: geometricPrecision !important;
        -webkit-font-smoothing: antialiased !important;
      }
      
      /* Create consistent paragraph styling */
      p, li, td, th, div, span {
        transform: rotate(0deg) !important; /* Prevent slanting */
        text-orientation: mixed !important;
        writing-mode: horizontal-tb !important;
        font-stretch: normal !important;
      }
      
      p {
        margin-bottom: 10px !important;
        orphans: 3 !important;
        widows: 3 !important;
        text-rendering: optimizeLegibility !important;
      }
      
      table { page-break-inside: avoid !important; display: table !important; width: 100% !important; }
      tr { page-break-inside: avoid !important; page-break-after: auto !important; }
      td, th { page-break-inside: avoid !important; word-break: break-word !important; text-align: left !important; }
      thead { display: table-header-group !important; }
      tfoot { display: table-footer-group !important; }
      img, svg { max-width: 100% !important; }
      h1, h2, h3, h4, h5 { page-break-after: avoid !important; }
      pre { white-space: pre-wrap !important; page-break-inside: avoid !important; }
      .code-block-container { page-break-inside: avoid !important; }
      ul, ol { page-break-inside: avoid !important; }
      li { margin-bottom: 5px !important; text-align: left !important; }
      
      /* Ensure proper content alignment */
      .py-6 {
        text-align: left !important;
        max-width: 100% !important;
      }
      
      /* Citation styles are applied directly */
      .citation-number {
        display: inline-flex !important;
        background-color: transparent !important;
        color: #1a5fb4 !important;
        margin: 0 2px !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        box-sizing: border-box !important;
      }
      
      /* Use margin to push content up rather than down */
      .citation-number > span {
        display: block !important;
        padding: 0 !important;
        margin-top: -2px !important; /* Move slightly up */
      }
    `;
    reportClone.appendChild(pdfStyle);

    // Process citations in the clone to ensure they appear in the PDF
    const citations = reportClone.querySelectorAll('.citation-number');
    citations.forEach(citation => {
      // Remove tooltips from the PDF version
      const tooltip = citation.querySelector('.citation-tooltip');
      if (tooltip) {
        tooltip.remove();
      }

      // Get the citation number
      const numText = citation.innerText.trim();

      // Create a modified citation with adjusted spacing but no background
      // Apply extensive text alignment fixes to prevent slanting
      citation.style.display = 'inline-block';
      citation.style.backgroundColor = 'transparent';
      citation.style.color = '#1a5fb4';
      citation.style.margin = '0 2px';
      citation.style.textAlign = 'center';
      citation.style.boxSizing = 'border-box';
      citation.style.fontKerning = 'normal';
      citation.style.fontFeatureSettings = 'normal';
      citation.style.textRendering = 'geometricPrecision';
      citation.style.fontStretch = 'normal';
      citation.style.transform = 'rotate(0deg)';
      citation.style.textOrientation = 'mixed';
      citation.style.writingMode = 'horizontal-tb';

      // Keep text centered but remove dimensions to allow natural sizing
      citation.style.display = 'inline-flex';
      citation.style.justifyContent = 'center';
      citation.style.alignItems = 'center';

      // Move the text up slightly
      citation.innerHTML = `<span style="display:block; padding:0; margin-top:-2px;">${numText}</span>`;
    });

    // Find all tables and ensure they have proper widths for PDF
    const tables = reportClone.querySelectorAll('table');
    tables.forEach(table => {
      table.style.width = '100%';
      table.style.tableLayout = 'fixed';
      const cells = table.querySelectorAll('td, th');
      cells.forEach(cell => {
        cell.style.wordBreak = 'break-word';
        cell.style.maxWidth = '100%';
      });
    });

    // Apply additional fixes to ensure text is properly aligned
    const paragraphs = reportClone.querySelectorAll('p, li, h1, h2, h3, h4, h5, h6');
    paragraphs.forEach(p => {
      p.style.textAlign = 'left';
      p.style.textJustify = 'none';
      p.style.lineHeight = '1.5';
      p.style.letterSpacing = 'normal';
      p.style.wordSpacing = 'normal';
      p.style.fontKerning = 'normal';
      p.style.textRendering = 'optimizeLegibility';
    });

    // Configure PDF options with better text rendering
    const options = {
      margin: [20, 15, 20, 15], // Top, right, bottom, left (increased top/bottom margins)
      filename: 'research-report.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        letterRendering: true,
        scrollY: 0,
        windowWidth: 1200, // Force a consistent width
        allowTaint: true,
        fontFix: true,
        removeContainer: true
      },
      jsPDF: {
        unit: 'mm',
        format: 'a4',
        orientation: 'portrait',
        compress: true,
        precision: 16,
        hotfixes: ['px_scaling', 'px_lineheight']
      },
      pagebreak: { mode: ['avoid-all', 'css', 'legacy'] },
      enableLinks: false
    };

    // Generate and download PDF
    html2pdf()
      .from(reportClone)
      .set(options)
      .toPdf() // Convert to PDF
      .output('save', 'research-report.pdf') // Save with better compression
      .then(() => {
        // Reset generating state
        setPdfGenerating(false);

        // Process citations to ensure they remain active
        processCitations();
        setProcessCount(count => count + 1);
      })
      .catch(err => {
        console.error('Failed to generate PDF:', err);
        setPdfGenerating(false);
      });
  };

  // Function to generate and download an HTML file (compatible with Word/Google Docs)
  const downloadWordDoc = () => {
    if (!reportContainerRef.current || !reportContent) return;

    setWordGenerating(true);

    // Clone the report container to get the rendered HTML
    const reportClone = reportContainerRef.current.cloneNode(true);

    // Remove interactive elements and styles not suitable for Word
    // Remove buttons (copy, pdf, word)
    const buttonContainer = reportClone.querySelector('.copy-button-container');
    if (buttonContainer) buttonContainer.remove();

    // Remove code block toolbars
    const toolbars = reportClone.querySelectorAll('.code-block-toolbar');
    toolbars.forEach(toolbar => toolbar.remove());

    // Remove citation tooltips
    const tooltips = reportClone.querySelectorAll('.citation-tooltip');
    tooltips.forEach(tooltip => tooltip.remove());

    // Simplify citations to plain text numbers
    const citations = reportClone.querySelectorAll('.citation-number');
    citations.forEach(citation => {
      const numText = citation.innerText.trim();
      citation.replaceWith(`[${numText}]`);
    });

    // Process images to ensure proper sizing and containers
    const images = reportClone.querySelectorAll('img');
    images.forEach(img => {
      // Skip if image is already in a proper container
      if (img.parentElement.tagName === 'FIGURE' ||
        img.parentElement.classList.contains('image-container') ||
        img.parentElement.classList.contains('chart-container') ||
        img.parentElement.classList.contains('graph-container')) {
        return;
      }

      // Check if it's a chart or graph based on image attributes or src
      const isChart = img.src.includes('chart') || img.src.includes('graph') ||
        img.alt.includes('chart') || img.alt.includes('graph') ||
        img.width > 500;

      // Create a container for the image
      const container = document.createElement('div');
      container.className = isChart ? 'chart-container' : 'image-container';

      // Replace the image with the container + image
      img.parentNode.insertBefore(container, img);
      container.appendChild(img);

      // Ensure image has alt text
      if (!img.alt) img.alt = 'Report image';
    });

    // Extract the report title
    const reportTitle = extractReportTitle(reportContent);

    // Add a title at the top of the report if it doesn't already have one
    const firstHeading = reportClone.querySelector('h1');
    if (!firstHeading || !firstHeading.textContent.includes(reportTitle)) {
      const titleElement = document.createElement('h1');
      titleElement.className = 'report-title';
      titleElement.textContent = reportTitle;

      if (reportClone.firstChild) {
        reportClone.insertBefore(titleElement, reportClone.firstChild);
      } else {
        reportClone.appendChild(titleElement);
      }
    }

    // Get the inner HTML
    let htmlContent = reportClone.innerHTML;

    // Basic HTML structure and minimal styling for Word import
    const finalHtml = `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>${reportTitle}</title>
        <style>
          body { font-family: 'Roboto', sans-serif; line-height: 1.4; max-width: 800px; margin: 20px auto; }
          .report-title { font-size: 1.8rem; font-weight: 700; margin-top: 1rem; margin-bottom: 1.5rem; text-align: center; }
          h1, h2, h3, h4 { margin-top: 1.5em; margin-bottom: 0.5em; }
          p { margin-bottom: 1em; }
          ul, ol { padding-left: 2em; margin-bottom: 1em; }
          li { margin-bottom: 0.5em; }
          pre { background-color: #f5f5f5; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }
          code { font-family: monospace; }
          table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          blockquote { border-left: 4px solid #ccc; padding-left: 1em; margin-left: 0; font-style: italic; color: #555; }
          
          /* Image styling */
          img { max-width: 100%; height: auto; display: block; margin: 1em auto; }
          .image-container { text-align: center; margin: 1.5em 0; }
          figure { margin: 1.5em 0; }
          figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
          figcaption { text-align: center; font-size: 0.9em; color: #555; margin-top: 0.5em; }
          
          /* Chart and graph specific styling */
          .chart-container img, .graph-container img { max-width: 650px; width: 100%; }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
      </head>
      <body>
        ${htmlContent}
      </body>
      </html>
    `;

    // Create a Blob and trigger download
    const blob = new Blob([finalHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'research-report.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setWordGenerating(false);
  };

  // Function to generate and download a .docx file
  const downloadDocx = async () => {
    if (!reportContent) return;

    setDocxGenerating(true);

    try {
      // Basic text extraction (for simplicity, formatting is lost)
      const textContent = reportContent || '';

      // Split content into paragraphs based on newlines
      const paragraphs = textContent.split('\n').map(text =>
        new Paragraph({
          children: [new TextRun(text)]
        })
      );

      // Create a new Document
      const doc = new Document({
        sections: [{
          properties: {},
          children: paragraphs,
        }],
      });

      // Generate the blob
      const blob = await Packer.toBlob(doc);

      // Trigger download
      saveAs(blob, 'research-report.docx');

    } catch (error) {
      console.error('Failed to generate .docx file:', error);
    } finally {
      setDocxGenerating(false);
    }
  };

  // Function to copy code block content
  const copyCodeBlock = (code, id) => {
    navigator.clipboard.writeText(code)
      .then(() => {
        // Update the specific code block's copy state
        setCodeBlockStates(prev => ({
          ...prev,
          [id]: {
            ...prev[id],
            copied: true
          }
        }));

        // Reset after 2 seconds
        setTimeout(() => {
          setCodeBlockStates(prev => ({
            ...prev,
            [id]: {
              ...prev[id],
              copied: false
            }
          }));
        }, 2000);
      })
      .catch(err => {
        console.error('Failed to copy code:', err);
      });
  };

  // Toggle code block collapse
  const toggleCollapse = (id) => {
    setCodeBlockStates(prev => ({
      ...prev,
      [id]: {
        ...prev[id],
        collapsed: !prev[id]?.collapsed
      }
    }));
  };

  // Toggle code block wrap
  const toggleWrap = (id) => {
    setCodeBlockStates(prev => ({
      ...prev,
      [id]: {
        ...prev[id],
        wrap: !prev[id]?.wrap
      }
    }));
  };

  // Toggle code block theme
  const toggleTheme = (id) => {
    setCodeBlockStates(prev => ({
      ...prev,
      [id]: {
        ...prev[id],
        darkTheme: !prev[id]?.darkTheme
      }
    }));
  };

  // Toggle all code blocks with the same property
  const toggleAllCodeBlocks = (property) => {
    // Determine the target state based on the majority of current states
    let targetState = true;
    let stateCount = { true: 0, false: 0 };

    // Count current states to determine majority
    Object.values(codeBlockStates).forEach(state => {
      if (property === 'theme') {
        stateCount[state.darkTheme ? 'true' : 'false']++;
      } else if (property === 'wrap') {
        stateCount[state.wrap ? 'true' : 'false']++;
      } else if (property === 'collapse') {
        stateCount[state.collapsed ? 'true' : 'false']++;
      }
    });

    // Set target state to the opposite of the majority
    if (Object.keys(codeBlockStates).length > 0) {
      if (property === 'theme') {
        targetState = stateCount.true <= stateCount.false;
      } else {
        targetState = stateCount.true <= stateCount.false;
      }
    }

    // Update all code blocks
    const newStates = {};
    Object.keys(codeBlockStates).forEach(id => {
      newStates[id] = { ...codeBlockStates[id] };

      if (property === 'theme') {
        newStates[id].darkTheme = targetState;
      } else if (property === 'wrap') {
        newStates[id].wrap = targetState;
      } else if (property === 'collapse') {
        newStates[id].collapsed = targetState;
      }
    });

    setCodeBlockStates(newStates);
  };

  // Helper function to extract the full citation text from the references section
  const extractFullCitationText = (citationNumber, referencesSection) => {
    try {
      // Try different patterns to match the full citation line
      const patterns = [
        // [1] Full citation text here
        new RegExp(`\\[${citationNumber}\\]\\s*(.+)$`, 'gm'),
        // 1. Full citation text here
        new RegExp(`^${citationNumber}\.\s*(.+)$`, 'gm')
      ];

      for (const pattern of patterns) {
        const matches = [...referencesSection.matchAll(pattern)];
        if (matches.length > 0) {
          return matches[0][1].trim();
        }
      }

      return '';
    } catch (e) {
      console.error('Error extracting full citation text:', e);
      return '';
    }
  };

  // Function to extract reference URL and description from a citation number
  const getReferenceLinkFromCitation = (citationNumber, content) => {
    try {
      console.log(`Processing citation ${citationNumber}`);

      // Cache the original content for extracting references
      const contentToSearch = content || '';

      // Try a variety of reference section patterns
      const referencesSectionPatterns = [
        // Standard "References" section 
        /(?:References|REFERENCES|References:)(?:\s*\n+)([\s\S]*?)(?:$|(?:\n+\s*(?:#|\*\*\*|--)))/,
        // References at the end of the document
        /(?:References|REFERENCES|References:)(?:\s*\n+)([\s\S]*?)$/,
        // Citation list without a header
        /(\[\d+\].*?\n(?:\[\d+\].*?\n)+)$/,
        // Any list of [number] style references
        /((?:\[\d+\].*?(?:\n|$))+)/,
        // References as numbered list
        /((?:\d+\.\s+.*?(?:\n|$))+)/
      ];

      let referencesSection = '';
      for (const pattern of referencesSectionPatterns) {
        const matches = contentToSearch.match(pattern);
        if (matches && matches[1]) {
          referencesSection = matches[1];
          break;
        }
      }

      // If we found a references section, search it for the citation
      if (referencesSection) {
        // Log the references section to debug
        console.log('Found references section:', referencesSection.substring(0, 200) + '...');

        // Try different citation patterns to handle various formats
        const patterns = [
          // [1] Title : URL
          new RegExp(`\\[${citationNumber}\\]\\s*(.*?)\\s*:\\s*(https?://\\S+)`, 'i'),
          // [1] Title. URL
          new RegExp(`\\[${citationNumber}\\]\\s*(.*?)\\. (https?://\\S+)`, 'i'),
          // [1] URL - Title
          new RegExp(`\\[${citationNumber}\\]\\s*(https?://\\S+)\\s*-\\s*(.*?)(?:\\n|$)`, 'i'),
          // [1] URL
          new RegExp(`\\[${citationNumber}\\]\\s*(https?://\\S+)`, 'i'),
          // Any URL in a line with the citation number
          new RegExp(`\\[${citationNumber}\\].*?(https?://\\S+)`, 'i'),
          // Number. Title: URL format
          new RegExp(`^${citationNumber}\.\s*(.*?)\\s*:\\s*(https?://\\S+)`, 'im'),
          // Number. URL format
          new RegExp(`^${citationNumber}\.\s*(https?://\\S+)`, 'im')
        ];

        for (const pattern of patterns) {
          const citationMatch = referencesSection.match(pattern);
          if (citationMatch) {
            console.log(`Found match with pattern:`, pattern, citationMatch);

            // If the first capture group is a URL and the second is text (3rd pattern)
            if (citationMatch[1] && citationMatch[1].match(/^https?:\/\//i) && citationMatch[2]) {
              return {
                url: citationMatch[1],
                title: citationMatch[2]
              };
            }
            // If only one match group and it's a URL
            else if (citationMatch[1] && citationMatch[1].match(/^https?:\/\//i) && !citationMatch[2]) {
              return {
                url: citationMatch[1],
                title: `Citation ${citationNumber}`
              };
            }
            // Standard pattern with title first, URL second
            else if (citationMatch[2]) {
              return {
                url: citationMatch[2],
                title: citationMatch[1] || `Citation ${citationNumber}`
              };
            }
          }
        }

        // Look for a line that contains the citation
        const linePatterns = [
          // [1] Some text...
          new RegExp(`\\[${citationNumber}\\]\\s*(.+)$`, 'gm'),
          // 1. Some text...
          new RegExp(`^\\s*${citationNumber}\.\\s*(.+)$`, 'gm')
        ];

        for (const linePattern of linePatterns) {
          const lineMatches = [...referencesSection.matchAll(linePattern)];
          if (lineMatches.length > 0) {
            const line = lineMatches[0][1];
            // Extract any URL from the line
            const urlMatch = line.match(/(https?:\/\/[^\s)\]"']+)/i);
            if (urlMatch) {
              console.log(`Found URL in citation line:`, urlMatch[1]);
              // Get better title by removing the URL and common suffixes
              let title = line.replace(urlMatch[1], '').trim();
              // Clean up the title by removing trailing punctuation and "URL" text
              title = title.replace(/(?::|\.|,)\s*$/, '').trim();
              title = title.replace(/\s*URL\s*$/i, '').trim();

              // Try to extract the full description/citation text from the references section
              const fullDescription = extractFullCitationText(citationNumber, referencesSection);

              return {
                url: urlMatch[1],
                title: title,
                description: fullDescription || ''
              };
            }
          }
        }
      }

      // Search the entire document for any explicit mention of the citation with a URL
      console.log(`Searching entire document for citation ${citationNumber}`);

      // Look for citation followed by URL on the same line
      const citationUrlPattern = new RegExp(`\\[${citationNumber}\\][^\n]*?(https?:\/\/[^\s)\]"']+)`, 'i');
      const citationUrlMatch = contentToSearch.match(citationUrlPattern);
      if (citationUrlMatch) {
        console.log(`Found URL near citation:`, citationUrlMatch[1]);
        return {
          url: citationUrlMatch[1],
          title: `Citation ${citationNumber}`
        };
      }

      // Find paragraphs mentioning the citation
      const citationMentionPattern = new RegExp(`\\[${citationNumber}\\]`, 'g');
      const paragraphs = contentToSearch.split(/\n\s*\n/);

      for (const paragraph of paragraphs) {
        if (paragraph.match(citationMentionPattern)) {
          // Extract any URLs in the paragraph
          const urlMatches = paragraph.match(/(https?:\/\/[^\s)\]"']+)/ig);
          if (urlMatches && urlMatches.length > 0) {
            console.log(`Found URL in paragraph with citation:`, urlMatches[0]);
            return {
              url: urlMatches[0],
              title: `Citation ${citationNumber}`
            };
          }
        }
      }

      // Look for URLs anywhere in the document as a last resort
      const allUrls = contentToSearch.match(/(https?:\/\/[^\s)\]"']+)/ig);
      if (allUrls && allUrls.length > parseInt(citationNumber) - 1) {
        // If citation number is within range of found URLs, use that
        const url = allUrls[parseInt(citationNumber) - 1];
        console.log(`Using URL by position:`, url);
        return {
          url: url,
          title: `Citation ${citationNumber}`
        };
      } else if (allUrls && allUrls.length > 0) {
        // Just use the first URL
        console.log(`Falling back to first URL found:`, allUrls[0]);
        return {
          url: allUrls[0],
          title: `Citation ${citationNumber}`
        };
      }

      // Generic fallback
      console.log(`No URL found for citation ${citationNumber}`);
      return {
        url: "#",
        title: `Citation ${citationNumber}`,
        description: ''
      };
    } catch (error) {
      console.error("Error parsing citation:", error);
      return {
        url: "#",
        title: `Citation ${citationNumber}`,
        description: ''
      };
    }
  };

  // Function to get reference link, with caching
  const getCachedReferenceLink = (citationNumber, content) => {
    // If we already have this citation in cache, return it
    if (citationCache[citationNumber]) {
      return citationCache[citationNumber];
    }

    // Otherwise, extract the reference and cache it
    const reference = getReferenceLinkFromCitation(citationNumber, content);
    setCitationCache(prev => ({
      ...prev,
      [citationNumber]: reference
    }));

    return reference;
  };

  // Post-processing function to add citation functionality
  const processCitations = () => {
    if (!reportContainerRef.current) return;

    // Get all paragraph elements, excluding code blocks
    const paragraphs = reportContainerRef.current.querySelectorAll('p, li, h1, h2, h3, h4, h5, h6, td, th');

    paragraphs.forEach(paragraph => {
      // Skip processing if in References section or inside a code block
      if (paragraph.closest('.references-section') || paragraph.closest('pre') || paragraph.closest('code')) return;

      // Get the text content
      const html = paragraph.innerHTML;

      // Preprocess the HTML to handle standalone citation numbers next to bracketed citations
      // This catches patterns like "[1] 2" where both 1 and 2 are citations
      let processedHtml = html.replace(/(\[\d+\])(\s+)(\d+)(\s|\.|<|$)/g, (match, bracketCitation, space, plainNum, end) => {
        // Only process if it looks like a citation number (not a year or large number)
        const n = parseInt(plainNum, 10);
        if ((n >= 1900 && n <= 2100) || n > 200) return match;

        // Wrap the plain number in brackets to make it a proper citation
        return `${bracketCitation}${space}[${plainNum}]${end}`;
      });

      // Replace citation patterns with interactive elements WITHOUT adding extra spaces
      // This improved pattern handles citations at the end of text nodes with potential spaces
      const newHtml = processedHtml.replace(/\[(\d+(?:[,\s]+\d+)*)\](\s*\.|\.?\s*$|\s*<)/g, (match, citationNumbers, endPart) => {
        // Use a single citation number (no splitting needed)
        const numbers = [citationNumbers];

        // Filter out years and large numbers
        const validNumbers = numbers.filter(num => {
          const n = parseInt(num, 10);
          return !(n >= 1900 && n <= 2100) && n <= 200;
        });

        if (validNumbers.length === 0) return match;

        // Create inline citations with no extra spaces between them
        return validNumbers.map(num => {
          const reference = getCachedReferenceLink(num, reportContent);
          const hasUrl = reference?.url && reference.url !== '#';

          return `<span class="citation-number ${hasUrl ? 'has-url' : 'no-url'}" data-citation="${num}" data-title="${reference?.title || `Citation ${num}`}" data-url="${hasUrl ? reference.url : ''}">${num}</span>`;
        }).join('') + (endPart || '');
      });

      // Process any remaining citations in brackets anywhere in the text
      let finalHtml = newHtml.replace(/\[(\d+)\]/g, (match, citationNumber) => {
        // Handle individual citation numbers
        const num = citationNumber.trim();

        // Filter out years and large numbers
        const n = parseInt(num, 10);
        if ((n >= 1900 && n <= 2100) || n > 200) return match;

        // Create inline citation
        const reference = getCachedReferenceLink(num, reportContent);
        const hasUrl = reference?.url && reference.url !== '#';

        // Process the title to remove URL suffix if present
        let title = reference?.title || `Citation ${num}`;
        title = title.replace(/\s*URL\s*$/i, '').trim();

        return `<span class="citation-number ${hasUrl ? 'has-url' : 'no-url'}" data-citation="${num}" data-title="${title}" data-description="${reference?.description || ''}" data-url="${hasUrl ? reference.url : ''}">${num}</span>`;
      });

      // Only update if changes were made
      if (finalHtml !== html) {
        paragraph.innerHTML = finalHtml;
      }
    });

    // Add event listeners to citation numbers
    const citations = reportContainerRef.current.querySelectorAll('.citation-number');
    citations.forEach(citation => {
      // Remove any existing event listeners
      const clone = citation.cloneNode(true);
      citation.parentNode.replaceChild(clone, citation);

      // Add new event listener
      clone.addEventListener('click', (e) => {
        const url = clone.getAttribute('data-url');
        if (url && url !== '#') {
          window.open(url, '_blank');
        }
        e.stopPropagation();
      });

      // Add tooltip
      const tooltip = document.createElement('div');
      tooltip.className = 'citation-tooltip';

      // Extract domain from URL
      let domain = '';
      const url = clone.getAttribute('data-url');
      if (url && url !== '#') {
        try {
          const urlObj = new URL(url);
          domain = urlObj.hostname.replace('www.', '');
        } catch (e) {
          // If URL parsing fails, just use the URL as is
          domain = url;
        }
      }

      // Clean up the citation title before displaying it
      let citationTitle = clone.getAttribute('data-title') || '';

      // Remove common title suffixes like "URL" or ":"
      citationTitle = citationTitle.replace(/\s*URL\s*$/i, '').trim();
      citationTitle = citationTitle.replace(/\s*:\s*$/i, '').trim();

      // If the title is just "Citation X", try to extract a better title from the URL
      if (/^Citation\s+\d+$/i.test(citationTitle) && url && url !== '#') {
        try {
          // Try to extract meaningful title from URL path
          const urlObj = new URL(url);
          const pathSegments = urlObj.pathname.split('/').filter(s => s.length > 0);

          // Get the last meaningful path segment
          if (pathSegments.length > 0) {
            const lastSegment = pathSegments[pathSegments.length - 1];
            // Convert kebab/snake case to readable title
            const formattedTitle = lastSegment
              .replace(/[-_]/g, ' ')           // Replace dashes and underscores with spaces
              .replace(/\.(html|php|asp)$/i, '') // Remove file extensions
              .split(' ')
              .map(word => word.length > 0 ? word[0].toUpperCase() + word.substring(1) : '')
              .join(' ');

            if (formattedTitle.length > 5) {
              citationTitle = formattedTitle;
            }
          }

          // If still generic, use domain name as fallback for title
          if (/^Citation\s+\d+$/i.test(citationTitle)) {
            citationTitle = `Article from ${domain}`;
          }
        } catch (e) {
          console.log('Error parsing URL for title extraction:', e);
        }
      }

      // Get the citation description if available
      const description = clone.getAttribute('data-description') || '';

      tooltip.innerHTML = `
        <div class="citation-tooltip-header">
          ${url && url !== '#' ?
          `<div class="citation-tooltip-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm7.5-6.923c-.67.204-1.335.82-1.887 1.855A7.97 7.97 0 0 0 5.145 4H7.5V1.077zM4.09 4a9.267 9.267 0 0 1 .64-1.539 6.7 6.7 0 0 1 .597-.933A7.025 7.025 0 0 0 2.255 4H4.09zm-.582 3.5c.03-.877.138-1.718.312-2.5H1.674a6.958 6.958 0 0 0-.656 2.5h2.49zM4.847 5a12.5 12.5 0 0 0-.338 2.5H7.5V5H4.847zM8.5 5v2.5h2.99a12.495 12.495 0 0 0-.337-2.5H8.5zM4.51 8.5a12.5 12.5 0 0 0 .337 2.5H7.5V8.5H4.51zm3.99 0V11h2.653c.187-.765.306-1.608.338-2.5H8.5zM5.145 12c.138.386.295.744.468 1.068.552 1.035 1.218 1.65 1.887 1.855V12H5.145zm.182 2.472a6.696 6.696 0 0 1-.597-.933A9.268 9.268 0 0 1 4.09 12H2.255a7.024 7.024 0 0 0 3.072 2.472zM3.82 11a13.652 13.652 0 0 1-.312-2.5h-2.49c.062.89.291 1.733.656 2.5H3.82zm6.853 3.472A7.024 7.024 0 0 0 13.745 12H11.91a9.27 9.27 0 0 1-.64 1.539 6.688 6.688 0 0 1-.597.933zM8.5 12v2.923c.67-.204 1.335-.82 1.887-1.855.173-.324.33-.682.468-1.068H8.5zm3.68-1h2.146c.365-.767.594-1.61.656-2.5h-2.49a13.65 13.65 0 0 1-.312 2.5zm2.802-3.5a6.959 6.959 0 0 0-.656-2.5H12.18c.174.782.282 1.623.312 2.5h2.49zM11.27 2.461c.247.464.462.98.64 1.539h1.835a7.024 7.024 0 0 0-3.072-2.472c.218.284.418.598.597.933zM10.855 4a7.966 7.966 0 0 0-.468-1.068C9.835 1.897 9.17 1.282 8.5 1.077V4h2.355z"/>
              </svg>
            </div>` : ''}
          <div class="citation-tooltip-domain">${domain}</div>
        </div>
        <div class="citation-tooltip-title">${citationTitle}</div>
        ${description ? `<div class="citation-tooltip-description">${description}</div>` : ''}
        ${url && url !== '#' ?
          `<div class="citation-tooltip-url">${url}</div>` :
          '<div class="citation-tooltip-hint">No URL available</div>'}
      `;
      clone.appendChild(tooltip);
    });
  };

  // Apply citation processing after render
  useEffect(() => {
    // Initialize citation cache on first load
    if (reportContent && Object.keys(citationCache).length === 0) {
      // Pre-process citations 1-20 to populate cache
      for (let i = 1; i <= 20; i++) {
        getCachedReferenceLink(i.toString(), reportContent);
      }
    }

    // Reset process count when content changes
    setProcessCount(0);

    // Initial processing
    const timer1 = setTimeout(() => {
      if (reportContainerRef.current) {
        processCitations();
        setProcessCount(prev => prev + 1);
      }
    }, 100);

    // Cleanup function
    return () => {
      clearTimeout(timer1);
    };
  }, [reportContent, codeBlockStates]); // Dependencies don't include isFocusedView to avoid reprocessing

  // Separate effect for fullscreen toggle
  useEffect(() => {
    // Only run if reportContent exists and not on first render
    if (reportContent && processCount > 0) {
      const timer = setTimeout(() => {
        if (reportContainerRef.current) {
          processCitations();
        }
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [isFocusedView, processCount]);

  // Add styles
  useEffect(() => {
    // Add the required CSS
    const style = document.createElement('style');
    style.innerHTML = `
      /* Custom Tippy Styles */
      .tippy-box {
        background-color: #333;
        color: white;
        border-radius: 4px;
        font-size: 12px;
        font-weight: normal;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
      
      .tippy-arrow {
        color: #333;
      }
      
      /* Copy button styles */
      .copy-button-container {
        position: fixed;
        top: 1rem;
        right: 3rem; /* Adjust distance to accommodate the new button */
        z-index: 20;
        display: flex;
        align-items: center; /* Vertically center items in the container */
        gap: 6px; /* Reduced gap for closer spacing */
      }
      
      .copy-button, .pdf-button, .word-button, .docx-button { /* Added .docx-button */
        background: transparent;
        border: none;
        color: #6B7280;
        cursor: pointer;
        width: 2.5rem;
        height: 2.5rem;
        display: flex; /* Use flex to center icon inside button */
        align-items: center;
        justify-content: center;
        border-radius: 0.25rem;
        transition: all 0.2s ease;
        padding: 0;
      }
      
      .copy-button:disabled, .pdf-button:disabled, .word-button:disabled, .docx-button:disabled { /* Added .docx-button */
        opacity: 0.5;
        cursor: not-allowed;
      }
      
      .copy-button:hover {
        background-color: rgba(0, 0, 0, 0.05);
        color: #374151;
      }
      
      .pdf-button.generating {
        color: #3b82f6;
      }
      
      .word-button.generating { /* Added style for word generating */
        color: #1d4ed8; /* Different color for distinction */
      }
      
      .docx-button.generating { /* Added style for docx generating */
        color: #1e40af; /* Another distinct color */
      }
      
      /* PDF icon container for the loading indicator */
      .pdf-icon-container { /* Renamed slightly for clarity */
        display: flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
      }
      
      /* Word icon container */
      .word-icon-container { /* Added for word icon */
        display: flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
      }
      
      /* Docx icon container */
      .docx-icon-container { /* Added for docx icon */
        display: flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
      }
      
      /* PDF Generation loading overlay */
      .pdf-loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        backdrop-filter: blur(3px);
      }
      
      /* Spinner animation for PDF generation */
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      
      .spinner {
        animation: spin 1.5s linear infinite;
      }
      
      /* Icons */
      .copy-icon, .pdf-icon, .word-icon, .docx-icon { /* Added .docx-icon */
        width: 1.25rem; /* Consistent icon size */
        height: 1.25rem; /* Consistent icon size */
        display: block;
      }
      
      /* Code block toolbar styles */
      .code-block-container {
        position: relative;
        margin: 1.5rem 0;
      }
      
      .code-block-toolbar {
        display: flex;
        align-items: center;
        background-color: #f3f4f6;
        border-top-left-radius: 0.375rem;
        border-top-right-radius: 0.375rem;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        color: #6b7280;
        border-bottom: 1px solid #e5e7eb;
        justify-content: space-between;
      }
      
      .code-block-toolbar-left {
        display: flex;
        align-items: center;
      }
      
      .code-block-toolbar-right {
        display: flex;
        gap: 0.25rem;
      }
      
      .code-block-language {
        font-weight: 500;
        color: #4b5563;
        padding-right: 0.5rem;
      }
      
      .code-block-button {
        background: transparent;
        border: none;
        color: #6b7280;
        cursor: pointer;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        border-radius: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
        transition: all 0.2s ease;
      }
      
      .code-block-button:hover {
        background-color: rgba(0, 0, 0, 0.05);
        color: #374151;
      }
      
      .code-block-button.active {
        background-color: rgba(0, 0, 0, 0.1);
      }
      
      .code-block-button.success {
        color: #047857;
      }
      
      .code-block-content.collapsed {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease;
      }
      
      .code-block-content.wrap pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
      }
      
      /* Existing styles */
      .citation-group {
        display: inline-flex;
        align-items: center;
        position: relative;
        margin-right: 2px;
      }
      .citation-number {
        cursor: pointer;
        position: relative;
        font-weight: normal;
        background-color: #f1f1f1;
        border-radius: 4px;
        font-size: 0.85em;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 16px;
        height: 22px;
        margin: 0 2px;
        padding: 0 6px;
        color: #1a73e8;
        /* Ensure citations stay closer to text */
        vertical-align: middle;
        line-height: normal;
      }
      /* Special handling for list items */
      li .citation-number {
        margin-left: 2px;
        margin-right: 2px;
      }
      .citation-number.has-url {
        color: #1a73e8;
        background-color: #f1f1f1;
      }
      .citation-number.has-url:hover {
        background-color: #e8f0fe;
        color: #1967d2;
      }
      .citation-number.no-url {
        color: #93c5fd;
        background-color: #f0f0f0;
      }
      .citation-tooltip {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%) translateY(-8px);
        background-color: white;
        color: #333;
        padding: 16px;
        border-radius: 8px;
        font-size: 14px;
        width: 300px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        z-index: 9999;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.2s ease, visibility 0.2s ease;
        border: 1px solid #e0e0e0;
      }
      .citation-number:hover .citation-tooltip {
        opacity: 1;
        visibility: visible;
      }
      .citation-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 8px;
        border-style: solid;
        border-color: white transparent transparent transparent;
      }
      .citation-tooltip-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
      }
      
      .citation-tooltip-icon {
        margin-right: 8px;
        display: flex;
        align-items: center;
        color: #5f6368;
      }
      
      .citation-tooltip-domain {
        font-size: 13px;
        color: #5f6368;
        font-weight: 500;
      }
      
      .citation-tooltip-title {
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 16px;
        color: #000;
      }
      .citation-tooltip-url {
        color: #1a73e8;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        word-break: break-all;
        margin-bottom: 6px;
      }
      .citation-tooltip-hint {
        color: #5f6368;
        font-size: 12px;
        margin-top: 6px;
      }
      
      /* Enhanced Markdown Styling */
      /* Improved list styling */
      ul, ol {
        padding-left: 1.5rem;
        margin: 0.5rem 0;
      }
      
      ul ul, ol ol, ul ol, ol ul {
        margin: 0.25rem 0 0.5rem 0;
      }
      
      li {
        margin-bottom: 0.25rem;
        position: relative;
      }
      
      /* Adjustments for nested lists */
      li > ul, li > ol {
        margin-top: 0.25rem;
        margin-left: 1rem;
      }
      
      /* Custom list styling */
      ul.contains-task-list {
        list-style-type: none;
        padding-left: 0.5rem;
      }
      
      .task-list-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.5rem;
      }
      
      .task-list-item-checkbox {
        margin-right: 0.5rem;
        margin-top: 0.25rem;
      }
      
      /* Code blocks styling */
      pre {
        margin: 1rem 0;
        overflow-x: auto;
        border-radius: 0.375rem;
        font-size: 0.8rem !important;
      }
      
      code {
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        font-size: 0.8rem;
      }
      
      /* Inline code */
      :not(pre) > code {
        background-color: rgba(0, 0, 0, 0.05);
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
        font-size: 0.75em;
        color: #24292e;
      }
      
      /* Tables */
      table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        font-size: 0.9rem;
      }
      
      th, td {
        border: 1px solid #e5e7eb;
        padding: 0.5rem;
        text-align: left;
      }
      
      th {
        background-color: #f9fafb;
        font-weight: 600;
      }
      
      /* Blockquotes */
      blockquote {
        border-left: 4px solid #e5e7eb;
        padding-left: 1rem;
        color: #4b5563;
        margin: 1rem 0;
        font-style: italic;
      }
      
      /* Horizontal rule */
      hr {
        border: 0;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
      }
      
      /* Bullet point styling */
      .bullet-point {
          position: relative;
          padding-left: 1.5em;
          margin: 0.5em 0;
          line-height: 1.6;
      }
      
      .bullet-point::before {
          content: "•";
          position: absolute;
          left: 0.5em;
          color: #4a5568;
      }
      
      /* Section item styling */
      .section-item {
          margin: 1em 0;
          line-height: 1.6;
      }
      
      .section-item strong {
          color: #2d3748;
          margin-right: 0.5em;
      }
    `;
    document.head.appendChild(style);

    // Clean up
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Generate unique ID for each code block
  const getCodeBlockId = (index, language) => `code-block-${language || 'text'}-${index}`;

  // Common Tippy props for consistency
  const tippyProps = {
    animation: 'shift-away',
    duration: [50, 100], // [enter, exit] in ms - much faster than default
    delay: [0, 0], // [enter, exit] in ms - Set exit delay to 0 to prevent ghosting
    placement: 'top',
    arrow: true,
    theme: 'custom'
  };

  // Extract the report title for UI display
  const reportTitle = extractReportTitle(reportContent);

  // Track if we've rendered the title
  const [titleRendered, setTitleRendered] = useState(false);

  // Reset titleRendered when report content changes
  useEffect(() => {
    setTitleRendered(false);
  }, [reportContent]);

  // Extract the research topic from the URL if available
  const getResearchTopicFromUI = () => {
    try {
      // Look for the research topic in the UI (typically shown in the header)
      const headerElement = document.querySelector('.research-header h1');
      if (headerElement) {
        return headerElement.textContent.trim();
      }

      // Alternative: try to get it from the URL or page title
      if (typeof window !== 'undefined') {
        const urlParams = new URLSearchParams(window.location.search);
        const topic = urlParams.get('topic') || document.title.replace('Research - ', '');
        if (topic && topic.length > 3) {
          return topic;
        }
      }

      return null;
    } catch (e) {
      console.error('Error getting research topic from UI:', e);
      return null;
    }
  };

  // Use the extracted title or fall back to the research topic from UI
  const displayTitle = reportTitle !== 'Research Report' ? reportTitle : getResearchTopicFromUI();

  // Function to download report as HTML
  const downloadHtml = () => {
    try {
      // Process the content to format dates and table of contents
      let processedContent = reportContent;

      // Format dates (e.g., "May 10, 2025")
      processedContent = processedContent.replace(/([A-Z][a-z]+ \d{1,2}, \d{4})/g, '<div class="report-date">$1</div>');

      // Convert bullet points with asterisks to list items
      processedContent = processedContent.replace(/^\s*\*\s+([^\n]+)/gm, (match, content) => {
        // Check if it's a section header (contains a colon)
        if (content.includes(':')) {
          const [header, text] = content.split(':', 2);
          return `<div class="section-item"><strong>${header.trim()}:</strong>${text || ''}</div>`;
        }
        // Regular bullet point
        return `<div class="bullet-point">${content}</div>`;
      });

      // Convert markdown headers (###) to proper HTML headers
      processedContent = processedContent.replace(/###\s+([^#\n]+)/g, '<h3>$1</h3>');
      processedContent = processedContent.replace(/##\s+([^#\n]+)/g, '<h2>$1</h2>');
      processedContent = processedContent.replace(/#\s+([^#\n]+)/g, '<h1>$1</h1>');

      // Convert bold markdown (**text**) to HTML bold
      processedContent = processedContent.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

      // Convert tables with markdown-style separators
      processedContent = processedContent.replace(
        /\|\s*([^|\n]+)\s*\|\s*([^|\n]+)\s*\|\s*\n\|\s*[-\s|]+\|\s*[-\s|]+\|\s*\n((?:\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|\s*\n)+)/g,
        (match, header1, header2, rows) => {
          const headerRow = `<tr><th>${header1.trim()}</th><th>${header2.trim()}</th></tr>`;
          const bodyRows = rows
            .split('\n')
            .filter(row => row.trim())
            .map(row => {
              const [col1, col2] = row.split('|').filter(cell => cell.trim());
              return `<tr><td>${col1.trim()}</td><td>${col2.trim()}</td></tr>`;
            })
            .join('');
          return `<table class="report-table">${headerRow}${bodyRows}</table>`;
        }
      );

      // Format table of contents
      processedContent = processedContent.replace(
        /(Table of Contents\n)((?:[-\s]*[0-9.]*\s+[^\n]+\n)+)/g,
        (match, header, items) => {
          const formattedItems = items
            .split('\n')
            .filter(item => item.trim())
            .map(item => {
              const indentMatch = item.match(/^(\s*[-]*\s*)/);
              const indent = indentMatch ? indentMatch[1].length : 0;
              const cleanItem = item.replace(/^[-\s]*/, '').trim();
              return `<div class="toc-item" style="padding-left: ${indent * 8}px">${cleanItem}</div>`;
            })
            .join('\n');

          return `<div class="table-of-contents">
            <h2>${header.trim()}</h2>
            <div class="toc-items">
              ${formattedItems}
            </div>
          </div>`;
        }
      );

      // Create HTML content with enhanced styling
      const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Date styling */
        .report-date {
            font-size: 1.1em;
            color: #666;
            margin: 1em 0;
            font-weight: 500;
        }
        
        /* Table of Contents styling */
        .table-of-contents {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .table-of-contents h2 {
            margin-top: 0;
            color: #2d3748;
            font-size: 1.5em;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        .toc-items {
            font-size: 0.95em;
        }
        
        .toc-item {
            margin: 8px 0;
            color: #4a5568;
            line-height: 1.4;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #2d3748;
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }
        
        h1 { font-size: 2em; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3em; }
        h3 { font-size: 1.25em; }
        h4 { font-size: 1em; }
        
        /* Strong text */
        strong {
            font-weight: 600;
            color: #1a202c;
        }
        
        /* Tables */
        .report-table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .report-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            text-align: left;
            padding: 12px;
            border: 1px solid #e2e8f0;
        }
        
        .report-table td {
            padding: 12px;
            border: 1px solid #e2e8f0;
        }
        
        .report-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        /* Links */
        a {
            color: #3182ce;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Lists */
        ul, ol {
            padding-left: 24px;
            margin: 1em 0;
        }
        
        li {
            margin: 0.5em 0;
        }
        
        /* Paragraphs */
        p {
            margin: 1em 0;
            line-height: 1.6;
        }
        
        /* Images */
        img {
            max-width: 100%;
            height: auto;
            margin: 1em 0;
            border-radius: 4px;
        }
        
        /* Bullet point styling */
        .bullet-point {
            position: relative;
            padding-left: 1.5em;
            margin: 0.5em 0;
            line-height: 1.6;
        }
        
        .bullet-point::before {
            content: "•";
            position: absolute;
            left: 0.5em;
            color: #4a5568;
        }
        
        /* Section item styling */
        .section-item {
            margin: 1em 0;
            line-height: 1.6;
        }
        
        .section-item strong {
            color: #2d3748;
            margin-right: 0.5em;
        }
    </style>
</head>
<body>
    ${processedContent}
</body>
</html>`;

      // Create blob and download link
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'research-report.html';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading HTML:', error);
    }
  };

  return (
    <div className="min-h-full" ref={reportContainerRef}>
      {/* Report Actions toolbar */}
      <div className="sticky top-0 bg-white z-10 border-b border-gray-200 flex items-center justify-end py-2 px-4 gap-2">
        <div className="report-toolbar-wrapper flex items-center space-x-2">
          <Tippy content="Copy to clipboard" {...tippyProps}>
            <button
              onClick={copyToClipboard}
              className="report-toolbar-button flex items-center justify-center w-8 h-8 rounded hover:bg-gray-100 transition-colors"
              aria-label="Copy to clipboard"
            >
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
              </svg>
            </button>
          </Tippy>

          <Tippy content="Download as HTML" {...tippyProps}>
            <button
              onClick={downloadHtml}
              className="report-toolbar-button flex items-center justify-center w-8 h-8 rounded hover:bg-gray-100 transition-colors"
              aria-label="Download as HTML"
            >
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
              </svg>
            </button>
          </Tippy>

          <Tippy content="Download as PDF" {...tippyProps}>
            <button
              onClick={downloadPDF}
              disabled={pdfGenerating}
              className="report-toolbar-button flex items-center justify-center w-8 h-8 rounded hover:bg-gray-100 transition-colors"
              aria-label="Download as PDF"
            >
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </Tippy>

          <Tippy content="Download as DOCX" {...tippyProps}>
            <button
              onClick={downloadDocx}
              disabled={docxGenerating}
              className="report-toolbar-button flex items-center justify-center w-8 h-8 rounded hover:bg-gray-100 transition-colors"
              aria-label="Download as DOCX"
            >
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </button>
          </Tippy>
        </div>
      </div>

      {/* Report Content */}
      <div className="pt-6 pb-12 px-8">
        {isBenchmark && benchmarkData ? (
          // Render benchmark result using the enhanced card component
          <BenchmarkResultCard
            question={getResearchTopicFromUI() || 'Research Question'}
            answer={benchmarkData.answer || 'No answer provided'}
            confidence={benchmarkData.confidence}
            confidenceLevel={benchmarkData.confidenceLevel}
            evidence={benchmarkData.evidence}
            sources={benchmarkData.sources}
            expectedAnswer={benchmarkData.expectedAnswer}
            isCorrect={benchmarkData.isCorrect}
            reasoning={benchmarkData.reasoning}
            limitations={benchmarkData.limitations}
          />
        ) : isHtmlContent(reportContent) ? (
          // Render pure HTML content (like database reports) directly
          <div
            className="database-report-container"
            style={{ maxWidth: '100%', overflowX: 'auto' }}
            dangerouslySetInnerHTML={{ __html: reportContent }}
          />
        ) : (
          // Render regular markdown content for non-benchmark reports
          <>
            {/* Display the report title at the top of the UI */}
            {displayTitle && !titleRendered && (
              <h1 className="text-3xl font-bold mb-6 pb-2 border-b border-gray-200">{displayTitle}</h1>
            )}

            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[
                [rehypeRaw, { allowDangerousHtml: true }]
              ]}
              urlTransform={uri => {
                if (uri.startsWith('data:')) {
                  return uri; // Allow data URIs
                }
                return defaultUrlTransform(uri);
              }}
              components={{
                // Customize heading styles with lighter font weights and consistent sizes
                h1: ({ node, children, ...props }) => {
                  const titleText = children.toString();
                  // Skip if we've already rendered the title or if it matches displayTitle
                  if (titleRendered || (displayTitle && titleText === displayTitle)) {
                    return null;
                  }
                  // If this is the first h1 and matches our criteria, mark it as rendered
                  if (!titleRendered && titleText === displayTitle) {
                    setTitleRendered(true);
                    return null;
                  }
                  const id = slugify(titleText);
                  return <h1 id={id} className="text-2xl font-semibold mb-4 pb-2 border-b border-gray-200" {...props}>{children}</h1>;
                },
                h2: ({ node, children, ...props }) => {
                  const id = slugify(String(children));
                  return <h2 id={id} className="text-xl font-semibold mt-6 mb-3" {...props}>{children}</h2>;
                },
                h3: ({ node, children, ...props }) => {
                  const id = slugify(String(children));
                  return <h3 id={id} className="text-lg font-medium mt-5 mb-2" {...props}>{children}</h3>;
                },
                h4: ({ node, children, ...props }) => {
                  const id = slugify(String(children));
                  return <h4 id={id} className="text-base font-medium mt-4 mb-2" {...props}>{children}</h4>;
                },

                // Customize list styles
                ul: ({ node, children, ...props }) => <ul className="list-disc pl-5 my-3 space-y-1" {...props}>{children}</ul>,
                ol: ({ node, children, ...props }) => <ol className="list-decimal pl-5 my-3 space-y-1" {...props}>{children}</ol>,
                li: ({ node, children, ...props }) => {
                  // Check if this is a task list item
                  const isTaskItem =
                    typeof children[0] === 'object' &&
                    children[0]?.type === 'input' &&
                    children[0]?.props?.type === 'checkbox';

                  return (
                    <li className={`my-1 ${isTaskItem ? 'flex items-start' : ''}`} {...props}>
                      {children}
                    </li>
                  );
                },

                // Customize paragraph styles
                p: ({ node, children, ...props }) => <p className="my-2 leading-relaxed text-gray-800" {...props}>{children}</p>,

                // Customize link styles
                a: ({ node, children, ...props }) => <a className="text-blue-600 hover:underline" {...props}>{children}</a>,

                // Style tables properly
                table: ({ node, children, ...props }) => <table className="border-collapse table-auto w-full my-4 border border-gray-300" {...props}>{children}</table>,
                thead: ({ node, children, ...props }) => <thead className="bg-gray-100" {...props}>{children}</thead>,
                tbody: ({ node, children, ...props }) => <tbody {...props}>{children}</tbody>,
                tr: ({ node, children, ...props }) => <tr className="border-b border-gray-200" {...props}>{children}</tr>,
                th: ({ node, children, ...props }) => <th className="p-2 text-left font-medium border border-gray-300" {...props}>{children}</th>,
                td: ({ node, children, ...props }) => <td className="p-2 border border-gray-300" {...props}>{children}</td>,

                // Unwrap <pre> so custom code override handles block code
                pre: ({ node, children, ...props }) => <>{children}</>,

                // Style code blocks with syntax highlighting
                code: ({ node, className, children, ...props }) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const language = match ? match[1] : '';
                  const codeBlockIndex = node?.position?.start?.offset || Math.random();
                  const codeId = getCodeBlockId(codeBlockIndex, language);

                  // Get current state for this code block
                  const codeState = codeBlockStates[codeId] || {
                    collapsed: false,
                    copied: false,
                    wrap: false,
                    darkTheme: true
                  };

                  const code = String(children).replace(/\n$/, '');

                  // Initialize state for this code block if it doesn't exist
                  if (!codeBlockStates[codeId]) {
                    setCodeBlockStates(prev => ({
                      ...prev,
                      [codeId]: codeState
                    }));
                  }

                  // Render fenced code blocks when language class is present
                  if (match) {
                    return (
                      <div className="code-block-container">
                        <div className="code-block-toolbar">
                          <div className="code-block-toolbar-left">
                            {language && <span className="code-block-language">{language}</span>}
                          </div>
                          <div className="code-block-toolbar-right">
                            <Tippy content={codeState.collapsed ? "Expand code" : "Collapse code"} {...tippyProps}>
                              <button
                                className={`code-block-button ${codeState.collapsed ? 'active' : ''}`}
                                onClick={() => toggleCollapse(codeId)}
                              >
                                {codeState.collapsed ? "Expand" : "Collapse"}
                              </button>
                            </Tippy>

                            <Tippy content={codeState.wrap ? "Disable text wrapping" : "Enable text wrapping"} {...tippyProps}>
                              <button
                                className={`code-block-button ${codeState.wrap ? 'active' : ''}`}
                                onClick={() => toggleWrap(codeId)}
                              >
                                {codeState.wrap ? "Unwrap" : "Wrap"}
                              </button>
                            </Tippy>

                            <Tippy content={codeState.darkTheme ? "Switch to light theme" : "Switch to dark theme"} {...tippyProps}>
                              <button
                                className={`code-block-button ${codeState.darkTheme ? '' : 'active'}`}
                                onClick={() => toggleTheme(codeId)}
                              >
                                {codeState.darkTheme ? "Light" : "Dark"}
                              </button>
                            </Tippy>

                            <Tippy content={codeState.copied ? "Copied!" : "Copy code"} {...tippyProps}>
                              <button
                                className={`code-block-button ${codeState.copied ? 'success' : ''}`}
                                onClick={() => copyCodeBlock(code, codeId)}
                              >
                                {codeState.copied ? "Copied!" : "Copy"}
                              </button>
                            </Tippy>
                          </div>
                        </div>

                        <div className={`code-block-content ${codeState.collapsed ? 'collapsed' : ''} ${codeState.wrap ? 'wrap' : ''}`}>
                          <SyntaxHighlighter
                            style={codeState.darkTheme ? tomorrow : vs}
                            language={language || 'text'}
                            PreTag="div"
                            customStyle={{
                              margin: 0,
                              borderTopLeftRadius: 0,
                              borderTopRightRadius: 0,
                              borderBottomLeftRadius: '0.375rem',
                              borderBottomRightRadius: '0.375rem',
                              fontSize: '0.8rem',
                              lineHeight: 1.4
                            }}
                            {...props}
                          >
                            {code}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    );
                  }

                  return (
                    <code className="inline-block px-2 py-1 bg-gray-100 border border-gray-300 rounded shadow-sm font-mono text-sm text-gray-800" {...props}>
                      {children}
                    </code>
                  );
                },
                img: ({ node, ...props }) => <img {...props} />
              }}
            >
              {reportContent || ''}
            </ReactMarkdown>
          </>
        )}
      </div>

      {/* Analysis Report Section */}
      {analysisData && analysisData.analysis_required && (
        <div className="border-t-2 border-indigo-200 mt-8 pt-6 px-8 pb-8">
          {/* Analysis Header */}
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-semibold text-indigo-900">Data Analysis Report</h2>
              <p className="text-sm text-indigo-600">Analysis and visualizations generated from research data</p>
            </div>
          </div>

          {/* Analysis Summary */}
          {analysisData.analysis_output && (
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 mb-4">
              <h3 className="text-sm font-medium text-indigo-800 mb-2">Analysis Summary</h3>
              <div className="text-sm text-indigo-700 leading-relaxed whitespace-pre-wrap">
                {analysisData.analysis_output}
              </div>
            </div>
          )}

          {/* Charts list */}
          {analysisData.analysis_charts && analysisData.analysis_charts.length > 0 && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
              <h3 className="text-sm font-medium text-gray-800 mb-2">
                Generated Charts ({analysisData.analysis_charts.length})
              </h3>
              <ul className="space-y-1">
                {analysisData.analysis_charts.map((chartPath, index) => (
                  <li key={index} className="flex items-center gap-2 text-sm text-gray-600">
                    <svg className="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="truncate">{chartPath.replace(/\\/g, '/').split('/').pop()}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* View and Download Analysis Report Buttons */}
          {analysisData.analysis_report_path && (
            <div className="flex items-center gap-3 flex-wrap">
              <a
                href={`${process.env.REACT_APP_API_BASE_URL || ''}/analysis-report/${analysisData.analysis_report_path.replace(/\\/g, '/').split('/').pop()}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors shadow-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                View Analysis Report
              </a>
              <a
                href={`${process.env.REACT_APP_API_BASE_URL || ''}/analysis-report/${analysisData.analysis_report_path.replace(/\\/g, '/').split('/').pop()}`}
                download={`analysis-report-${new Date().toISOString().slice(0, 10)}.html`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 transition-colors shadow-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download Analysis Report (HTML)
              </a>
              <span className="text-xs text-gray-500">
                Interactive report with charts and visualizations
              </span>
            </div>
          )}
        </div>
      )}

      {/* Full-screen loading overlay for PDF generation */}
      {pdfGenerating && (
        <div className="pdf-loading-overlay">
          <LoadingIndicator
            type="spinner"
            size="large"
            color="#1a5fb4"
            text="Generating PDF..."
          />
        </div>
      )}
    </div>
  );
}

export default FinalReport; 