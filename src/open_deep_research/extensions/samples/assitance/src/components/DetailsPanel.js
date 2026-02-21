import React, { useState, useEffect, useMemo } from 'react';
import ReactDOM from 'react-dom/client';
import FinalReport from './FinalReport';
import { generateDetailsHtml } from '../utils/helpers';
import Tippy from '@tippyjs/react';
import CodeSnippetViewer from './CodeSnippetViewer';
import ReactMarkdown from 'react-markdown';

// Import React for Children API
const { Children } = React;

// Collapsible Section Component - Salesforce Theme
const CollapsibleSection = ({ title, isDefaultOpen = false, children, icon, colorClass, taskCount }) => {
  const [isOpen, setIsOpen] = useState(isDefaultOpen); // Respect isDefaultOpen prop

  return (
    <div className="mb-4 border border-slate-200/60 rounded-xl overflow-hidden bg-gradient-to-br from-white/80 to-slate-50/80 shadow-sm hover:shadow-xl hover:shadow-[#0176d3]/10 transition-all duration-300 backdrop-blur-sm">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between px-5 py-4 ${colorClass} hover:bg-gradient-to-r hover:from-slate-50/50 hover:to-white transition-all duration-200 group`}
      >
        <div className="flex items-center gap-3">
          <div className="transition-transform duration-200 group-hover:scale-110">{icon}</div>
          <h3 className="text-sm font-bold tracking-wide">{title}</h3>
          {taskCount !== undefined && taskCount > 0 && (
            <span className="ml-2 px-3 py-1 bg-gradient-to-r from-slate-100 to-slate-200 text-slate-700 text-xs rounded-full font-bold shadow-sm">
              {taskCount}
            </span>
          )}
        </div>
        <svg
          className={`w-4 h-4 transition-all duration-300 ${isOpen ? 'rotate-180' : ''} group-hover:scale-110`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="px-5 py-4 bg-gradient-to-br from-white/50 to-slate-50/30 border-t border-slate-200/40 backdrop-blur-sm">
          {children}
        </div>
      )}
    </div>
  );
};

// Helper function to extract report title (moved from ContentPanel.js)
const extractReportTitle = (content) => {
  if (!content) return null;

  // Check for HTML h1 title
  const h1Match = content.match(/<h1>(.*?)<\/h1>/i);
  if (h1Match && h1Match[1]) {
    const h1Title = h1Match[1].trim();
    // Skip if the title is just the search query
    if (!h1Title.toLowerCase().includes('what is') &&
      !h1Title.toLowerCase().includes('show me')) {
      return h1Title;
    }
  }

  // Try to find a title in the first few lines of the report
  const lines = content.split('\n').slice(0, 30);

  // First, look for specific title patterns that indicate a formal report title
  for (const line of lines) {
    const cleanLine = line.trim();
    // Look for titles with colons that describe frameworks, technologies, etc.
    if (cleanLine.match(/^\w+:\s+[A-Z]/i) ||
      cleanLine.match(/^[A-Z][\w\s]+:\s+[A-Z]/i)) {
      return cleanLine;
    }
  }

  // Then look for other title patterns
  for (const line of lines) {
    const cleanLine = line.trim();
    if (cleanLine.startsWith('# ')) {
      const title = cleanLine.replace(/^# /, '');
      // Skip if the title is just the search query
      if (!title.toLowerCase().includes('what is') &&
        !title.toLowerCase().includes('show me')) {
        return title;
      }
    }
    if (cleanLine.match(/^Profile of/i) ||
      cleanLine.match(/^Analysis of/i) ||
      cleanLine.match(/^Research on/i) ||
      cleanLine.match(/^State-of-the-Art/i) ||
      cleanLine.match(/^Overview of/i) ||
      cleanLine.match(/^Introduction to/i) ||
      cleanLine.match(/^Understanding/i)) {
      return cleanLine;
    }
  }

  return null;
};

// Helper function to remove "Thinking..." sections from report content
const removeThinkingSections = (content) => {
  if (!content) return content;

  // Check if content has "Thinking..." sections
  if (content.includes("Thinking...")) {
    // Find the first occurrence of a meaningful header (like "# Salesforce's Investment Thesis")
    // or the first non-thinking paragraph that appears to be part of the final report

    // Look for the first Markdown header that likely starts the actual report
    const headerMatch = content.match(/^#+\s+[^*\n]+/m);

    if (headerMatch && headerMatch.index > 0) {
      // Return content starting from the first header
      return content.substring(headerMatch.index);
    }

    // If we can't find a header, try to find where the actual report begins
    // by looking for patterns that indicate the end of thinking sections
    const sections = content.split(/\n\s*\n/); // Split by empty lines
    let startIndex = 0;

    // Find where thinking sections end and real content begins
    for (let i = 0; i < sections.length; i++) {
      const section = sections[i];
      // If the section doesn't start with "Thinking..." or "**", and appears substantive
      if (!section.includes("Thinking...") && !section.includes("**") &&
        section.length > 100 && !section.match(/^\s*\*\*/)) {
        startIndex = i;
        break;
      }
    }

    if (startIndex > 0) {
      return sections.slice(startIndex).join("\n\n");
    }
  }

  return content; // Return original if no thinking sections found
};

// Function to render web links in a structured format
const renderWebLinks = (links) => {
  console.log('[DetailsPanel - renderWebLinks] Received links:', links);
  if (!links || !Array.isArray(links) || links.length === 0) {
    return '<div class="text-gray-500 italic">No web links available</div>';
  }

  const generatedHtml = `
    <div class="space-y-3">
      ${links.map(link => {
    // Handle different link formats safely
    const url = typeof link === 'string' ? link : link.url || link.href || '';
    const title = link.title || link.name || (typeof link === 'string' ? link : url);
    const description = link.description || link.summary || link.snippet || '';

    // Safely extract domain for favicon
    let domain = 'unknown';
    let faviconUrl = '';
    try {
      if (url && url.includes('://')) {
        domain = new URL(url).hostname;
        faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
      } else if (url && url.includes('.')) {
        // Handle URLs without protocol
        domain = url.split('/')[0];
        faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
      }
    } catch (e) {
      console.warn('Error parsing URL for favicon:', e);
      faviconUrl = ''; // No favicon if URL parsing fails
    }

    // Only render if we have a valid URL
    if (!url) return '';

    return `
          <div class="border-b border-gray-100 pb-3 last:border-b-0">
            <div class="flex items-start">
              <div class="flex-shrink-0 mt-1">
                ${faviconUrl ?
        `<img src="${faviconUrl}" class="w-5 h-5" alt="${domain}" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" />
                   <svg class="w-5 h-5 text-blue-500 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                   </svg>` :
        `<svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                   </svg>`
      }
              </div>
              <div class="ml-3 flex-1">
                <a href="${url}" target="_blank" class="font-medium text-blue-700 hover:underline block">${title}</a>
                ${description ? `<div class="text-sm text-gray-600 mt-1">${description}</div>` : ''}
                <div class="text-xs text-gray-500 mt-1">${domain}</div>
              </div>
            </div>
          </div>
        `;
  }).join('')}
    </div>
  `;
  console.log('[DetailsPanel - renderWebLinks] Generated HTML for web links (first 300 chars):', generatedHtml.substring(0, 300) + '...');
  return generatedHtml;
};

// Function to render code snippets with CodeSnippetViewer component
const renderCodeSnippets = (snippets) => {
  if (!snippets || !Array.isArray(snippets) || snippets.length === 0) {
    return '<div class="text-gray-500 italic">No code snippets available</div>';
  }

  // Return a placeholder div that will be replaced by React
  return `
    <div class="space-y-4" id="code-snippets-container" data-snippets='${JSON.stringify(snippets).replace(/'/g, "&#39;")}'>
      <div class="text-gray-500 italic">Loading code snippets...</div>
    </div>
  `;
};

// Function to render images
const renderImages = (images) => {
  if (!images || !Array.isArray(images) || images.length === 0) {
    return '<div class="text-gray-500 italic">No images available</div>';
  }

  return `
    <div class="space-y-4">
      ${images.map((image, index) => {
    return `
          <div class="border border-gray-200 rounded-md overflow-hidden">
            <img src="${image.src || image.url}" alt="${image.description || 'Research image'}" class="w-full h-auto" />
            ${image.description ? `<div class="p-3 text-sm text-gray-600">${image.description}</div>` : ''}
          </div>
        `;
  }).join('')}
    </div>
  `;
};

// Enhanced function to generate more detailed HTML for various content types
const generateEnhancedDetailsHtml = (item) => {
  if (!item) return '';

  // Initialize HTML sections
  let contentHtml = '';
  let webLinksHtml = '';
  let codeSnippetsHtml = '';
  let imagesHtml = '';

  // Debug logging
  console.log('[DetailsPanel - generateEnhancedDetailsHtml] Processing item:', item);

  // Process main content
  if (item.content) {
    contentHtml = `
      <div class="mb-6">
        <h3 class="text-lg font-medium mb-3">Content</h3>
        <div class="prose max-w-none">
          ${item.content.replace(/`([^`]+)`/g, '<code class="inline-block px-2 py-1 bg-gray-100 border border-gray-300 rounded shadow-sm font-mono text-sm text-gray-800">$1</code>')}
        </div>
      </div>
    `;
  }

  // Process enriched data if available
  if (item.enrichedData) {
    console.log('Found enriched data:', item.enrichedData);
    console.log('[DetailsPanel - generateEnhancedDetailsHtml] item.enrichedData:', item.enrichedData);

    // Extract web links from various possible properties
    const extractLinks = () => {
      const allLinks = [];

      // Check common link properties
      if (item.enrichedData.sources && Array.isArray(item.enrichedData.sources)) {
        allLinks.push(...item.enrichedData.sources);
      }

      if (item.enrichedData.links && Array.isArray(item.enrichedData.links)) {
        allLinks.push(...item.enrichedData.links);
      }

      if (item.enrichedData.urls && Array.isArray(item.enrichedData.urls)) {
        allLinks.push(...item.enrichedData.urls);
      }

      // Check for domains field which may contain URLs
      if (item.enrichedData.domains && Array.isArray(item.enrichedData.domains)) {
        const domainLinks = item.enrichedData.domains.map(domain => {
          return typeof domain === 'string' ?
            { url: domain.includes('://') ? domain : `https://${domain}`, title: domain } :
            domain;
        });
        allLinks.push(...domainLinks);
      }

      // Check if we have a single source object
      if (item.enrichedData.source && typeof item.enrichedData.source === 'object') {
        allLinks.push(item.enrichedData.source);
      }

      // Check for references
      if (item.enrichedData.references && Array.isArray(item.enrichedData.references)) {
        allLinks.push(...item.enrichedData.references);
      }

      console.log('Extracted links:', allLinks);
      return allLinks;
    };

    const links = extractLinks();
    console.log('[DetailsPanel - generateEnhancedDetailsHtml] Extracted links for renderWebLinks:', links);

    // Web links
    if (links.length > 0) {
      webLinksHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Web Links</h3>
          ${renderWebLinks(links)}
        </div>
      `;
    }

    // Code snippets
    if (item.enrichedData.code_snippets && item.enrichedData.code_snippets.length > 0) {
      codeSnippetsHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Code Snippets</h3>
          ${renderCodeSnippets(item.enrichedData.code_snippets)}
        </div>
      `;
    }

    // Images
    if (item.enrichedData.images && item.enrichedData.images.length > 0) {
      imagesHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Images</h3>
          ${renderImages(item.enrichedData.images)}
        </div>
      `;
    }
  }

  // Special handling for nodeData
  if (item.nodeData) {
    console.log('Found node data:', item.nodeData);

    // Handle sources in nodeData
    if (item.nodeData.sources && Array.isArray(item.nodeData.sources) && !webLinksHtml) {
      webLinksHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Web Links</h3>
          ${renderWebLinks(item.nodeData.sources)}
        </div>
      `;
    }
  }

  // Combine all HTML sections
  return `
    <div class="details-content">
      ${contentHtml}
      ${webLinksHtml}
      ${codeSnippetsHtml}
      ${imagesHtml}
    </div>
  `;
};

function DetailsPanel({
  isVisible,
  onClose,
  selectedItem,
  showFinalReport,
  reportContent,
  analysisData,
  showTodoPlan,
  todoPlanContent,
  query,
  isResearching
}) {
  const [reportTitle, setReportTitle] = useState(null);
  const [isFocusedView, setIsFocusedView] = useState(false);
  const [filteredReportContent, setFilteredReportContent] = useState(null);
  const [isRendered, setIsRendered] = useState(false);

  // Handle animation states
  useEffect(() => {
    if (isVisible) {
      // Small delay to ensure DOM is ready before triggering animations
      setTimeout(() => {
        setIsRendered(true);
      }, 50);
    } else {
      setIsRendered(false);
    }
  }, [isVisible]);

  // Update report content and filter it
  useEffect(() => {
    if (!reportContent) {
      setReportTitle(null);
      setFilteredReportContent(null);
      // Reset document title
      document.title = 'AI Research Assistant';
    } else {
      // Filter out thinking sections from report content
      const cleanedContent = removeThinkingSections(reportContent);
      setFilteredReportContent(cleanedContent);

      // Extract title from filtered report content
      const title = extractReportTitle(cleanedContent);
      setReportTitle(title);

      // Update document title with the report title
      if (title) {
        document.title = title;
      } else {
        // Fallback to query if no title was extracted
        document.title = query ? `Research: ${query}` : 'AI Research Assistant';
      }
    }
  }, [reportContent, query]);

  useEffect(() => {
    // Insert code snippets after the component has rendered
    if (isVisible && selectedItem && selectedItem.enrichedData && selectedItem.enrichedData.code_snippets) {
      const container = document.getElementById('code-snippets-container');
      if (container) {
        try {
          const snippets = JSON.parse(container.getAttribute('data-snippets'));

          // Clear the container first
          while (container.firstChild) {
            container.removeChild(container.firstChild);
          }

          // Store references to roots for cleanup
          const roots = [];

          // Render each snippet as a React component
          snippets.forEach(snippet => {
            const snippetContainer = document.createElement('div');
            snippetContainer.className = 'mb-4';
            container.appendChild(snippetContainer);

            // Use ReactDOM createRoot (React 18+)
            const root = ReactDOM.createRoot(snippetContainer);
            root.render(<CodeSnippetViewer snippet={snippet} initialCollapsed={false} />);
            roots.push(root);
          });

          // Cleanup function to unmount components when dependencies change
          return () => {
            roots.forEach(root => {
              try {
                root.unmount();
              } catch (e) {
                console.error('Error unmounting root:', e);
              }
            });
          };
        } catch (error) {
          console.error('Error rendering code snippets:', error);
        }
      }
    }
  }, [isVisible, selectedItem]);

  const handleFullscreen = () => {
    // Toggle focused view
    setIsFocusedView(!isFocusedView);
  };

  // Determine title based on content type
  let title = "Details";
  if (showFinalReport) {
    title = reportTitle || "Research Summary";
  } else if (selectedItem && selectedItem.title) {
    title = selectedItem.title;
  } else if (selectedItem) {
    // Fallback for items without a specific title, e.g. raw activity
    title = selectedItem.type ? selectedItem.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : "Details";
  }

  // Render content placeholder when no report is available
  const renderPlaceholder = () => (
    <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-8">
      <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>

      <p className="text-center max-w-md">
        {isResearching
          ? "Research in progress. The report will appear here when ready."
          : "Start a research query to generate a report."}
      </p>
    </div>
  );

  return (
    <div
      id="details-panel"
      className={`bg-gradient-to-br from-blue-50/20 via-white to-slate-50/30 flex flex-col ${isVisible ? 'visible' : ''}`}
      aria-hidden={!isVisible}
    >
      {/* Fixed Header - only show for report/item details, not for todo plan */}
      {!showTodoPlan && (
        <div className="border-b border-gray-200 p-3 flex items-center justify-between bg-white sticky top-0 z-10 flex-shrink-0">
          <div className="flex items-center space-x-3 flex-1">
            <div className="p-2 bg-gray-100 rounded flex-shrink-0">
              {showFinalReport ? (
                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <h2 className="font-medium text-lg truncate" id="details-title">
                {showFinalReport && query ? query : title}
              </h2>
            </div>
          </div>
          <div className="flex items-center ml-3">
            <button
              id="minimize-details"
              className="p-1 text-gray-500 hover:bg-gray-100 rounded flex items-center justify-center w-8 h-8"
              onClick={onClose}
              aria-label="Minimize panel"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Scrollable Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto pb-6">
        {/* Content based on what's being shown */}
        {!showFinalReport && selectedItem && (
          <div
            id="details-content"
            className="p-6"
            dangerouslySetInnerHTML={{ __html: generateEnhancedDetailsHtml(selectedItem) }}
          />
        )}

        {/* Final Report */}
        {showFinalReport && (
          <div className="min-h-full">
            {filteredReportContent ? (
              <FinalReport reportContent={filteredReportContent} isFocusedView={isFocusedView} analysisData={analysisData} />
            ) : (
              renderPlaceholder()
            )}
          </div>
        )}

        {/* Todo Plan - Enhanced Professional Design */}
        {showTodoPlan && (
          <div className="h-full flex flex-col">
            {todoPlanContent ? (
              <div className="bg-white h-full flex flex-col">
                {/* Enhanced Header with Summary - Salesforce Theme */}
                <div className="bg-gradient-to-r from-[#f3f3f3]/80 via-white/50 to-blue-50/50 border-b border-slate-200/60 px-6 py-5 flex-shrink-0 backdrop-blur-sm shadow-[0_2px_12px_rgba(0,0,0,0.06)]">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="relative p-3 bg-gradient-to-br from-[#0176d3] to-[#014486] rounded-2xl shadow-[0_4px_12px_rgba(1,118,211,0.3)] hover:shadow-[0_6px_16px_rgba(1,118,211,0.4)] transition-all duration-300 group">
                        <svg className="w-6 h-6 text-white transition-transform duration-300 group-hover:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                        </svg>
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-br from-green-400 to-green-500 rounded-full border-2 border-white shadow-sm animate-pulse"></div>
                      </div>
                      <div className="flex items-center gap-3">
                        <h2 className="text-xl font-bold bg-gradient-to-r from-[#032d60] via-[#0176d3] to-[#0176d3] bg-clip-text text-transparent tracking-tight">
                          Research Progress
                        </h2>
                        {(() => {
                          const versionMatch = todoPlanContent.match(/v(\d+)/);
                          if (versionMatch) {
                            return (
                              <span className="px-3 py-1 bg-gradient-to-r from-[#0176d3] to-[#014486] text-white text-xs font-bold rounded-lg shadow-[0_2px_8px_rgba(1,118,211,0.3)] hover:shadow-[0_2px_12px_rgba(1,118,211,0.4)] transition-all duration-300">
                                v{versionMatch[1]}
                              </span>
                            );
                          }
                          return null;
                        })()}
                      </div>
                    </div>
                    <button
                      onClick={onClose}
                      className="p-2.5 rounded-xl hover:bg-slate-100 transition-all duration-200 hover:scale-110 text-slate-500 hover:text-[#0176d3] group"
                      aria-label="Close"
                    >
                      <svg className="w-5 h-5 transition-transform duration-200 group-hover:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Scrollable Content - Gradient Background */}
                <div className="flex-1 overflow-y-auto p-5 bg-gradient-to-br from-blue-50/30 via-white to-slate-50/50">
                  {(() => {
                    // Extract version for logging
                    const versionMatch = todoPlanContent.match(/v(\d+)/);

                    // Parse markdown into sections
                    const sections = [];
                    const lines = todoPlanContent.split('\n');
                    let currentSection = null;
                    let sectionContent = [];
                    let headerLine = '';

                    lines.forEach((line, idx) => {
                      // Save header (first # line)
                      if (line.startsWith('# ') && !headerLine) {
                        headerLine = line.replace(/^# /, '');
                        return;
                      }

                      // Skip metadata lines (Topic, Last Updated, etc.)
                      if (line.startsWith('**Topic:**') || line.startsWith('**Last Updated:**')) {
                        return;
                      }

                      if (line.startsWith('## ')) {
                        // Save previous section (even if empty)
                        if (currentSection) {
                          sections.push({
                            ...currentSection,
                            content: sectionContent.join('\n')
                          });
                        }

                        // Start new section - remove any emojis from title
                        let title = line.replace('## ', '');
                        // Remove emojis from title
                        title = title.replace(/[\u{1F300}-\u{1F9FF}]/gu, '').trim();

                        let icon = '📋';
                        let colorClass = 'text-gray-700';
                        let isDefaultOpen = false;

                        if (title.includes('Pending')) {
                          icon = (
                            <svg className="w-5 h-5 text-[#0176d3]" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                            </svg>
                          );
                          colorClass = 'text-[#0176d3]';
                          isDefaultOpen = false;
                        } else if (title.includes('Processing')) {
                          icon = (
                            <svg className="w-5 h-5 text-amber-600" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                            </svg>
                          );
                          colorClass = 'text-amber-600';
                          isDefaultOpen = true; // Always show processing tasks
                        } else if (title.includes('Completed')) {
                          icon = (
                            <svg className="w-5 h-5 text-[#04844b]" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                          );
                          colorClass = 'text-[#04844b]';
                          isDefaultOpen = false;
                        } else if (title.includes('Cancelled')) {
                          icon = (
                            <svg className="w-5 h-5 text-slate-500" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                          );
                          colorClass = 'text-slate-500';
                          isDefaultOpen = false;
                        }

                        currentSection = { title, icon, colorClass, isDefaultOpen };
                        sectionContent = [];
                      } else if (currentSection) {
                        sectionContent.push(line);
                      }
                    });

                    // Save last section (even if empty)
                    if (currentSection) {
                      sections.push({
                        ...currentSection,
                        content: sectionContent.join('\n')
                      });
                    }

                    console.log('[DetailsPanel] Rendering todo plan - version in content:', versionMatch ? versionMatch[1] : 'unknown');
                    console.log('[DetailsPanel] Parsed sections:', sections.length, sections.map(s => ({ title: s.title, taskCount: (s.content.match(/^[ ]*[-*] \[[ x~]/gm) || []).length })));
                    console.log('[DetailsPanel] Full todoPlanContent length:', todoPlanContent?.length);

                    // Render sections with continuous numbering
                    let globalTaskNumber = 0; // Global counter across all sections

                    return (
                      <div className="space-y-2">
                        {/* Collapsible Sections */}
                        {sections.length > 0 ? sections.map((section, idx) => {
                          // Count tasks in this section (count lines starting with [ ])
                          const taskCount = (section.content.match(/^[ ]*[-*] \[[ x~]/gm) || []).length;

                          return (
                            <CollapsibleSection
                              key={idx}
                              title={section.title}
                              isDefaultOpen={section.isDefaultOpen}
                              icon={section.icon}
                              colorClass={section.colorClass}
                              taskCount={taskCount}
                            >
                              {(() => {
                                // Track current section for processing/pending differentiation
                                // Initialize based on the section title we already parsed
                                let currentSection = null;
                                if (section.title.includes('Pending')) currentSection = 'pending';
                                else if (section.title.includes('Processing')) currentSection = 'processing';
                                else if (section.title.includes('Completed')) currentSection = 'completed';
                                else if (section.title.includes('Cancelled')) currentSection = 'cancelled';

                                return (
                                  <ReactMarkdown
                                    className="markdown-todo-content"
                                    components={{
                                      h1: ({ node, ...props }) => (
                                        <h1 className="text-2xl font-bold mb-6 pb-3 text-gray-900 border-b-2 border-[#0176d3]" {...props} />
                                      ),
                                      h2: ({ node, children, ...props }) => {
                                        const text = String(children);

                                        // Track current section
                                        if (text.includes('Pending')) currentSection = 'pending';
                                        else if (text.includes('Processing')) currentSection = 'processing';
                                        else if (text.includes('Completed')) currentSection = 'completed';
                                        else if (text.includes('Cancelled')) currentSection = 'cancelled';

                                        // Remove emojis from display text
                                        const cleanText = text.replace(/[\u{1F300}-\u{1F9FF}]/gu, '').trim();

                                        let textColor = 'text-gray-700';

                                        if (text.includes('🎯') || text.includes('Pending') || text.includes('Active')) {
                                          textColor = 'text-[#0176d3]';
                                        } else if (text.includes('🔄') || text.includes('Processing')) {
                                          textColor = 'text-amber-600';
                                        } else if (text.includes('✅') || text.includes('Completed')) {
                                          textColor = 'text-[#04844b]';
                                        } else if (text.includes('❌') || text.includes('Cancelled')) {
                                          textColor = 'text-slate-500';
                                        }

                                        return (
                                          <div className={`flex items-center gap-2 mb-4 mt-6 first:mt-2`}>
                                            <h2 className={`text-sm font-bold ${textColor}`} {...props}>{cleanText}</h2>
                                          </div>
                                        );
                                      },
                                      h3: ({ node, ...props }) => (
                                        <h3 className="text-base font-semibold mb-3 mt-6 text-gray-700" {...props} />
                                      ),
                                      ul: ({ node, ...props }) => (
                                        <ul className="space-y-0.5 my-1" {...props} />
                                      ),
                                      li: ({ node, children, ...props }) => {
                                        // Extract text properly from children (handle React nodes)
                                        // BUT stop at nested lists to avoid including nested metadata in main text
                                        const extractText = (node, depth = 0) => {
                                          if (typeof node === 'string') return node;
                                          if (Array.isArray(node)) {
                                            let result = '';
                                            for (let i = 0; i < node.length; i++) {
                                              const child = node[i];
                                              // Check if this is a React element with type 'ul' or 'ol'
                                              if (child && typeof child === 'object' && child.type) {
                                                const typeName = typeof child.type === 'string' ? child.type : child.type?.name;
                                                if (typeName === 'ul' || typeName === 'ol') {
                                                  // Found nested list, stop here
                                                  break;
                                                }
                                              }
                                              result += extractText(child, depth + 1);
                                            }
                                            return result;
                                          }
                                          // Check for React elements
                                          if (node && typeof node === 'object' && node.type) {
                                            const typeName = typeof node.type === 'string' ? node.type : node.type?.name;
                                            if (typeName === 'ul' || typeName === 'ol') {
                                              return '';
                                            }
                                          }
                                          if (node && node.props && node.props.children) {
                                            return extractText(node.props.children, depth + 1);
                                          }
                                          return '';
                                        };

                                        // Also extract timestamp from nested metadata (before stopping at nested lists)
                                        const extractTimestamp = (node) => {
                                          if (typeof node === 'string') {
                                            const match = node.match(/(\d{2}:\d{2}:\d{2})/);
                                            return match ? match[1] : '';
                                          }
                                          if (Array.isArray(node)) {
                                            for (let n of node) {
                                              const ts = extractTimestamp(n);
                                              if (ts) return ts;
                                            }
                                            return '';
                                          }
                                          if (node && node.props && node.props.children) {
                                            return extractTimestamp(node.props.children);
                                          }
                                          return '';
                                        };

                                        const text = extractText(children);
                                        const timestamp = extractTimestamp(children);

                                        // Skip ALL nested metadata bullets - check for content patterns
                                        // These include: "From user:", "Created:", "Completed at:", "Reason:", "Found via", etc.
                                        if (text.match(/^\s*(From user:|Created:|Completed at:|Reason:|Found via|✓ Found via)/i)) {
                                          return null; // Don't render these nested metadata bullets
                                        }

                                        // Also skip if the text is wrapped in emphasis/italic (starts with italic content)
                                        if (text.match(/^\s*[\*_]/) || (Array.isArray(children) && children[0]?.type === 'em')) {
                                          return null;
                                        }

                                        const isChecked = text.match(/^\s*\[x\]|^\s*\[X\]/i);
                                        const isUnchecked = text.match(/^\s*\[ \]/);
                                        const isStrikethrough = text.match(/^\s*\[~\]/);

                                        if (isChecked || isUnchecked || isStrikethrough) {
                                          // Determine task status using currentSection
                                          const isPending = isUnchecked && currentSection === 'pending';
                                          const isProcessing = isUnchecked && currentSection === 'processing';
                                          const isCompleted = isChecked;
                                          const isCancelled = isStrikethrough;

                                          // Increment global task number for each task item
                                          globalTaskNumber++;
                                          const currentTaskNumber = globalTaskNumber;

                                          // Extract source from emojis BEFORE cleaning (check original text)
                                          const hasSteeringEmoji = text.includes('🎯');
                                          const hasOriginalEmoji = text.includes('📋');
                                          const hasGapEmoji = text.includes('🔍');

                                          // Clean text from checkbox markers
                                          let taskText = text.replace(/^\s*\[(x|X| |~)\]\s*/g, '');

                                          // Extract task ID BEFORE removing bold markers (format: **[ID]**)
                                          const taskIdMatch = taskText.match(/\*\*\[(\d+)\]\*\*/);
                                          const taskId = taskIdMatch ? taskIdMatch[1] : null;

                                          // Remove bold markers
                                          taskText = taskText.replace(/\*\*/g, '');

                                          // Remove task ID from text
                                          taskText = taskText.replace(/\[\d+\]\s*/g, '');

                                          // Remove ALL emojis and unicode symbols from display text
                                          taskText = taskText
                                            .replace(/[\u{1F000}-\u{1FFFF}]/gu, '') // All emoji ranges
                                            .replace(/[\u2600-\u27BF]/g, '') // Misc symbols
                                            .replace(/[\uE000-\uF8FF]/g, '') // Private use area
                                            .replace(/✓|✗|⏳|🔄|✅|❌|🎯|📋|🔍|☐|☑|☒/g, '') // Specific symbols
                                            .trim();

                                          // Final clean display text (no emojis, no task ID, no checkboxes)
                                          const cleanDisplayText = taskText;

                                          return (
                                            <li className="group mb-2 last:mb-0" {...props}>
                                              <div className={`relative border transition-all duration-200 overflow-hidden shadow-sm hover:shadow-md ${isCompleted
                                                ? 'border-[#04844b]/30 bg-gradient-to-r from-[#04844b]/5 to-transparent rounded-lg'
                                                : isCancelled
                                                  ? 'border-slate-300 bg-slate-50/80 rounded-lg opacity-75'
                                                  : 'border-slate-200/70 bg-white rounded-xl hover:border-[#0176d3]/50'
                                                }`}>
                                                {/* Left Status Indicator Bar */}
                                                <div className={`absolute left-0 top-0 bottom-0 w-1 ${isCompleted
                                                  ? 'bg-gradient-to-b from-[#04844b] to-[#04844b]/60'
                                                  : isCancelled
                                                    ? 'bg-slate-400'
                                                    : isProcessing
                                                      ? 'bg-gradient-to-b from-amber-600 to-amber-500'
                                                      : 'bg-gradient-to-b from-[#0176d3] to-[#0176d3]/70'
                                                  }`}></div>

                                                <div className="pl-4 pr-4 py-3">
                                                  {/* Task Number and Description Line */}
                                                  <div className="flex items-start gap-2">
                                                    {/* Bullet - simple dot */}
                                                    <div className="flex-shrink-0 mt-1">
                                                      <span className={`font-bold text-sm ${isCompleted
                                                        ? 'text-[#04844b]'
                                                        : isCancelled
                                                          ? 'text-slate-400'
                                                          : isProcessing
                                                            ? 'text-amber-600'
                                                            : 'text-[#0176d3]'
                                                        }`}>
                                                        •
                                                      </span>
                                                    </div>

                                                    {/* Task ID */}
                                                    {taskId && (
                                                      <span className="text-xs text-slate-500 font-mono font-medium flex-shrink-0">[{taskId}]</span>
                                                    )}

                                                    {/* Task Description - Normal for all types */}
                                                    <div className={`flex-1 text-sm leading-relaxed ${isCancelled
                                                      ? 'text-slate-800 line-through'
                                                      : 'text-slate-800'
                                                      }`}>
                                                      {cleanDisplayText}
                                                    </div>
                                                  </div>

                                                  {/* Metadata Row - Plain text timestamp + Source badge */}
                                                  <div className="flex items-center gap-2 ml-5 mt-1.5 flex-wrap">
                                                    {/* Timestamp - Plain grey text - Show for pending/completed/cancelled, hide for processing */}
                                                    {timestamp && !isProcessing && (
                                                      <span className="text-xs text-slate-400 whitespace-nowrap">
                                                        {isCompleted ? 'Completed' : isCancelled ? 'Cancelled' : 'Created'} {timestamp}
                                                      </span>
                                                    )}

                                                    {/* Source Badge - Always show */}
                                                    {hasSteeringEmoji && (
                                                      <span className="inline-flex items-center px-2 py-0.5 bg-gradient-to-r from-[#0176d3] to-[#014486] text-white text-xs font-semibold rounded whitespace-nowrap">
                                                        Steering
                                                      </span>
                                                    )}
                                                    {hasOriginalEmoji && (
                                                      <span className="inline-flex items-center px-2 py-0.5 bg-gradient-to-r from-slate-700 to-slate-600 text-white text-xs font-semibold rounded whitespace-nowrap">
                                                        Initial Query
                                                      </span>
                                                    )}
                                                    {hasGapEmoji && (
                                                      <span className="inline-flex items-center px-2 py-0.5 bg-gradient-to-r from-amber-50 to-orange-50 text-amber-700 text-xs font-semibold rounded border border-amber-200 whitespace-nowrap">
                                                        Research Gap
                                                      </span>
                                                    )}
                                                  </div>
                                                </div>
                                              </div>
                                            </li>
                                          );
                                        }

                                        // Non-task list items (regular bullet points)
                                        return <li className="flex items-start gap-2 px-3 py-1.5 text-sm leading-relaxed text-slate-700" {...props}>
                                          <span className="text-slate-400 mt-1.5">•</span>
                                          <div className="flex-1">{children}</div>
                                        </li>;
                                      },
                                      p: ({ node, ...props }) => (
                                        <p className="text-slate-700 leading-relaxed mb-4 text-sm" {...props} />
                                      ),
                                      strong: ({ node, ...props }) => (
                                        <strong className="font-semibold text-slate-900" {...props} />
                                      ),
                                      em: ({ node, ...props }) => (
                                        <em className="not-italic text-slate-600 text-sm" {...props} />
                                      ),
                                      code: ({ node, inline, ...props }) =>
                                        inline ? (
                                          <code className="bg-slate-100 px-2 py-1 rounded text-sm font-mono text-slate-800 border border-slate-200" {...props} />
                                        ) : (
                                          <code className="block bg-slate-50 p-4 rounded-lg text-sm font-mono text-slate-800 overflow-x-auto border border-slate-200 my-3" {...props} />
                                        ),
                                      blockquote: ({ node, ...props }) => (
                                        <blockquote className="border-l-4 border-[#0176d3] pl-4 py-2 my-4 bg-blue-50 rounded-r italic text-slate-700" {...props} />
                                      ),
                                    }}
                                  >
                                    {section.content}
                                  </ReactMarkdown>
                                );
                              })()}
                            </CollapsibleSection>
                          );
                        }) : (
                          <div className="text-center py-8 text-gray-500">
                            <p>No tasks yet</p>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center p-12 bg-gradient-to-br from-blue-50/30 via-white to-slate-50/50">
                <div className="text-center max-w-md">
                  <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-[#0176d3]/10 via-white to-blue-50/50 rounded-3xl mb-6 shadow-lg shadow-[#0176d3]/10 border border-slate-200/60">
                    <svg className="w-12 h-12 text-[#0176d3]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold bg-gradient-to-r from-[#032d60] to-[#0176d3] bg-clip-text text-transparent mb-3">
                    No Research Plan Yet
                  </h3>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    Your research plan will appear here when the research begins. Tasks will be tracked and updated in real-time.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

    </div>
  );
}

export default DetailsPanel; 