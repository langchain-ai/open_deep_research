import React, { useState, useCallback } from 'react';
import Navbar from './components/Navbar';
import ResearchPanel from './components/ResearchPanel';
import DetailsPanel from './components/DetailsPanel'; // Now handles both item details and report content
import './App.css'; // For global styles

function App() {
  const [isResearching, setIsResearching] = useState(false);
  const [currentQuery, setCurrentQuery] = useState('');
  const [extraEffort, setExtraEffort] = useState(false);
  const [minimumEffort, setMinimumEffort] = useState(false);
  const [benchmarkMode, setBenchmarkMode] = useState(false);
  const [modelProvider, setModelProvider] = useState('gemini'); // Default provider (dev)
  const [modelName, setModelName] = useState('gemini-2.5-flash'); // Default model (dev)
  const [uploadedFileContent, setUploadedFileContent] = useState(null); // Added state for uploaded file content
  const [databaseInfo, setDatabaseInfo] = useState(null); // Added state for database info

  const [isDetailsPanelOpen, setIsDetailsPanelOpen] = useState(false);
  const [detailsPanelContentType, setDetailsPanelContentType] = useState(null); // 'item' or 'report'
  const [detailsPanelContentData, setDetailsPanelContentData] = useState(null);

  // Steering state
  const [currentTodoPlan, setCurrentTodoPlan] = useState("");
  const [todoPlanVersion, setTodoPlanVersion] = useState(0);

  const handleBeginResearch = useCallback((query, extra, minimum, benchmark, modelConfig, fileContent, databaseInfo) => { // Added fileContent and databaseInfo
    setCurrentQuery(query);
    setExtraEffort(extra);
    setMinimumEffort(minimum);
    setBenchmarkMode(benchmark);
    if (modelConfig) {
      setModelProvider(modelConfig.provider);
      setModelName(modelConfig.model);
    }

    setUploadedFileContent(fileContent); // Set uploaded file content
    // Store database info for the research agent
    if (databaseInfo && databaseInfo.length > 0) {
      console.log('Database info passed to research agent:', databaseInfo);
      setDatabaseInfo(databaseInfo); // Store database info in state
    }
    setIsResearching(true);
    setIsDetailsPanelOpen(false); // Close details panel when new research starts
    // Wait for animation to complete before resetting content
    setTimeout(() => {
      setDetailsPanelContentType(null);
      setDetailsPanelContentData(null);
    }, 300);
  }, [uploadedFileContent]);

  const handleShowItemDetails = useCallback((item) => {
    // Set data first, then trigger animation
    setDetailsPanelContentData(item);
    setDetailsPanelContentType('item');
    // Small delay to ensure data is set before animation starts
    setTimeout(() => {
      setIsDetailsPanelOpen(true);
    }, 10);
  }, []);

  const handleShowReportDetails = useCallback((reportContent) => {
    // Set data first, then trigger animation
    setDetailsPanelContentData(reportContent);
    setDetailsPanelContentType('report');
    // Small delay to ensure data is set before animation starts
    setTimeout(() => {
      setIsDetailsPanelOpen(true);
    }, 10);
  }, []);

  const handleCloseDetailsPanel = useCallback(() => {
    setIsDetailsPanelOpen(false);
    // Wait for animation to complete before clearing data
    setTimeout(() => {
      setDetailsPanelContentData(null);
      setDetailsPanelContentType(null);
    }, 300); // Match the transition timing in CSS
  }, []);

  // This callback is used by ResearchPanel to inform App.js that a report is ready.
  const [finalReportData, setFinalReportData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null); // Analysis report data
  const handleReportGenerated = useCallback((report, analysis) => {
    setFinalReportData(report); // Store report data
    if (analysis) {
      setAnalysisData(analysis); // Store analysis data if available
    }
    // Optionally, automatically open the report:
    // handleShowReportDetails(report);
  }, []);

  const handleStopResearch = useCallback(() => {
    console.log('Stopping research from App.js');
    setIsResearching(false);

    // Clear all research-related state
    setFinalReportData(null);
    setAnalysisData(null);
    setCurrentTodoPlan("");
    setTodoPlanVersion(0); // Reset version counter

    // Close details panel if open
    if (isDetailsPanelOpen) {
      setIsDetailsPanelOpen(false);
      setTimeout(() => {
        setDetailsPanelContentType(null);
        setDetailsPanelContentData(null);
      }, 300);
    }
  }, [isDetailsPanelOpen]);

  const handleTodoPlanUpdate = useCallback((todoPlan) => {
    if (todoPlan !== currentTodoPlan) {
      setCurrentTodoPlan(todoPlan);
      setTodoPlanVersion(prev => prev + 1);
    }

    if (todoPlan && !isDetailsPanelOpen) {
      setDetailsPanelContentType('todo');
      setDetailsPanelContentData(todoPlan);
      setTimeout(() => {
        setIsDetailsPanelOpen(true);
      }, 10);
    } else if (detailsPanelContentType === 'todo' && isDetailsPanelOpen) {
      setDetailsPanelContentData(todoPlan);
    } else if (!isDetailsPanelOpen && detailsPanelContentType === 'todo') {
      setDetailsPanelContentData(todoPlan);
    }
  }, [isDetailsPanelOpen, detailsPanelContentType, currentTodoPlan]);

  const handleToggleProgress = useCallback(() => {
    if (detailsPanelContentType === 'todo' && isDetailsPanelOpen) {
      handleCloseDetailsPanel();
    } else if (currentTodoPlan) {
      handleTodoPlanUpdate(currentTodoPlan);
    }
  }, [detailsPanelContentType, isDetailsPanelOpen, currentTodoPlan, handleCloseDetailsPanel, handleTodoPlanUpdate]);

  const handleToggleReport = useCallback(() => {
    if (detailsPanelContentType === 'report' && isDetailsPanelOpen) {
      handleCloseDetailsPanel();
    } else if (finalReportData) {
      handleShowReportDetails(finalReportData);
    }
  }, [detailsPanelContentType, isDetailsPanelOpen, finalReportData, handleCloseDetailsPanel, handleShowReportDetails]);

  return (
    <div className={`app-root ${isDetailsPanelOpen ? 'details-panel-active' : ''}`}>
      <Navbar
        isResearching={isResearching}
        onShowProgress={handleToggleProgress}
        onShowReport={handleToggleReport}
        hasReport={!!finalReportData}
        isProgressOpen={isDetailsPanelOpen && detailsPanelContentType === 'todo'}
        isReportOpen={isDetailsPanelOpen && detailsPanelContentType === 'report'}
      />

      <div className={`app-container ${isResearching ? 'research-active' : ''} ${isDetailsPanelOpen ? 'details-panel-active' : ''}`}>
        <div className="main-panel-wrapper">
          <ResearchPanel
            query={currentQuery}
            extraEffort={extraEffort}
            minimumEffort={minimumEffort}
            benchmarkMode={benchmarkMode}
            modelProvider={modelProvider}
            modelName={modelName}
            uploadedFileContent={uploadedFileContent} // Pass uploadedFileContent
            databaseInfo={databaseInfo} // Pass databaseInfo
            isResearching={isResearching}
            onBeginResearch={handleBeginResearch}
            onReportGenerated={handleReportGenerated}
            onShowItemDetails={handleShowItemDetails}
            onShowReportDetails={handleShowReportDetails}
            onStopResearch={handleStopResearch}
            onTodoPlanUpdate={handleTodoPlanUpdate}
          />
        </div>

        <DetailsPanel
          key={detailsPanelContentType === 'todo' ? `todo-v${todoPlanVersion}` : detailsPanelContentType}
          isVisible={isDetailsPanelOpen}
          onClose={handleCloseDetailsPanel}
          selectedItem={detailsPanelContentType === 'item' ? detailsPanelContentData : null}
          showFinalReport={detailsPanelContentType === 'report'}
          reportContent={detailsPanelContentType === 'report' ? detailsPanelContentData : null}
          analysisData={analysisData}
          showTodoPlan={detailsPanelContentType === 'todo'}
          todoPlanContent={detailsPanelContentType === 'todo' ? detailsPanelContentData : null}
          query={currentQuery}
          isResearching={isResearching}
        />
      </div>
    </div>
  );
}

export default App; 