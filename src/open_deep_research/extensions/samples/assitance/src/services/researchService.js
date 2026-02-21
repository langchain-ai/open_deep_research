/**
 * Service to handle research API requests and SSE streaming
 */
let activeRequest = null;
let lastQuery = null;
let lastExtraEffort = false;
let lastMinimumEffort = false;
let lastBenchmarkMode = false;
let lastModelProvider = null;
let lastModelName = null;
let lastUploadedFileContent = null; // Added to store uploaded file content
let lastParallelEnabled = true;
let lastParallelMax = 2;
let isReconnecting = false;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
// Add references to track active reader and connection check interval
let activeReader = null;
let activeConnectionCheckInterval = null;

// Add this variable to track if the connection should auto-reconnect
let shouldAutoReconnect = true;

// Add variables for connection management
let eventSource = null;
let staleConnectionTimer = null;
let reconnectTimeout = null;
let onEvent = null;
let onComplete = null;
let onError = null;

// Add session tracking to prevent duplicate research requests
let currentSessionId = null;
let lastResponseUrl = null;

// Track request timestamp to prevent rapid duplicate requests
let lastRequestTimestamp = 0;
const REQUEST_DEBOUNCE_TIME = 2000; // 2 seconds between requests

// Track requests in flight to prevent duplicates
let requestsInProgress = new Set();

// Flag to indicate if research is complete
let isResearchComplete = false;

/**
 * Set the auto-reconnect behavior
 * @param {boolean} value - Whether to auto-reconnect on connection loss
 */
export const setShouldAutoReconnect = (value) => {
  console.log(`Setting shouldAutoReconnect to ${value}`);
  shouldAutoReconnect = value;
};

// Export a function to disconnect the event source without destroying history
export const disconnectResearchEventSource = () => {
  console.log('Manually disconnecting research event source');

  if (eventSource) {
    console.log('Closing SSE connection manually');
    eventSource.close();
    eventSource = null;
  }

  if (staleConnectionTimer) {
    console.log('Clearing stale connection timer');
    clearInterval(staleConnectionTimer);
    staleConnectionTimer = null;
  }

  // Reset reconnect state but don't clear history
  isReconnecting = false;
  reconnectAttempts = 0;
};

// Make the disconnect function available globally for emergency use
window.disconnectResearchEventSource = disconnectResearchEventSource;

/**
 * Cancel any ongoing research request
 * @returns {boolean} true if a request was canceled, false otherwise
 */
export const cancelResearch = () => {
  console.log('cancelResearch called');

  // Explicitly set flag to prevent auto-reconnection
  shouldAutoReconnect = false;

  // Request backend to stop research execution
  if (currentSessionId) {
    console.log(`Requesting backend to stop research for session ${currentSessionId}`);
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';
    fetch(`${API_BASE_URL}/stop/${currentSessionId}`, {
      method: 'POST',
    })
      .then(response => response.json())
      .then(data => {
        console.log('Backend stop response:', data);
      })
      .catch(err => {
        console.log('Error requesting backend stop:', err);
      });
  }

  if (activeReader) {
    console.log('Canceling active reader');
    activeReader.cancel("User canceled research").catch(err => {
      console.log('Error canceling reader:', err);
    });
    activeReader = null;
  }

  // Clear the connection check interval if it exists
  if (activeConnectionCheckInterval) {
    console.log('Clearing connection check interval');
    clearInterval(activeConnectionCheckInterval);
    activeConnectionCheckInterval = null;
  }

  // Reset the active request
  const canceledRequest = activeRequest;
  activeRequest = null;

  console.log(`Successfully canceled research request for "${canceledRequest}"`);

  // Clear the stale connection timer
  if (staleConnectionTimer) {
    clearTimeout(staleConnectionTimer);
    staleConnectionTimer = null;
  }

  // Clear any pending reconnection attempts
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }

  // Close the event source if it exists
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Clear all in-progress requests to prevent orphaned requests
  requestsInProgress.clear();

  // Clear polling interval if active
  if (pollingInterval) {
    console.log('Clearing polling interval');
    clearInterval(pollingInterval);
    pollingInterval = null;
  }

  // Stop plan polling
  stopPlanPolling();

  // Reset session tracking
  currentSessionId = null;
  lastResponseUrl = null;

  // Reset research completion flag
  isResearchComplete = false;

  return true;
};

export const startResearch = async (
  query,
  extraEffort = false,
  minimumEffort = false,
  benchmarkMode = false,
  provider = null,
  model = null,
  uploadedFileContent = null, // Add uploadedFileContent parameter
  databaseInfo = null, // Add databaseInfo parameter
  onEventHandler,
  onCompleteHandler,
  onErrorHandler,
  enableSteering = true, // Add steering parameter
  enableParallel = true,
  parallelMax = 2
) => {
  console.log('[DEBUG] startResearch called with:', { query, provider, model, uploadedFileContent, databaseInfo });

  // Store the handlers in module-level variables so connectToEventSource can access them
  onEvent = onEventHandler;
  onComplete = onCompleteHandler;
  onError = onErrorHandler;

  console.log('[DEBUG] Stored event handlers:', { onEvent: typeof onEvent, onComplete: typeof onComplete, onError: typeof onError });

  try {
    // Define your API base URL
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

    // FIRST: Cleanup any previous sessions to ensure fresh start
    console.log('Cleaning up any previous sessions for fresh start...');
    try {
      const cleanupResponse = await fetch(`${API_BASE_URL}/cleanup-all`, {
        method: 'POST',
      });
      if (cleanupResponse.ok) {
        const cleanupData = await cleanupResponse.json();
        console.log('✅ Cleanup complete:', cleanupData);
      }
    } catch (cleanupErr) {
      console.log('⚠️ Cleanup request failed (non-fatal):', cleanupErr);
    }

    // Use the original deep-research endpoint
    const apiUrl = `${API_BASE_URL}/deep-research`;
    // console.log(`Starting research with URL: ${apiUrl}, steering: ${enableSteering}`);

    // Log the extra effort and minimum effort settings
    console.log(`Research settings: extraEffort=${extraEffort}, minimumEffort=${minimumEffort}, benchmarkMode=${benchmarkMode}`);

    // Store current request parameters for potential reconnection
    lastQuery = query;
    lastExtraEffort = extraEffort;
    lastMinimumEffort = minimumEffort;
    lastBenchmarkMode = benchmarkMode;
    lastModelProvider = provider;
    lastModelName = model;
    lastUploadedFileContent = uploadedFileContent;
    lastParallelEnabled = enableParallel;
    lastParallelMax = parallelMax;

    // Reset research completion flag
    isResearchComplete = false;
    shouldAutoReconnect = true;

    // Generate a unique session ID for this research request
    const sessionId = generateSessionId();
    currentSessionId = sessionId;

    // Check for duplicate requests
    const now = Date.now();
    if (now - lastRequestTimestamp < REQUEST_DEBOUNCE_TIME) {
      console.log('Ignoring duplicate request within debounce time');
      return;
    }
    lastRequestTimestamp = now;

    // Check if this request is already in progress
    const requestKey = `${query}-${extraEffort}-${minimumEffort}-${benchmarkMode}-${provider}-${model}`;
    if (requestsInProgress.has(requestKey)) {
      console.log('Request already in progress, ignoring duplicate');
      return;
    }
    requestsInProgress.add(requestKey);

    // Create cleanup function for this request
    const cleanupRequest = () => {
      console.log(`Cleaning up request: ${requestKey}`);
      requestsInProgress.delete(requestKey);
      activeRequest = null;
    };

    // Prepare the request body - always use original format but add steering flag
    const requestBody = {
      query: query,
      extra_effort: extraEffort,
      minimum_effort: minimumEffort,
      benchmark_mode: benchmarkMode,
      streaming: true,
      steering_enabled: enableSteering,  // Add steering flag
      parallel_search_enabled: enableParallel,
      parallel_search_max_concurrency: parallelMax
    };

    // Add database_info if provided
    if (databaseInfo && Array.isArray(databaseInfo) && databaseInfo.length > 0) {
      console.log('[DEBUG] Adding database_info to request:', databaseInfo);
      requestBody.database_info = databaseInfo;
    }

    // Add model configuration if provided
    if (provider) {
      requestBody.provider = provider;
    }
    if (model) {
      requestBody.model = model;
    }

    // Convert uploadedFileContent array to string if provided
    if (uploadedFileContent && Array.isArray(uploadedFileContent) && uploadedFileContent.length > 0) {
      console.log('[DEBUG] Converting uploaded file content array to string');
      const combinedContent = uploadedFileContent.map(fileData => {
        if (fileData && fileData.content) {
          return `=== ${fileData.filename || 'Uploaded File'} ===\n${fileData.content}\n`;
        }
        return '';
      }).join('\n');

      if (combinedContent.trim()) {
        requestBody.uploaded_data_content = combinedContent;
        console.log('[DEBUG] Added uploaded_data_content to request body, length:', combinedContent.length);
      }
    } else if (uploadedFileContent && typeof uploadedFileContent === 'string') {
      // Handle case where it's already a string
      requestBody.uploaded_data_content = uploadedFileContent;
      console.log('[DEBUG] Added uploaded_data_content (string) to request body, length:', uploadedFileContent.length);
    }

    console.log(`[MODEL SELECTION] Final request → provider="${requestBody.provider}", model="${requestBody.model}"`);
    console.log('Making research request with body:', requestBody);

    // Make the initial request to start research
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Research request response:', data);

    if (enableSteering) {
      // console.log('[STEERING] Research started with steering enabled');
      // Extract session ID from the stream URL for steering
      const streamUrl = data.stream_url;
      if (streamUrl) {
        // Stream URL format is "/stream/uuid", so extract the UUID part
        const sessionId = streamUrl.replace('/stream/', '');
        currentSessionId = sessionId;
        // console.log(`[STEERING] Set current session ID: ${currentSessionId}`);

        // Start plan polling for real-time updates
        // console.log('[STEERING] Starting plan polling for real-time updates');
        startPlanPolling();
      }
    }

    // Extract the stream URL from the response
    const streamUrl = data.stream_url;
    if (!streamUrl) {
      throw new Error('No stream URL provided in response');
    }

    console.log(`Received stream URL: ${streamUrl}`);
    lastResponseUrl = streamUrl;

    // Connect to the event stream using the helper function
    connectToEventSource(streamUrl, cleanupRequest);

  } catch (error) {
    console.error('Error in startResearch:', error);
    if (onError) {
      onError(error);
    }
  }
};

/**
 * Helper function to connect to an EventSource at the given URL
 */
const connectToEventSource = (url, cleanupRequest) => {
  console.log(`[DEBUG] connectToEventSource called with url: ${url}, onEvent handler:`, typeof onEvent);

  // Close existing event source if any
  if (eventSource) {
    eventSource.close();
  }

  // Define your API base URL as relative path
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  // Create the full stream URL
  const fullUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;

  console.log(`Connecting to event source: ${fullUrl}`);

  // Create a new EventSource to stream responses
  eventSource = new EventSource(fullUrl);

  // Track lastEventTime for timeout detection
  let lastEventTime = Date.now();

  // --- Add onopen handler --- 
  eventSource.onopen = (event) => {
    console.log(`SSE connection opened successfully! ReadyState: ${eventSource?.readyState}`, event);
    // Reset last event time on successful open
    lastEventTime = Date.now();
    // Reset reconnect attempts if we successfully reopen after a failure
    if (isReconnecting) {
      console.log('Successfully reconnected to SSE stream.');
      reconnectAttempts = 0;
      isReconnecting = false;
    }
  };
  // --- End onopen handler --- 

  // Clear any existing timers
  if (staleConnectionTimer) {
    clearTimeout(staleConnectionTimer);
  }

  // Set up a timer to detect stale connections (30 seconds with no events)
  staleConnectionTimer = setInterval(() => {
    // CRITICAL CHECK: If research is already complete, immediately clear this timer and return
    if (isResearchComplete) {
      console.log('Research already complete - clearing stale connection timer');
      clearInterval(staleConnectionTimer);
      staleConnectionTimer = null;
      return;
    }

    const timeSinceLastEvent = Date.now() - lastEventTime;
    if (timeSinceLastEvent > 30000) { // 30 seconds (increased from 20)
      console.log('Connection appears to be stale - no events in 30 seconds');
      // Double-check research isn't complete before handling loss
      if (!isResearchComplete) {
        handleConnectionLoss(cleanupRequest);
      }
    }
  }, 5000); // Check every 5 seconds

  // Add handler for all event types using a general processing function
  const processEvent = (event, eventType) => {
    // Update the last event time for all events
    lastEventTime = Date.now();

    console.log(`[DEBUG] processEvent called with eventType: ${eventType}, event.data:`, event.data);

    try {
      // Check if event.data is undefined, empty, or not valid JSON
      if (!event.data || event.data === 'undefined') {
        console.log(`Received ${eventType} event with empty or undefined data`);
        return;
      }

      // Parse the event data
      const parsedData = JSON.parse(event.data);
      console.log(`[DEBUG] Parsed data for ${eventType}:`, parsedData);

      // Log the event type (except heartbeats to avoid spam)
      if (eventType !== 'heartbeat') {
        console.log(`[DEBUG] Received event type: ${eventType}`, parsedData);
      }

      // --- START MODIFICATION: Handle array of events ---
      if (Array.isArray(parsedData)) {
        console.log(`Received array of ${parsedData.length} events for type ${eventType}`);
        parsedData.forEach((singleEventData, index) => {
          // Forward each event in the array individually
          if (eventType !== 'heartbeat' && onEvent) {
            console.log(`[DEBUG] Calling onEvent for array item ${index + 1}/${parsedData.length}`);
            onEvent(singleEventData);
          }
        });
      } else {
        // Forward single event to the handler
        if (eventType !== 'heartbeat' && onEvent) {
          console.log(`[DEBUG] Calling onEvent for single event of type ${eventType}`);
          onEvent(parsedData);
        } else if (!onEvent) {
          console.log(`[DEBUG] onEvent handler is null/undefined for event type ${eventType}`);
        }
      }
      // --- END MODIFICATION ---

    } catch (error) {
      console.error(`Error handling SSE ${eventType} event: ${error}`);
      console.error(`Event data that caused error:`, event.data);

      // Don't propagate parsing errors to onError as they're usually not fatal
      if (error instanceof SyntaxError) {
        console.warn(`JSON parsing error in ${eventType} event - ignoring`);
      } else if (onError) {
        onError(error);
      }
    }
  };

  // Add listeners for specific event types
  const expectedEventTypes = [
    'connected', 'node_start', 'node_end', 'tool_start', 'tool_end',
    'search_started', 'search_results', 'knowledge_gap', 'research_complete',
    'report_generation', 'compiling_information', 'new_sources', 'error',
    'reconnecting', 'connection_interrupted', 'custom_event', 'update',
    // Add these specific event types from the server code
    'generating_report', 'improving_report', 'final_report', 'search_sources_found',
    'token_stream', 'tool_event', 'heartbeat', 'activity_generated',
    'code_snippet_generated', // If you decide to use a separate event
    // Steering-related events
    'steering_plan_updated', 'steering_message_processed', 'todo_updated',
    // Analysis pipeline events
    'analysis_started', 'analysis_complete'
  ];

  // Add listeners for all expected event types
  expectedEventTypes.forEach(eventType => {
    eventSource.addEventListener(eventType, (event) => {
      // Special handling for research_complete events to disable auto-reconnect
      if (eventType === 'research_complete') {
        console.log('Research complete event received through named listener - disabling auto-reconnect');
        shouldAutoReconnect = false;
        isResearchComplete = true; // <-- Set the research complete flag

        // Stop plan polling when research completes
        stopPlanPolling();

        // Clean up this request since research is complete
        if (cleanupRequest) {
          console.log('Research complete - cleaning up request');
          cleanupRequest();
        }

        // Properly close the connection and clear any timers
        if (eventSource) {
          console.log('Closing SSE connection after receiving research_complete');
          eventSource.close();
          eventSource = null;
        }

        if (staleConnectionTimer) {
          console.log('Clearing stale connection timer after research_complete');
          clearInterval(staleConnectionTimer);
          staleConnectionTimer = null;
        }

        // Reset reconnect state and shutdown all connections
        isReconnecting = false;
        reconnectAttempts = 0;

        // Reset tracking variables
        activeRequest = null;
        currentSessionId = null;

        // Call the complete callback
        if (onComplete) {
          onComplete();
        }
      }
      processEvent(event, eventType);
    });
  });

  // Add a catch-all listener for unnamed events or events with types not in our list
  eventSource.addEventListener('message', (event) => {
    console.log(`[DEBUG] Received event through 'message' listener:`, event.data);

    try {
      if (!event.data || event.data === 'undefined') {
        console.log(`Received generic message event with empty or undefined data`);
        return;
      }

      const data = JSON.parse(event.data);
      const eventType = data.event_type || 'unknown';

      console.log(`[DEBUG] Received event through 'message' listener with type: ${eventType}`, data);

      // Check if this event type should have been caught by a specific listener
      if (expectedEventTypes.includes(eventType)) {
        console.warn(`Event type '${eventType}' was received through 'message' listener but should have been caught by specific listener`);
      }

      // Forward all events to the handler
      if (onEvent) {
        console.log(`[DEBUG] Calling onEvent from message listener for type: ${eventType}`);
        onEvent(data);
      } else {
        console.log(`[DEBUG] onEvent handler is null/undefined in message listener`);
      }
    } catch (error) {
      console.error(`Error handling generic message event: ${error}`);
      console.error(`Event data that caused error:`, event.data);
    }
  });

  // Keep the onmessage handler for backward compatibility and for any unnamed events
  eventSource.onmessage = (event) => {
    console.log(`[DEBUG] Received event through onmessage:`, event.data);

    // Update the last event time
    lastEventTime = Date.now();

    try {
      // Check if event data is valid
      if (!event.data || event.data === 'undefined') {
        console.log(`Received onmessage event with empty or undefined data`);
        return;
      }

      // Parse the event data
      const parsedData = JSON.parse(event.data);
      console.log(`[DEBUG] Parsed data in onmessage:`, parsedData);

      // --- START MODIFICATION: Handle array of events ---
      if (Array.isArray(parsedData)) {
        console.log(`Received array of ${parsedData.length} events via onmessage`);
        parsedData.forEach((singleEventData, index) => {
          const eventType = singleEventData.event_type || 'unknown_in_array';
          // If it's a recognized event_type from the expectedEventTypes, skip (already handled)
          if (expectedEventTypes.includes(eventType)) {
            console.log(`[DEBUG] Skipping duplicate SSE event on fallback 'message' for: ${eventType}`);
            return;
          }

          // Disable auto-reconnect when research is complete
          if (eventType === 'research_complete') {
            console.log('Research complete event received in onmessage handler (array) - disabling auto-reconnect');
            shouldAutoReconnect = false;
            if (cleanupRequest) cleanupRequest(); // Clean up request
            if (onComplete) onComplete(); // Call completion callback
          }

          console.log(`[DEBUG] Processing event ${index + 1}/${parsedData.length} from onmessage array with type: ${eventType}`);
          if (onEvent) {
            console.log(`[DEBUG] Calling onEvent from onmessage array for type: ${eventType}`);
            onEvent(singleEventData);
          }
        });
      } else {
        // Process single event
        const eventType = parsedData.event_type || 'unknown';

        // If it's a recognized event_type from the expectedEventTypes, skip
        if (expectedEventTypes.includes(eventType)) {
          console.log(`[DEBUG] Skipping duplicate SSE event on fallback 'message' for: ${eventType}`);
          return;
        }

        // Disable auto-reconnect when research is complete
        if (eventType === 'research_complete') {
          console.log('Research complete event received in onmessage handler (single) - disabling auto-reconnect');
          shouldAutoReconnect = false;
          if (cleanupRequest) cleanupRequest(); // Clean up request
          if (onComplete) onComplete(); // Call completion callback
        }

        console.log(`Received unnamed single message event with type: ${eventType}`);
        if (onEvent) {
          onEvent(parsedData);
        }
      }
      // --- END MODIFICATION ---

    } catch (error) {
      console.error(`Error handling SSE message: ${error}`);
      // Don't propagate parsing errors to onError
      if (!(error instanceof SyntaxError)) {
        if (onError) {
          onError(error);
        }
      }
    }
  };

  eventSource.onerror = (error) => {
    // If research is complete, ignore errors - they're expected when closing
    if (isResearchComplete) {
      console.log('Ignoring EventSource error since research is already complete');
      return;
    }

    console.error('EventSource error:', error); // Handle error
    console.error(`SSE error event received. ReadyState: ${eventSource?.readyState}`);
    // Log the full event object if possible, though it might be generic
    console.error('Full SSE error event:', error);

    // Check if we should attempt reconnection
    if (shouldAutoReconnect && !isResearchComplete) {
      console.log('Auto-reconnect is enabled, will try to reconnect...');
    } else {
      // If we're deliberately closing, don't attempt to reconnect
      if (eventSource) {
        console.log('Closing event source due to error and auto-reconnect disabled.');
        eventSource.close();
        eventSource = null;
      }
      if (staleConnectionTimer) {
        clearTimeout(staleConnectionTimer);
        staleConnectionTimer = null;
      }

      // If we're not reconnecting, and this is a terminal error, clean up the request
      if (cleanupRequest) {
        console.log('Connection terminated without reconnect - cleaning up request');
        cleanupRequest();
      }
    }
  };
};

// Generate a unique session ID
const generateSessionId = () => {
  return 'research-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
};

// Polling function for interactive research updates
let pollingInterval = null;
const pollForUpdates = (sessionId, cleanupRequest) => {
  // console.log(`[STEERING] Starting polling for session: ${sessionId}`);

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  pollingInterval = setInterval(async () => {
    try {
      // console.log(`[STEERING] Polling session status for: ${sessionId}`);
      const response = await fetch(`${API_BASE_URL}/steering/interactive/session/${sessionId}`);
      if (!response.ok) {
        console.error(`[STEERING] Failed to poll session status: ${response.status}`);
        return;
      }

      const sessionData = await response.json();
      // console.log('[STEERING] Session status:', sessionData);

      // Emit session status as events
      if (onEvent) {
        onEvent({
          event_type: 'session_status',
          data: {
            status: sessionData.status,
            current_plan: sessionData.current_plan,
            todo_version: sessionData.todo_version
          },
          timestamp: new Date().toISOString()
        });
      }

      // Check if research is complete
      if (sessionData.status === 'completed' || sessionData.status === 'error') {
        console.log('Research completed, stopping polling');
        clearInterval(pollingInterval);
        pollingInterval = null;

        isResearchComplete = true;
        shouldAutoReconnect = false;

        if (cleanupRequest) {
          cleanupRequest();
        }

        if (onComplete) {
          onComplete();
        }
      }

    } catch (error) {
      console.error('Error polling session status:', error);
    }
  }, 5000); // Poll every 5 seconds
};

// Function to send steering messages
export const sendSteeringMessage = async (message) => {
  if (!currentSessionId) {
    throw new Error('No active research session');
  }

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  try {
    // console.log(`[STEERING] Sending message to session ${currentSessionId}: ${message}`);
    const response = await fetch(`${API_BASE_URL}/steering/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        message: message
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to send steering message: ${response.status}`);
    }

    const data = await response.json();
    // console.log('[STEERING] Message sent successfully:', data);

    // Immediately poll for plan updates after sending message
    setTimeout(() => {
      pollForPlanUpdates();
    }, 1000);

    return data;

  } catch (error) {
    console.error('[STEERING] Error sending steering message:', error);
    throw error;
  }
};

// Function to get current research plan
export const getResearchPlan = async () => {
  if (!currentSessionId) {
    throw new Error('No active research session');
  }

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  try {
    const response = await fetch(`${API_BASE_URL}/steering/plan/${currentSessionId}`);

    if (!response.ok) {
      throw new Error(`Failed to get research plan: ${response.status}`);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('[STEERING] Error getting research plan:', error);
    throw error;
  }
};

// Function to get plan status for real-time updates
export const getPlanStatus = async () => {
  if (!currentSessionId) {
    return null;
  }

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  try {
    // console.log(`[STEERING] Fetching plan status for session: ${currentSessionId}`);
    const response = await fetch(`${API_BASE_URL}/steering/interactive/session/${currentSessionId}`);

    if (!response.ok) {
      // console.warn(`[STEERING] Failed to fetch plan status: ${response.status}`);
      return null; // Silently fail for status checks
    }

    const data = await response.json();
    // console.log('[STEERING] Received plan status:', {
    //   version: data.todo_version,
    //   status: data.status,
    //   planLength: data.current_plan ? data.current_plan.length : 0
    // });
    return data;

  } catch (error) {
    console.error('[STEERING] Error getting plan status:', error);
    return null;
  }
};

// Function to poll for plan updates
let planPollingInterval = null;

export const startPlanPolling = () => {
  if (planPollingInterval) {
    clearInterval(planPollingInterval);
  }

  planPollingInterval = setInterval(async () => {
    try {
      const status = await getPlanStatus();
      if (status && status.todo_version > 0) {
        // Emit plan update event for UI
        window.dispatchEvent(new CustomEvent('planStatusUpdate', { detail: status }));
      }
    } catch (error) {
      console.error('[STEERING] Polling error:', error);
    }
  }, 3000); // Poll every 3 seconds
};

export const stopPlanPolling = () => {
  if (planPollingInterval) {
    console.log('[STEERING] Stopping plan polling');
    clearInterval(planPollingInterval);
    planPollingInterval = null;
  }
};

export const pollForPlanUpdates = async () => {
  try {
    const status = await getPlanStatus();
    if (status) {
      window.dispatchEvent(new CustomEvent('planStatusUpdate', { detail: status }));
    }
  } catch (error) {
    console.error('[STEERING] Error polling for plan updates:', error);
  }
};

// Function to get current session ID
export const getCurrentSessionId = () => {
  return currentSessionId;
};

// Function to get current session todo plan
export const getCurrentTodoPlan = async () => {
  if (!currentSessionId) {
    return null;
  }

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

  try {
    console.log(`[STEERING] Getting todo plan for session: ${currentSessionId}`);
    const response = await fetch(`${API_BASE_URL}/steering/plan/${currentSessionId}`);
    if (!response.ok) {
      console.log(`[STEERING] Failed to get todo plan: ${response.status}`);
      return null;
    }

    const planData = await response.json();
    console.log('[STEERING] Got todo plan:', planData);
    return planData.plan;

  } catch (error) {
    console.error('[STEERING] Error getting todo plan:', error);
    return null;
  }
};

// Helper function to handle connection loss and reconnection attempts
const handleConnectionLoss = (cleanupFn) => {
  // Check if research is complete - if so, don't reconnect
  if (isResearchComplete) {
    console.log('Research is already complete - will not attempt to reconnect');
    return;
  }

  reconnectAttempts += 1;
  const attempt = reconnectAttempts;
  // Skip reconnection if canceled explicitly
  if (!shouldAutoReconnect) {
    console.log('Not reconnecting as auto-reconnect is disabled');

    // Clean up the request if we're not reconnecting
    if (cleanupFn) {
      console.log('Not reconnecting - cleaning up request');
      cleanupFn();
    }
    return;
  }

  const maxAttempts = 3;

  // Clear the existing event source
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Clear stale connection timer
  if (staleConnectionTimer) {
    clearTimeout(staleConnectionTimer);
    staleConnectionTimer = null;
  }

  if (attempt <= maxAttempts) {
    console.log(`Connection lost. Reconnection attempt ${attempt}/${maxAttempts} scheduled...`);

    // Set reconnecting flag
    isReconnecting = true;
    reconnectAttempts = attempt; // ensure variable stays in sync

    // Emit a reconnecting event for the UI
    if (onEvent) {
      onEvent({
        event_type: 'reconnecting',
        data: {
          attempt,
          max_attempts: maxAttempts,
          message: `Connection lost. Attempting to reconnect (${attempt}/${maxAttempts})...`,
          error: 'Connection timeout - no events received in 30 seconds'
        },
        timestamp: new Date().toISOString()
      });
    }

    // Schedule a reconnection attempt with exponential backoff
    const reconnectDelay = Math.min(1000 * Math.pow(2, attempt - 1), 8000);
    reconnectTimeout = setTimeout(() => {
      console.log(`Attempting reconnection ${attempt}/${maxAttempts}...`);

      // Important: If we have the lastResponseUrl, reconnect to it directly instead of making a new request
      if (lastResponseUrl) {
        console.log(`Reconnecting to existing stream: ${lastResponseUrl}`);
        connectToEventSource(lastResponseUrl);
      } else {
        // Schedule reconnection
        if (lastQuery) {
          console.log(`Attempting to reconnect for query: "${lastQuery}"`);
          startResearch(
            lastQuery,
            lastExtraEffort,
            lastMinimumEffort,
            lastBenchmarkMode,
            lastModelProvider,
            lastModelName,
            lastUploadedFileContent,
            onEvent,
            onComplete,
            onError,
            true,
            lastParallelEnabled,
            lastParallelMax
          );
        } else {
          console.error('No previous query to reconnect to.');
          isReconnecting = false;
        }
      }
    }, reconnectDelay);
  } else {
    console.log('Maximum reconnection attempts reached.');
    // Reset reconnecting flag and attempts
    isReconnecting = false;
    reconnectAttempts = 0;
    // Emit a reconnect_failed event for the UI
    if (onError) {
      onError(new Error(`Failed to reconnect after ${maxAttempts} attempts. Please try again.`));
    }
  }
};