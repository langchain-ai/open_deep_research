"use client"
import { useState, useRef, useEffect, useCallback } from "react"
import InitialScreen from "./InitialScreen"
import ResearchItemList from "./ResearchItemList"
import { startResearch, setShouldAutoReconnect, cancelResearch, sendSteeringMessage, getCurrentTodoPlan, getCurrentSessionId } from "../services/researchService"

// Modern WaveText component with sophisticated animation
const WaveText = ({ children }) => (
  <span
    className="wave-text-animation"
    style={{
      background:
        "linear-gradient(90deg, rgba(71, 85, 105, 0.4) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(71, 85, 105, 0.4) 100%)",
      backgroundSize: "200% 100%",
      WebkitBackgroundClip: "text",
      backgroundClip: "text",
      color: "transparent",
      animation: "shimmer 3s ease-in-out infinite",
      fontWeight: 600,
      letterSpacing: "0.5px",
      marginLeft: "2px",
    }}
  >
    {children}
    <style>{`
      @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
    `}</style>
  </span>
)

// Helper function to generate a simple hash for a string (for code snippet ID)
const generateCodeSnippetIdentifier = (code) => {
  if (!code) return `code_empty_${Date.now()}`
  let hash = 0
  for (let i = 0; i < code.length; i++) {
    const char = code.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash |= 0 // Convert to 32bit integer
  }
  return `code_hash_${hash}`
}

// Modern CollapsibleSearchQuery component
const CollapsibleSearchQuery = ({ query }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [shouldShowToggle, setShouldShowToggle] = useState(false)
  const textRef = useRef(null)

  useEffect(() => {
    if (textRef.current) {
      const lineHeight = Number.parseInt(window.getComputedStyle(textRef.current).lineHeight)
      const maxHeight = lineHeight * 3 // 3 lines
      const actualHeight = textRef.current.scrollHeight
      setShouldShowToggle(actualHeight > maxHeight)
    }
  }, [query])

  const toggleExpansion = () => {
    setIsExpanded(!isExpanded)
  }

  return (
    <div className="px-6 py-5 border-b border-slate-200/60 bg-gradient-to-r from-[#f3f3f3]/80 via-white/50 to-blue-50/50 sticky top-0 z-20 backdrop-blur-sm">
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl border border-slate-200/60 shadow-[0_4px_20px_rgb(0,0,0,0.08)] hover:shadow-[0_4px_25px_rgb(0,0,0,0.12)] transition-all duration-300 p-5">
        <div className="relative">
          <div
            ref={textRef}
            className={`text-base font-semibold text-slate-800 transition-all duration-300 leading-relaxed ${!isExpanded && shouldShowToggle ? "overflow-hidden pr-14" : ""
              }`}
            style={{
              maxHeight: !isExpanded && shouldShowToggle ? "4.5em" : "none",
              lineHeight: "1.5em",
            }}
          >
            {query}
          </div>
          {shouldShowToggle && (
            <button
              onClick={toggleExpansion}
              className="absolute top-0 right-0 w-10 h-10 bg-slate-50/80 hover:bg-slate-100/80 border border-slate-200/60 rounded-xl flex items-center justify-center text-slate-600 hover:text-slate-700 transition-all duration-300 hover:shadow-sm hover:scale-105 backdrop-blur-sm"
              aria-label={isExpanded ? "Show less" : "Show more"}
            >
              <svg
                className={`w-4 h-4 transform transition-transform duration-300 ${isExpanded ? "rotate-180" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function ResearchPanel({
  query,
  extraEffort,
  minimumEffort,
  benchmarkMode,
  modelProvider,
  modelName,
  uploadedFileContent,
  databaseInfo,
  onBeginResearch: onBeginResearchApp,
  isResearching: isResearchingApp,
  onReportGenerated,
  onShowItemDetails,
  onShowReportDetails,
  onStopResearch,
  onTodoPlanUpdate,
}) {
  const [researchItems, setResearchItems] = useState([])
  const [localFinalReportContent, setLocalFinalReportContent] = useState("")
  const [isResearchComplete, setIsResearchComplete] = useState(false)
  const [isActiveResearch, setIsActiveResearch] = useState(false)
  const [error, setError] = useState(null)
  const [isLoadingResults, setIsLoadingResults] = useState(false)
  const [receivedActivities, setReceivedActivities] = useState({})

  // Steering-related state
  const [steeringMessage, setSteeringMessage] = useState("")
  const [currentTodoPlan, setCurrentTodoPlan] = useState("")
  const [todoVersion, setTodoVersion] = useState(0)
  const [queuedMessages, setQueuedMessages] = useState([]) // Track queued steering messages

  // Global static reference for visualization tracking that persists across component lifecycles
  if (!window.sentVisualizationIdentifiers) {
    window.sentVisualizationIdentifiers = new Set()
  }
  const visualizationTrackingRef = useRef(window.sentVisualizationIdentifiers)

  // Global static reference for code snippet tracking
  if (!window.sentCodeSnippetIdentifiers) {
    window.sentCodeSnippetIdentifiers = new Set()
  }
  const codeSnippetTrackingRef = useRef(window.sentCodeSnippetIdentifiers)

  // Debug state
  const [debugLastEvent, setDebugLastEvent] = useState(null)

  // Keep a reference to the research items to prevent issues with stale closures
  const researchItemsRef = useRef(researchItems)
  researchItemsRef.current = researchItems
  const requestInProgressRef = useRef(false)
  requestInProgressRef.current = isActiveResearch
  const previousQueryRef = useRef(query)
  previousQueryRef.current = query
  const messagesEndRef = useRef(null)
  const eventCounterRef = useRef(0)
  const activeParentItemsRef = useRef({})
  const shownSourcesRef = useRef(new Set())
  const searchQueriesRef = useRef(new Set())

  const handleStopResearch = useCallback(() => {
    setShouldAutoReconnect(false)
    cancelResearch()

    // Reset all state to initial values (complete cleanup for fresh session)
    setIsActiveResearch(false)
    setIsResearchComplete(false)
    setIsLoadingResults(false)
    setResearchItems([])
    setLocalFinalReportContent("")
    setError(null)
    setDebugLastEvent(null)
    setSteeringMessage("")
    setCurrentTodoPlan("")
    setTodoVersion(0)
    setQueuedMessages([])
    setReceivedActivities({})

    // Clear tracking references
    if (visualizationTrackingRef.current) {
      visualizationTrackingRef.current.clear()
    }
    if (codeSnippetTrackingRef.current) {
      codeSnippetTrackingRef.current.clear()
    }
    activeParentItemsRef.current = {}
    shownSourcesRef.current.clear()
    searchQueriesRef.current.clear()
    eventCounterRef.current = 0

    if (researchRequestRef.current) {
      researchRequestRef.current.inProgress = false
    }

    if (onStopResearch) {
      onStopResearch()
    }
  }, [onStopResearch])

  // Handle starting a new research (reset everything)
  const handleNewResearch = useCallback(() => {
    // Cancel any ongoing research
    setShouldAutoReconnect(false)
    cancelResearch()

    // Reset all state to initial values
    setIsActiveResearch(false)
    setIsResearchComplete(false)
    setIsLoadingResults(false)
    setResearchItems([])
    setLocalFinalReportContent("")
    setError(null)
    setDebugLastEvent(null)
    setSteeringMessage("")
    setCurrentTodoPlan("")
    setTodoVersion(0)
    setQueuedMessages([]) // Clear queued steering messages
    setReceivedActivities({})

    // Clear tracking references
    if (visualizationTrackingRef.current) {
      visualizationTrackingRef.current.clear()
    }
    if (codeSnippetTrackingRef.current) {
      codeSnippetTrackingRef.current.clear()
    }
    activeParentItemsRef.current = {}
    shownSourcesRef.current.clear()
    searchQueriesRef.current.clear()
    eventCounterRef.current = 0

    if (researchRequestRef.current) {
      researchRequestRef.current.inProgress = false
    }

    // Call onStopResearch to clear parent state
    if (onStopResearch) {
      onStopResearch()
    }
  }, [onStopResearch])

  // Listen for plan status updates from polling
  useEffect(() => {
    const handlePlanUpdate = (event) => {
      if (event.detail) {
        if (event.detail.queued_message_list !== undefined) {
          setQueuedMessages(event.detail.queued_message_list);
        }

        if (event.detail.current_plan) {
          setCurrentTodoPlan(event.detail.current_plan);
          if (onTodoPlanUpdate) {
            onTodoPlanUpdate(event.detail.current_plan);
          }
        }

        if (event.detail.todo_version !== undefined) {
          setTodoVersion(event.detail.todo_version);
        }
      }
    };

    window.addEventListener('planStatusUpdate', handlePlanUpdate);

    return () => {
      window.removeEventListener('planStatusUpdate', handlePlanUpdate);
    };
  }, [onTodoPlanUpdate]);

  // Handle sending steering messages
  const handleSendSteeringMessage = useCallback(async () => {
    if (!steeringMessage.trim() || !isActiveResearch) return;

    const messageText = steeringMessage.trim();

    // NO optimistic update - let polling handle queue display
    // This ensures UI always reflects backend state

    try {
      setSteeringMessage(""); // Clear the input immediately
      await sendSteeringMessage(messageText);

      // Polling will automatically update the queue from backend response
      // Queue will appear within 1-3 seconds and clear when processed

      // Refresh todo plan after sending message
      setTimeout(async () => {
        const todoPlan = await getCurrentTodoPlan();
        if (todoPlan) {
          setCurrentTodoPlan(todoPlan);
        }
      }, 1000);

    } catch (error) {
      console.error('Error sending steering message:', error);
      setError(`Failed to send steering message: ${error.message}`);
    }
  }, [steeringMessage, isActiveResearch]);

  // Helper function to format timestamps
  const formatTimestamp = (timestamp) => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    } catch (e) {
      console.error("Error formatting timestamp:", e)
      return "Unknown time"
    }
  }

  // Helper function to extract domain from URL
  const extractDomainFromUrl = (url) => {
    try {
      if (!url) return ""
      if (url.startsWith("http")) {
        return new URL(url).hostname
      } else if (url.includes(".")) {
        return url.split("/")[0]
      }
      return url
    } catch (e) {
      return url
    }
  }

  // Helper function to get relative time string
  const getRelativeTimeString = (timestamp) => {
    const date = new Date(timestamp)
    const now = new Date()
    const seconds = Math.round((now - date) / 1000)
    const minutes = Math.round(seconds / 60)
    const hours = Math.round(minutes / 60)
    const days = Math.round(hours / 24)

    if (seconds < 60) return `${seconds}s ago`
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    return `${days}d ago`
  }

  // Helper function to truncate strings
  const truncateString = (str, maxLength = 100) => {
    if (!str) return ""
    if (str.length <= maxLength) return str
    return str.substring(0, maxLength) + "..."
  }

  // Generate a unique ID
  const generateUniqueId = () => {
    const timestamp = Date.now()
    const counter = eventCounterRef.current++
    return `${timestamp}-${counter}`
  }

  // Helper function to add a research item
  const addResearchItem = useCallback(
    (item) => {
      const uniqueId = generateUniqueId()
      const parentId = item.parentNodeName ? activeParentItemsRef.current[item.parentNodeName] : null
      const fullItem = {
        ...item,
        id: uniqueId,
        timestamp: item.timestamp || new Date().toISOString(),
        parentId: parentId,
        children: [],
      }

      // Update active parent tracking
      if (item.type === "assistant-thinking") {
        activeParentItemsRef.current[item.nodeName] = uniqueId
      }

      setResearchItems((prevItems) => {
        researchItemsRef.current = [...prevItems]
        const newItems = [...prevItems]

        // Function to recursively find parent and add child
        const findAndAddChild = (items) => {
          for (let i = 0; i < items.length; i++) {
            if (items[i].id === parentId) {
              items[i].children.push(fullItem)
              return true
            }
            if (items[i].children && items[i].children.length > 0) {
              if (findAndAddChild(items[i].children)) {
                return true
              }
            }
          }
          return false
        }

        if (parentId) {
          if (!findAndAddChild(newItems)) {
            newItems.push(fullItem)
          }
        } else {
          newItems.push(fullItem)
        }

        researchItemsRef.current = newItems
        return newItems
      })

      // Scroll to bottom after adding an item
      setTimeout(() => {
        const container = document.getElementById("research-items-container")
        if (container) {
          container.scrollTop = container.scrollHeight
        }
      }, 100)
    },
    [setResearchItems],
  )

  // Helper function to generate a consistent identifier for visualizations based on content
  const generateVisualizationIdentifier = (imageData) => {
    if (!imageData || !imageData.src) return "unknown"

    // For base64 images, use a hash of the first part of the data
    if (imageData.src.startsWith("data:image")) {
      const base64Part = imageData.src.split(",")[1]
      if (base64Part && base64Part.length > 50) {
        return `base64_${base64Part.substring(0, 50)}`
      }
    }

    return imageData.description || imageData.filename || imageData.src.substring(0, 50)
  }

  // Helper function to extract activity text from an event for processing
  const extractActivityText = useCallback((event) => {
    if (!event) return ""

    if (event.data && event.data.activity) {
      const activity = event.data.activity
      if (!activity) return ""
      return activity.replace(/^\s*[{[].*[}\]]\s*$/, "").trim()
    }

    if (event.data) {
      if (typeof event.data.activity === "string") {
        return event.data.activity
      }

      if (
        event.data.generations &&
        Array.isArray(event.data.generations) &&
        event.data.generations[0] &&
        Array.isArray(event.data.generations[0]) &&
        event.data.generations[0][0] &&
        event.data.generations[0][0].text
      ) {
        return event.data.generations[0][0].text
      }

      if (
        event.data.generations &&
        Array.isArray(event.data.generations) &&
        event.data.generations[0] &&
        Array.isArray(event.data.generations[0]) &&
        event.data.generations[0][0] &&
        event.data.generations[0][0].message &&
        event.data.generations[0][0].message.content
      ) {
        return event.data.generations[0][0].message.content
      }

      if (event.data.data) {
        if (typeof event.data.data.activity === "string") {
          return event.data.data.activity
        }

        if (
          event.data.data.generations &&
          Array.isArray(event.data.data.generations) &&
          event.data.data.generations[0] &&
          Array.isArray(event.data.data.generations[0]) &&
          event.data.data.generations[0][0] &&
          event.data.data.generations[0][0].text
        ) {
          return event.data.data.generations[0][0].text
        }
      }

      if (event.data.output) {
        if (typeof event.data.output.activity === "string") {
          return event.data.output.activity
        }

        if (
          event.data.output.generations &&
          Array.isArray(event.data.output.generations) &&
          event.data.output.generations[0] &&
          Array.isArray(event.data.output.generations[0]) &&
          event.data.output.generations[0][0] &&
          event.data.output.generations[0][0].text
        ) {
          return event.data.output.generations[0][0].text
        }
      }
    }

    return null
  }, [])

  // Handle node_end events
  const handleNodeEndEvent = useCallback(
    (event) => {
      try {
        if (event.data && event.data.output && event.data.output.running_summary) {
          console.warn(`Removing running_summary from node_end event for ${event.data.node_name || "unknown node"}`)
          delete event.data.output.running_summary
        }

        if (event.data && event.data.node_name) {
          if (event.data.output && "source_citations" in event.data.output) {
            // Process source citations...
          }

          if (event.data && event.data.node_name === "knowledge_gap" && event.data.output) {
            // Process knowledge gaps...
          }

          if (event.data && event.data.node_name && activeParentItemsRef.current[event.data.node_name]) {
            delete activeParentItemsRef.current[event.data.node_name]
          }
        }
      } catch (error) {
        console.error("Error processing node_end event:", error)
      }
    },
    [activeParentItemsRef],
  )

  // Handle events from the research service
  const handleResearchEvent = useCallback(
    (event) => {
      setDebugLastEvent(JSON.stringify(event).substring(0, 300) + "...")
      console.log('[RESEARCH_EVENT]', event.event_type || event.type, event)

      // Handle session status events for steering
      if (event.event_type === 'session_status' && event.data) {
        if (event.data.current_plan) {
          setCurrentTodoPlan(event.data.current_plan);
          if (onTodoPlanUpdate) {
            onTodoPlanUpdate(event.data.current_plan);
          }
        }
        if (event.data.todo_version !== undefined) {
          setTodoVersion(event.data.todo_version);
        }
        return;
      }

      // Handle steering_updated events
      if (event.event_type === 'steering_updated' && event.data) {
        if (event.data.current_plan) {
          setCurrentTodoPlan(event.data.current_plan);
          if (onTodoPlanUpdate) {
            onTodoPlanUpdate(event.data.current_plan);
          }
        }
        if (event.data.todo_version !== undefined) {
          setTodoVersion(event.data.todo_version);
        }
        return;
      }

      // Frontend Check 1: Basic Event ID Deduplication
      if (event.eventId && receivedActivities[event.eventId]) {
        return
      }
      if (event.eventId) {
        setReceivedActivities((prev) => ({ ...prev, [event.eventId]: true }))
      }

      const item = null
      const parentNodeName = event.nodeData?.parent_node_name

      // Frontend Check 2: Visualization Content Deduplication for standalone images
      if (
        event.data &&
        event.data.enriched_data &&
        Array.isArray(event.data.enriched_data.images) &&
        event.data.enriched_data.images.length > 0
      ) {
        const originalImages = event.data.enriched_data.images
        const dedupedImages = []

        for (const img of originalImages) {
          const imageIdentifier = generateVisualizationIdentifier(img)
          if (visualizationTrackingRef.current.has(imageIdentifier)) {
            console.warn(
              "[handleResearchEvent] Skipping duplicate image with id:",
              imageIdentifier.length > 80 ? imageIdentifier.substring(0, 80) + "...[TRUNCATED]" : imageIdentifier,
            )
          } else {
            visualizationTrackingRef.current.add(imageIdentifier)
            dedupedImages.push(img)
          }
        }
        event.data.enriched_data.images = dedupedImages
        if (dedupedImages.length === 0 && originalImages.length > 0) {
          console.warn("[handleResearchEvent] All images in this event were duplicates; proceeding without images.")
        }
      }

      // Frontend Check 3: Code Snippet Content Deduplication
      if (
        event.data &&
        event.data.enriched_data &&
        Array.isArray(event.data.enriched_data.code_snippets) &&
        event.data.enriched_data.code_snippets.length > 0
      ) {
        const originalSnippets = event.data.enriched_data.code_snippets
        const dedupedSnippets = []

        for (const snippet of originalSnippets) {
          if (snippet && snippet.code) {
            const snippetIdentifier = generateCodeSnippetIdentifier(snippet.code)
            if (codeSnippetTrackingRef.current.has(snippetIdentifier)) {
              console.warn("[handleResearchEvent] Skipping duplicate code snippet with id:", snippetIdentifier)
            } else {
              codeSnippetTrackingRef.current.add(snippetIdentifier)
              dedupedSnippets.push(snippet)
            }
          } else {
            dedupedSnippets.push(snippet)
          }
        }
        event.data.enriched_data.code_snippets = dedupedSnippets
        if (dedupedSnippets.length === 0 && originalSnippets.length > 0) {
          console.warn(
            "[handleResearchEvent] All code snippets in this event were duplicates; proceeding without these snippets.",
          )
        }
      }

      const eventType = event.event_type || event.type

      try {
        switch (eventType) {
          case "connected":
            // If steering is enabled, fetch initial todo plan
            const sessionId = getCurrentSessionId();
            if (sessionId) {
              getCurrentTodoPlan().then(plan => {
                if (plan) {
                  setCurrentTodoPlan(plan);
                  setTodoVersion(prev => prev + 1);
                  if (onTodoPlanUpdate) {
                    onTodoPlanUpdate(plan);
                  }
                }
              }).catch(err => console.error('Failed to fetch initial todo plan:', err));
            }
            break

          case "activity_generated":
            if (event.data && event.data.activity) {
              const formattedTime = event.timestamp ? formatTimestamp(event.timestamp) : "Unknown time"

              const activity = event.data.activity
              const nodeName = event.data.node_name || "unknown"
              const relatedEventType = event.data.related_event_type || "unknown"

              let itemType = "assistant-thinking"
              let title = "Research Process"

              if (relatedEventType.includes("search")) {
                itemType = "assistant-searching"
                title = "Search Process"
              } else if (relatedEventType.includes("report")) {
                itemType = "assistant-result"
                title = "Generating Report"
              }

              const enriched = event.data.enriched_data ? { ...event.data.enriched_data } : {}
              if (Array.isArray(event.data.domains)) enriched.domains = event.data.domains
              if (event.data.domain_count) enriched.domain_count = event.data.domain_count

              addResearchItem({
                id: `activity-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`,
                title,
                content: activity,
                type: "thought-process",
                activityText: activity,
                enrichedData: enriched,
                timestamp: formattedTime,
              })
            }
            break

          case "stream_start":
            break

          case "reconnecting":
            addResearchItem({
              type: "status",
              title: "Connection Lost",
              content: event.data.message,
              level: "warning",
            })
            break

          case "reconnect_failed":
            addResearchItem({
              type: "status",
              title: "Connection Failed",
              content: event.data.message,
              level: "error",
            })
            setIsResearchComplete(true)
            break

          case "connection_interrupted":
            addResearchItem({
              type: "connection-warning",
              title: "Research Interrupted",
              timestamp: new Date().toTimeString().split(" ")[0],
              content: `Research was interrupted due to connection issues. ${event.data?.can_resume ? "Attempting to resume automatically." : "Please try again."}`,
            })
            break

          case "node_start":
            if (event.data && event.data.node_name) {
              const nodeName = event.data.node_name

              let title, content, itemType

              switch (nodeName) {
                case "query_planner":
                  title = "Planning Research Approach"
                  content = "Identifying key areas to explore for this research topic."
                  itemType = "assistant-thinking"
                  break
                case "deep_search_parallel":
                  title = "Searching for Information"
                  content = `Searching for authoritative sources${event.data.input && event.data.input.search_query ? ` on "${truncateString(event.data.input.search_query)}"` : ""}.`
                  itemType = "assistant-thinking"
                  break
                case "generate_report":
                  title = "Generating Research Report"
                  content = "Synthesizing findings into a comprehensive report."
                  itemType = "assistant-thinking"
                  break
                case "domains":
                  title = "Browsing Websites"
                  content = "Searching and browsing websites for relevant information."
                  itemType = "assistant-thinking"
                  break
                case "knowledge_gap":
                  title = "Identifying Knowledge Gaps"
                  content = "Identifying areas where additional information is needed."
                  itemType = "assistant-thinking"
                  break
                case "research_agent":
                  title = "Research Agent Active"
                  content = "The research agent is gathering information on this topic."
                  itemType = "assistant-thinking"
                  break
                case "reflect_on_report":
                  title = "Improving Report"
                  content = "Refining and improving the research report."
                  itemType = "assistant-thinking"
                  break
                case "_write":
                  title = "Compiling Information"
                  content = "Compiling and writing up gathered information."
                  itemType = "assistant-thinking"
                  break
                default:
                  title = `Process: ${nodeName}`
                  content = "Working on research task."
              }

              addResearchItem({
                title,
                timestamp: new Date().toTimeString().split(" ")[0],
                content,
                nodeData: event.data,
                type: itemType,
                nodeName: nodeName,
                eventType: "node_start",
                isFallback: true,
              })
            }
            break

          case "search_subtask":
            if (event.data) {
              const parentNodeName = event.data.parent_node || "deep_search_parallel"
              const taskType = event.data.task_type || "unknown"

              let title, content, itemType

              switch (taskType) {
                case "formulating_query":
                  title = "Formulating Search Query"
                  content = event.data.description || `Creating search query: "${event.data.query || "Unknown query"}"`
                  itemType = "assistant-thinking"
                  break
                case "analyzing_sources":
                  title = "Analyzing Sources"
                  const sourceCount = event.data.count || 0
                  content = event.data.description || `Analyzing ${sourceCount} source${sourceCount !== 1 ? "s" : ""}`
                  itemType = "assistant-analysis"
                  break
                case "processing_results":
                  title = "Processing Results"
                  const resultCount = event.data.count || 0
                  content = event.data.description || `Processing ${resultCount} result${resultCount !== 1 ? "s" : ""}`
                  itemType = "assistant-analysis"
                  break
                case "no_results":
                  title = "No Results Found"
                  content = event.data.description || `No results found for: "${event.data.query || "Unknown query"}"`
                  itemType = "assistant-result"
                  break
                default:
                  title = taskType.charAt(0).toUpperCase() + taskType.slice(1).replace(/_/g, " ")
                  content = event.data.description || "Performing research subtask"
                  itemType = "assistant-action"
              }

              if (taskType === "formulating_query" && event.data.query) {
                if (!searchQueriesRef.current.has(event.data.query)) {
                  searchQueriesRef.current.add(event.data.query)

                  addResearchItem(
                    {
                      title: "New Search",
                      timestamp: new Date().toTimeString().split(" ")[0],
                      content: `<div class="search-query-item">
                             <div class="text-sm text-slate-700 font-medium">Searching for:</div>
                             <div class="mt-1 bg-blue-50 p-2 rounded">\"${event.data.query}\"</div>
                           </div>`,
                      nodeData: { query: event.data.query },
                      type: "assistant-searching",
                    },
                    parentNodeName,
                  )
                }
              }

              addResearchItem(
                {
                  title,
                  timestamp: new Date().toTimeString().split(" ")[0],
                  content,
                  nodeData: event.data,
                  type: itemType,
                },
                parentNodeName,
              )
            }
            break

          case "search_sources_found":
          case "sources_update":
            if (event.data && event.data.sources) {
              const parentNodeName = event.data.parent_node || "deep_search_parallel"
              const uniqueSources = []

              event.data.sources.forEach((source) => {
                if (!shownSourcesRef.current.has(source.url)) {
                  shownSourcesRef.current.add(source.url)
                  uniqueSources.push(source)
                }
              })

              if (uniqueSources.length > 0) {
                addResearchItem(
                  {
                    title: "Sources Found",
                    timestamp: new Date().toTimeString().split(" ")[0],
                    content: `<div class="text-sm text-slate-700 mb-1">Found ${uniqueSources.length} new source${uniqueSources.length !== 1 ? "s" : ""} to analyze:</div>
                          <div class="source-list border-l-4 border-blue-200 pl-3 py-1">
                            ${uniqueSources
                        .map((source) => {
                          const domain = source.url ? new URL(source.url).hostname : "Unknown"
                          const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`
                          return `<div class="source-item p-1 hover:bg-slate-100 rounded flex items-center gap-2">
                                      <img src="${faviconUrl}" class="w-5 h-5" alt="${domain}" />
                                      <a href="${source.url}" target="_blank" class="text-blue-600 hover:underline overflow-hidden text-ellipsis">${source.title || source.url}</a>
                                    </div>`
                        })
                        .join("")}
                          </div>`,
                    nodeData: { sources: uniqueSources },
                    type: "assistant-result",
                  },
                  parentNodeName,
                )
              }
            }
            break

          case "node_end":
            if (event.data && event.data.node_name) {
              const nodeName = event.data.node_name
              console.log(`Node end: ${nodeName}`, event.data)

              handleNodeEndEvent(event)

              if (activeParentItemsRef.current[nodeName]) {
                delete activeParentItemsRef.current[nodeName]
              }

              if (nodeName === "generate_report" || nodeName === "reflect_on_report") {
                const reportOutput = event.data.output?.report_output
                if (reportOutput) {
                  setLocalFinalReportContent(reportOutput)
                  setIsResearchComplete(true)
                  setIsLoadingResults(false)

                  addResearchItem({
                    id: "final-report-link-" + Date.now(),
                    type: "final-report-link",
                    title: "View Full Research Report",
                    content: `
                    <div class="flex flex-col items-start mt-2">
                      <div class="flex items-center gap-3 mb-3">
                        <div class="w-8 h-8 bg-slate-100 border border-slate-200 text-slate-700 rounded-full flex items-center justify-center shadow-sm">
                          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                          </svg>
                        </div>
                        <span class="font-medium text-slate-800">Research Complete</span>
                      </div>
                      <button class="view-report-button flex items-center gap-2 px-4 py-2 bg-slate-50 border border-slate-200 text-slate-700 rounded-lg shadow-sm hover:bg-slate-100 hover:border-slate-300 transition-all duration-200 group">
                        <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                        </svg>
                        <span class="font-medium">View Full Report</span>
                        <svg class="w-3.5 h-3.5 text-slate-400 transform group-hover:translate-x-0.5 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                        </svg>
                      </button>
                    </div>
                  `,
                    timestamp: event.timestamp
                      ? formatTimestamp(event.timestamp)
                      : formatTimestamp(new Date().toISOString()),
                  })

                  if (onReportGenerated) {
                    onReportGenerated(reportOutput)
                  }
                }
              }
            }
            break

          case "research_complete":
            console.log("Received research complete event:", event)

            setIsActiveResearch(false)
            setIsResearchComplete(true)
            setIsLoadingResults(false)

            setQueuedMessages([])

            // Fetch final task status to update todo plan with all completed tasks
            const fetchFinalTodoStatus = async () => {
              try {
                const { getCurrentSessionId, pollForPlanUpdates } = await import('../services/researchService')
                const sessionId = getCurrentSessionId()
                if (sessionId) {
                  await pollForPlanUpdates()
                }
              } catch (error) {
                console.error("[COMPLETION] Error fetching final todo status:", error)
              }
            }
            fetchFinalTodoStatus()

            if (event.data) {
              const reportContent = event.data.report || event.data.summary
              if (reportContent) {
                setLocalFinalReportContent(reportContent)

                addResearchItem({
                  id: "final-report-link-" + Date.now(),
                  type: "final-report-link",
                  title: "View Full Research Report",
                  content: `
                  <div class="flex flex-col items-start mt-2">
                    <div class="flex items-center gap-3 mb-3">
                      <div class="w-8 h-8 bg-slate-100 border border-slate-200 text-slate-700 rounded-full flex items-center justify-center shadow-sm">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                      </div>
                      <span class="font-medium text-slate-800">Research Complete - All Tasks Finished</span>
                    </div>
                    <button class="view-report-button flex items-center gap-2 px-4 py-2 bg-slate-50 border border-slate-200 text-slate-700 rounded-lg shadow-sm hover:bg-slate-100 hover:border-slate-300 transition-all duration-200 group">
                      <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                      </svg>
                      <span class="font-medium">View Full Report</span>
                      <svg class="w-3.5 h-3.5 text-slate-400 transform group-hover:translate-x-0.5 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                      </svg>
                    </button>
                  </div>
                `,
                  timestamp: event.timestamp
                    ? formatTimestamp(event.timestamp)
                    : formatTimestamp(new Date().toISOString()),
                })

                if (onReportGenerated) {
                  // Pass analysis data if available from the research_complete event
                  const analysisInfo = event.data.analysis_required ? {
                    analysis_required: event.data.analysis_required,
                    analysis_report_path: event.data.analysis_report_path,
                    analysis_charts: event.data.analysis_charts || [],
                    analysis_output: event.data.analysis_output || '',
                  } : null;
                  onReportGenerated(reportContent, analysisInfo)
                }
              } else {
                console.warn("Research_complete event received without report content")
              }
            }

            try {
              setShouldAutoReconnect(false)

              setTimeout(() => {
                setIsActiveResearch(false)
                setIsResearchComplete(true)
                setIsLoadingResults(false)

                if (window.disconnectResearchEventSource) {
                  window.disconnectResearchEventSource()
                }
              }, 500)
            } catch (err) {
              console.error("Error during research completion handling:", err)
            }
            break

          case "error":
            setError(event.error || "An unknown error occurred")
            setIsLoadingResults(false)
            setIsActiveResearch(false)
            addResearchItem({
              type: "status",
              title: "Error During Research",
              content: event.error || "An unexpected error stopped the research process.",
              level: "error",
            })
            break

          default:
            if (event.data) {
              const formattedTime = event.timestamp ? formatTimestamp(event.timestamp) : "Unknown time"
              addResearchItem({
                id: `event-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`,
                title: `Event: ${eventType || "Unknown"}`,
                content: JSON.stringify(event.data, null, 2),
                type: "status",
                timestamp: formattedTime,
              })
            }
        }
      } catch (err) {
        console.error("Error processing research event:", err)
        setError(`Error processing event: ${err.message}`)
      }
    },
    [
      setDebugLastEvent,
      receivedActivities,
      setReceivedActivities,
      addResearchItem,
      handleNodeEndEvent,
      setIsResearchComplete,
      setIsActiveResearch,
      setIsLoadingResults,
      setLocalFinalReportContent,
      onReportGenerated,
      setError,
      setCurrentTodoPlan,
      setTodoVersion,
      onTodoPlanUpdate,
    ],
  )

  // Handle research errors
  const handleResearchError = useCallback(
    (error) => {
      console.error("Research error:", error)
      const errorItem = {
        type: "error",
        title: "Error during research",
        content: error.message,
        timestamp: new Date().toISOString(),
      }
      addResearchItem(errorItem)
      setError(error.message)
      setIsLoadingResults(false)
      setIsActiveResearch(false)

      if (researchRequestRef.current) {
        researchRequestRef.current.inProgress = false
      }
    },
    [addResearchItem, setError, setIsLoadingResults, setIsActiveResearch],
  )

  const researchRequestRef = useRef({
    inProgress: false,
    lastQueryTime: 0,
    queryHash: "",
  })

  // Generate hash of query parameters to detect duplicate requests
  const getQueryHash = useCallback((q, extra, min, benchmark, provider, model, fileContent) => {
    return `${q}:${extra}:${min}:${benchmark}:${provider}:${model}:${fileContent ? "file_present" : "no_file"}`
  }, [])

  const onBeginResearchInternal = useCallback(
    (q, extra, min, benchmark, modelCfg, fileContent, databaseInfo) => {
      // Clear all data
      setResearchItems([])
      setLocalFinalReportContent("")
      setIsResearchComplete(false)
      setError(null)
      setIsLoadingResults(true)
      setIsActiveResearch(true)

      // Clear refs
      researchItemsRef.current = []
      shownSourcesRef.current.clear()
      searchQueriesRef.current.clear()
      activeParentItemsRef.current = {}
      eventCounterRef.current = 0
      visualizationTrackingRef.current.clear()
      codeSnippetTrackingRef.current.clear()

      onBeginResearchApp(q, extra, min, benchmark, modelCfg, fileContent, databaseInfo)
    },
    [
      onBeginResearchApp,
      setResearchItems,
      setLocalFinalReportContent,
      setIsResearchComplete,
      setError,
      setIsActiveResearch,
      setIsLoadingResults,
    ],
  )

  const handleItemClick = useCallback(
    (item) => {
      if (item.type === "final-report-link") {
        if (localFinalReportContent) {
          onShowReportDetails(localFinalReportContent)
        } else {
          console.warn("Attempted to show report, but no report content is available.")
        }
      } else {
        onShowItemDetails(item)
      }
    },
    [localFinalReportContent, onShowReportDetails, onShowItemDetails],
  )

  const handleStartResearch = useCallback(() => {
    setError(null)

    startResearch(
      query,
      extraEffort,
      minimumEffort,
      benchmarkMode,
      modelProvider,
      modelName,
      uploadedFileContent,
      databaseInfo,
      handleResearchEvent,
      () => {
        setIsResearchComplete(true)
        setIsLoadingResults(false)
        setIsActiveResearch(false)
        if (researchRequestRef.current) {
          researchRequestRef.current.inProgress = false
        }
      },
      (error) => {
        if (researchRequestRef.current) {
          researchRequestRef.current.inProgress = false
        }
        handleResearchError(error)
        setIsActiveResearch(false)
      },
      true, // Enable steering
      true, // Enable parallel search
      2 // Parallel max concurrency
    )
  }, [
    query,
    extraEffort,
    minimumEffort,
    benchmarkMode,
    modelProvider,
    modelName,
    uploadedFileContent,
    handleResearchEvent,
    handleResearchError,
    setIsResearchComplete,
    setError,
    setIsLoadingResults,
  ])

  // Start research effect: trigger research when isResearching becomes true
  useEffect(() => {
    if (isResearchComplete) {
      return
    }

    if (isResearchingApp && query && !researchRequestRef.current.inProgress) {
      visualizationTrackingRef.current.clear()
      codeSnippetTrackingRef.current.clear()
    }

    const currentQueryHash = getQueryHash(
      query,
      extraEffort,
      minimumEffort,
      benchmarkMode,
      modelProvider,
      modelName,
      uploadedFileContent,
    )
    const lastQueryHash = researchRequestRef.current.queryHash
    const now = Date.now()
    const timeSinceLastRequest = now - researchRequestRef.current.lastQueryTime
    const DEBOUNCE_TIME = 5000

    if (researchRequestRef.current.inProgress) {
      return
    }

    if (currentQueryHash === lastQueryHash && timeSinceLastRequest < DEBOUNCE_TIME) {
      return
    }

    if (isResearchingApp && query) {
      researchRequestRef.current = {
        inProgress: true,
        lastQueryTime: now,
        queryHash: currentQueryHash,
      }

      handleStartResearch()
    }
  }, [
    isResearchingApp,
    query,
    extraEffort,
    minimumEffort,
    benchmarkMode,
    modelProvider,
    modelName,
    uploadedFileContent,
    handleStartResearch,
    getQueryHash,
    isResearchComplete,
  ])

  // Helper function to format model name for display
  const getFormattedModelName = (provider, modelName) => {
    if (!provider && !modelName) return "Unknown Model"
    if (provider === "azure") return "Azure OpenAI"
    if (provider === "gemini") return modelName || "Gemini 2.5 Flash"
    return modelName || provider
  }

  // Helper function to get model icon
  const getModelIcon = (provider, modelName) => {
    if (provider === "azure") {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M13.05 4.24L6.56 18.05h2.8l1.2-2.6h5.88l1.2 2.6h2.8L13.95 4.24h-.9zM11.36 13.45l2.14-4.63 2.14 4.63h-4.28z" fill="#0078D4"/>
        </svg>
      )
    } else if (provider === "gemini") {
      return (
        <svg width="16" height="16" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M14.0001 0L17.5001 10.5L28.0001 14L17.5001 17.5L14.0001 28L10.5001 17.5L0.000061 14L10.5001 10.5L14.0001 0Z" fill="url(#paint0_linear_gemini)" />
          <defs>
            <linearGradient id="paint0_linear_gemini" x1="0.000061" y1="14" x2="28.0001" y2="14" gradientUnits="userSpaceOnUse">
              <stop stopColor="#8E54E9" />
              <stop offset="1" stopColor="#4776E6" />
            </linearGradient>
          </defs>
        </svg>
      )
    }
    return null
  }

  return (
    <div className="h-full w-full">
      {!isResearchingApp && !isActiveResearch && researchItems.length === 0 ? (
        <InitialScreen onBeginResearch={onBeginResearchInternal} />
      ) : (
        <div className="flex flex-col h-full w-full bg-gradient-to-br from-blue-50 via-white to-[#f3f3f3] min-h-0">
          {/* User query display - modern header */}
          <CollapsibleSearchQuery query={query} />

          {/* Main content with enhanced styling - add bottom padding for controls */}
          <div
            id="research-items-container"
            className="flex-1 min-h-0 overflow-y-auto px-4 md:px-6 py-4 md:py-6 pb-0 space-y-4 md:space-y-6"
          >
            {/* AI response with research steps */}
            <div>
              <div className="space-y-3">
                {/* Enhanced Benchmark Mode Header */}
                {benchmarkMode && (
                  <div className="bg-gradient-to-r from-blue-50/80 to-[#e3f3ff] backdrop-blur-sm border border-[#0176d3]/30 rounded-2xl p-6 mb-6 shadow-sm hover:shadow-md transition-all duration-300">
                    <div className="flex items-center space-x-4">
                      <div className="p-3 bg-[#0176d3] rounded-xl shadow-lg">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2.5}
                            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-bold text-slate-900 mb-1">Benchmark Mode Active</h3>
                        <p className="text-sm text-slate-600 font-medium">
                          Generating focused Q&A response with confidence assessment
                        </p>
                      </div>
                      <div className="ml-auto">
                        <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-800 rounded-xl text-sm font-semibold shadow-sm">
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path
                              fillRule="evenodd"
                              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                              clipRule="evenodd"
                            />
                          </svg>
                          Q&A Mode
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <ResearchItemList items={researchItems} selectedItemId={null} onItemClick={handleItemClick} />
                {isActiveResearch && (
                  <div className="mt-4 text-sm flex items-center gap-3 py-3 px-4 text-slate-700 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/60 shadow-sm">
                    <svg width="32" height="32" viewBox="0 0 50 50" className="flex-shrink-0">
                      <defs>
                        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" style={{ stopColor: "#64748b", stopOpacity: 0.8 }} />
                          <stop offset="100%" style={{ stopColor: "#475569", stopOpacity: 1 }} />
                        </linearGradient>
                      </defs>
                      <g>
                        <rect
                          x="15"
                          y="15"
                          width="20"
                          height="20"
                          fill="url(#grad)"
                          rx="3"
                          stroke="#334155"
                          strokeWidth="0.5"
                          strokeOpacity="0.3"
                          transform="rotate(45 25 25)"
                        >
                          <animateTransform
                            attributeName="transform"
                            type="rotate"
                            from="0 25 25"
                            to="360 25 25"
                            dur="3s"
                            repeatCount="indefinite"
                            additive="sum"
                          ></animateTransform>
                          <animate
                            attributeName="opacity"
                            values="0.9;0.6;0.9"
                            dur="2s"
                            repeatCount="indefinite"
                          ></animate>
                        </rect>
                        <rect
                          x="15"
                          y="15"
                          width="20"
                          height="20"
                          fill="none"
                          rx="4"
                          stroke="#64748b"
                          strokeWidth="0.5"
                          strokeOpacity="0.2"
                          transform="rotate(-45 25 25)"
                        >
                          <animateTransform
                            attributeName="transform"
                            type="rotate"
                            from="0 25 25"
                            to="-360 25 25"
                            dur="4s"
                            repeatCount="indefinite"
                            additive="sum"
                          ></animateTransform>
                        </rect>
                      </g>
                    </svg>
                    <WaveText>
                      <span style={{ animationDelay: "0s" }}>{benchmarkMode ? "A" : "R"}</span>
                      <span style={{ animationDelay: "0.1s" }}>{benchmarkMode ? "n" : "e"}</span>
                      <span style={{ animationDelay: "0.2s" }}>{benchmarkMode ? "a" : "s"}</span>
                      <span style={{ animationDelay: "0.3s" }}>{benchmarkMode ? "l" : "e"}</span>
                      <span style={{ animationDelay: "0.4s" }}>{benchmarkMode ? "y" : "a"}</span>
                      <span style={{ animationDelay: "0.5s" }}>{benchmarkMode ? "z" : "r"}</span>
                      <span style={{ animationDelay: "0.6s" }}>{benchmarkMode ? "i" : "c"}</span>
                      <span style={{ animationDelay: "0.7s" }}>{benchmarkMode ? "n" : "h"}</span>
                      <span style={{ animationDelay: "0.8s" }}>{benchmarkMode ? "g" : "i"}</span>
                      <span style={{ animationDelay: "0.9s" }}>{benchmarkMode ? "." : "n"}</span>
                      <span style={{ animationDelay: "1.0s" }}>{benchmarkMode ? "." : "g"}</span>
                      <span style={{ animationDelay: "1.1s" }}>{benchmarkMode ? "." : "."}</span>
                      <span style={{ animationDelay: "1.2s" }}>.</span>
                      <span style={{ animationDelay: "1.3s" }}>.</span>
                    </WaveText>
                  </div>
                )}
                <div ref={messagesEndRef} className="pb-4" />
              </div>
            </div>
          </div>

          {/* Enhanced Footer with modern styling - always visible */}
          <div className="flex-shrink-0 border-t border-slate-200/60 px-4 md:px-5 py-2.5 md:py-3 bg-gradient-to-r from-[#f3f3f3]/80 via-white/50 to-blue-50/50 backdrop-blur-sm shadow-[0_-4px_20px_rgb(0,0,0,0.06)]">
            <div className="max-w-full mx-auto">
              {isResearchComplete ? (
                // Show only New Research button when complete
                <div className="flex justify-center">
                  <button
                    className="px-6 py-3 bg-[#0176d3] text-white rounded-xl hover:bg-[#014486] transition-all duration-300 font-semibold shadow-[0_4px_12px_rgba(1,118,211,0.3)] hover:shadow-[0_6px_16px_rgba(1,118,211,0.4)] hover:scale-105 active:scale-95 flex items-center gap-2"
                    onClick={handleNewResearch}
                    aria-label="Start New Research"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 4v16m8-8H4"
                      />
                    </svg>
                    New Research
                  </button>
                </div>
              ) : (
                // Show full controls during active research
                <div className="space-y-3 max-w-full">
                  {/* Queued Messages Display */}
                  {queuedMessages.length > 0 && (
                    <div className="space-y-2 max-w-full overflow-x-auto">
                      {queuedMessages.map((msg, index) => (
                        <div
                          key={index}
                          className="flex items-center gap-3 px-4 py-2.5 bg-gradient-to-r from-amber-50 to-orange-50/50 border border-amber-200/60 rounded-xl shadow-[0_2px_8px_rgb(245,158,11,0.15)] hover:shadow-[0_2px_12px_rgb(245,158,11,0.25)] backdrop-blur-sm transition-all duration-300 min-w-0"
                        >
                          <div className="flex items-center justify-center w-5 h-5 flex-shrink-0">
                            <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                          <span className="text-sm font-semibold text-slate-700 flex-1 truncate">{msg}</span>
                          <span className="text-xs text-amber-700 font-bold uppercase tracking-wide bg-gradient-to-r from-amber-100 to-orange-100 px-3 py-1 rounded-lg shadow-sm flex-shrink-0">Queued</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Input Controls */}
                  <div className="flex items-center gap-2 md:gap-3 max-w-full">
                    <input
                      type="text"
                      placeholder="Send steering message..."
                      className="flex-1 min-w-0 rounded-xl border-2 border-slate-200/60 px-3 md:px-4 py-3 bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-[#0176d3]/50 focus:border-[#0176d3] transition-all duration-300 placeholder-slate-400 font-medium shadow-[0_2px_8px_rgb(0,0,0,0.06)] hover:shadow-[0_2px_12px_rgb(0,0,0,0.08)] focus:shadow-[0_2px_12px_rgba(1,118,211,0.15)]"
                      value={steeringMessage}
                      onChange={(e) => setSteeringMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSendSteeringMessage()}
                      disabled={!isActiveResearch}
                    />

                    <button
                      className={`p-3 rounded-xl transition-all duration-300 flex-shrink-0 ${!isActiveResearch || !steeringMessage.trim() ? "text-slate-400 cursor-not-allowed bg-slate-100/50 shadow-sm" : "bg-[#0176d3] text-white hover:bg-[#014486] shadow-[0_4px_12px_rgba(1,118,211,0.3)] hover:shadow-[0_6px_16px_rgba(1,118,211,0.4)] hover:scale-105 active:scale-95"}`}
                      onClick={handleSendSteeringMessage}
                      disabled={!isActiveResearch || !steeringMessage.trim()}
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2.5}
                          d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                        />
                      </svg>
                    </button>
                    {isActiveResearch && (
                      <button
                        className="px-4 md:px-5 py-3 bg-red-600 text-white rounded-xl hover:bg-red-700 transition-all duration-300 font-semibold shadow-[0_4px_12px_rgba(220,38,38,0.3)] hover:shadow-[0_6px_16px_rgba(220,38,38,0.4)] hover:scale-105 active:scale-95 flex-shrink-0"
                        onClick={handleStopResearch}
                        aria-label="Stop Research"
                      >
                        Stop
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ResearchPanel
