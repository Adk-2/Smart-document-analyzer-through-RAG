import { useCallback, useEffect, useRef, useState } from "react";
import Header from "../components/layout/Header";
import ErrorBoundary from "../components/common/ErrorBoundary";
import Sidebar from "../components/layout/Sidebar";
import ChatArea from "../components/chat/ChatArea";
import InsightsPanel from "../components/insights/InsightsPanel";
import { askQuestion } from "../services/chatService";
import { useAppStore } from "../store/appStore";


export default function Dashboard() {
  const loadSources = useAppStore((state) => state.loadSources);
  const activeWorkspace = useAppStore((state) => state.activeWorkspace);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const messageIdRef = useRef(0);
  const requestCounterRef = useRef(0);
  const conversationIdRef = useRef(`conversation-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const activeRequestIdRef = useRef(null);
  const activeAbortControllerRef = useRef(null);
  const isMountedRef = useRef(true);
  const previousWorkspaceRef = useRef(activeWorkspace);
  const removeLoadingMessages = useCallback(
    (currentMessages) => currentMessages.filter((message) => !message?.isLoading),
    [],
  );
  const logChatLifecycle = useCallback((event, details) => {
    console.info(`[chat] ${event}`, details);
  }, []);
  const updateLoadingState = useCallback((isLoading, requestId, reason) => {
    logChatLifecycle('loading_state_changed', { requestId, isLoading, reason });
    setIsSendingMessage(isLoading);
  }, [logChatLifecycle]);
  const clearPendingRequest = useCallback((requestId, abortController) => {
    if (activeAbortControllerRef.current === abortController) {
      activeAbortControllerRef.current = null;
    }
    if (activeRequestIdRef.current === requestId) {
      activeRequestIdRef.current = null;
    }
    if (isMountedRef.current && activeRequestIdRef.current === null) {
      updateLoadingState(false, requestId, "request_finished");
    }
  }, [updateLoadingState]);

  useEffect(() => {
    loadSources();
  }, [loadSources]);

  useEffect(() => {
    isMountedRef.current = true;

    return () => {
      isMountedRef.current = false;
      activeAbortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (activeWorkspace) {
      localStorage.setItem(
        "workspace",
        activeWorkspace,
      );
    }
  }, [activeWorkspace]);

  useEffect(() => {
    if (previousWorkspaceRef.current === activeWorkspace) {
      return;
    }

    previousWorkspaceRef.current = activeWorkspace;

    if (!activeRequestIdRef.current) {
      return;
    }

    activeAbortControllerRef.current?.abort();
    activeAbortControllerRef.current = null;
    activeRequestIdRef.current = null;
    setMessages((currentMessages) => removeLoadingMessages(currentMessages));
    updateLoadingState(false, null, "workspace_changed");
  }, [activeWorkspace, removeLoadingMessages, updateLoadingState]);

  const createMessage = useCallback((role, content, extra = {}) => {
    messageIdRef.current += 1;
    return {
      id: `${role}-${messageIdRef.current}`,
      role,
      content,
      ...extra,
    };
  }, []);

  const buildConversationHistory = useCallback(() => (
    messages
      .filter((message) => (
        ['user', 'assistant'].includes(message.role)
        && !message.isLoading
        && !message.isError
        && typeof message.content === 'string'
        && message.content.trim()
      ))
      .slice(-10)
      .map((message) => ({
        role: message.role,
        content: message.content,
      }))
  ), [messages]);

  const handleSendMessage = useCallback(async (question) => {
    const sanitizedQuestion = question.trim();
    if (!sanitizedQuestion || activeRequestIdRef.current) {
      return;
    }

    activeAbortControllerRef.current?.abort();
    const abortController = new AbortController();
    requestCounterRef.current += 1;
    const requestId = `chat-${Date.now()}-${requestCounterRef.current}`;

    const userMessage = createMessage("user", sanitizedQuestion);
    const loadingMessage = createMessage("assistant", "Thinking...", {
      isLoading: true,
    });
    const loadingMessageId = loadingMessage.id;

    activeRequestIdRef.current = requestId;
    activeAbortControllerRef.current = abortController;
    logChatLifecycle("request_started", {
      requestId,
      workspaceName: activeWorkspace,
      questionLength: sanitizedQuestion.length,
    });

    setMessages((currentMessages) => [
      ...removeLoadingMessages(currentMessages),
      userMessage,
      loadingMessage,
    ]);
    setChatInput("");
    updateLoadingState(true, requestId, "request_started");

    try {
      const response = await askQuestion(
        sanitizedQuestion,
        activeWorkspace,
        {
          requestId,
          signal: abortController.signal,
          conversationId: conversationIdRef.current,
          history: buildConversationHistory(),
        },
      );
      if (!isMountedRef.current || activeRequestIdRef.current !== requestId) {
        return;
      }
      logChatLifecycle("request_completed", {
        requestId,
        workspaceName: response.workspace || activeWorkspace,
        sourceCount: Array.isArray(response.sources) ? response.sources.length : 0,
      });

      const assistantMessage = createMessage(
        "assistant",
        response.answer || "No answer returned.",
        {
          sources: Array.isArray(response.sources) ? response.sources : [],
          summary: response.summary || null,
        },
      );

      setMessages((currentMessages) => removeLoadingMessages(
        currentMessages.map((message) => (
          message.id === loadingMessageId ? assistantMessage : message
        )),
      ));
    } catch (error) {
      if (
        !isMountedRef.current
        || activeRequestIdRef.current !== requestId
        || abortController.signal.aborted
      ) {
        if (isMountedRef.current) {
          setMessages((currentMessages) => removeLoadingMessages(currentMessages));
        }
        logChatLifecycle("request_cancelled", { requestId });
        clearPendingRequest(requestId, abortController);
        return;
      }
      logChatLifecycle("request_failed", {
        requestId,
        error: error.message || "Unknown chat error",
      });

      const errorMessage = createMessage(
        "assistant",
        error.message || "Unable to answer right now.",
        { isError: true },
      );

      setMessages((currentMessages) => removeLoadingMessages(
        currentMessages.map((message) => (
          message.id === loadingMessageId ? errorMessage : message
        )),
      ));
    } finally {
      clearPendingRequest(requestId, abortController);
    }
  }, [
    activeWorkspace,
    buildConversationHistory,
    clearPendingRequest,
    createMessage,
    logChatLifecycle,
    removeLoadingMessages,
    updateLoadingState,
  ]);

  return (
    <>
      <Header />
      <div className="workspace">
        <ErrorBoundary label="Sidebar" className="sidebar">
          <Sidebar />
        </ErrorBoundary>
        <ErrorBoundary label="ChatArea" className="chat-area">
          <ChatArea
            input={chatInput}
            isThinking={isSendingMessage}
            messages={messages}
            onInputChange={setChatInput}
            onSend={handleSendMessage}
          />
        </ErrorBoundary>
        <ErrorBoundary label="InsightsPanel" className="insights-panel">
          <InsightsPanel />
        </ErrorBoundary>
      </div>
    </>
  );
}
