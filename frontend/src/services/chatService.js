import { apiRequest, buildApiUrl } from '../api/client';
import { DEFAULT_WORKSPACE_NAME, normalizeWorkspaceName } from '../config/workspace';


function normalizeChatResponse(payload) {
  const data = payload?.data && typeof payload.data === 'object'
    ? payload.data
    : payload;

  if (!data || typeof data !== 'object') {
    return {
      answer: '',
      sources: [],
      summary: null,
      workspace: null,
      request_id: null,
      success: false,
    };
  }

  return {
    ...data,
    answer: typeof data.answer === 'string' ? data.answer : '',
    sources: Array.isArray(data.sources) ? data.sources : [],
    summary: data.summary ?? null,
    workspace: data.workspace ?? null,
    request_id: data.request_id ?? null,
    success: data.success ?? true,
  };
}

export async function askQuestion(
  question,
  workspaceName = null,
  {
    requestId,
    signal,
    history = [],
    conversationId = null,
  } = {},
) {
  const sanitizedQuestion = (question || '').trim();
  const sanitizedWorkspaceName = normalizeWorkspaceName(workspaceName);
  const storedWorkspaceName = normalizeWorkspaceName(localStorage.getItem('workspace'));
  const effectiveWorkspaceName = sanitizedWorkspaceName || storedWorkspaceName || DEFAULT_WORKSPACE_NAME;

  console.info('[chatService] askQuestion', {
    endpoint: buildApiUrl('/chat'),
    requestId,
    hasQuestion: Boolean(sanitizedQuestion),
    workspaceName: sanitizedWorkspaceName || null,
    storedWorkspaceName: storedWorkspaceName || null,
    effectiveWorkspaceName,
    historyTurns: history.length,
    conversationId,
  });

  if (!sanitizedQuestion) {
    throw new Error('Question is required');
  }

  try {
    const response = await apiRequest('/chat', {
      method: 'POST',
      body: {
        question: sanitizedQuestion,
        workspace_name: effectiveWorkspaceName,
        conversation_id: conversationId,
        history,
      },
      timeoutMs: 100000,
      retries: 0,
      requestId,
      signal,
    });
    return normalizeChatResponse(response);
  } catch (error) {
    console.error('[chatService] askQuestion_failed', {
      endpoint: buildApiUrl('/chat'),
      requestId,
      error: error.message,
    });
    throw new Error(error.message || 'Failed to get chat response');
  }
}
