const DEFAULT_API_BASE_URL = 'http://localhost:8000';
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL
  || DEFAULT_API_BASE_URL
).trim().replace(/\/+$/, '');
const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_RETRY_COUNT = 1;

export class ApiError extends Error {
  constructor(message, {
    status = null,
    data = null,
    cause = null,
    isCancelled = false,
  } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
    this.cause = cause;
    this.isCancelled = isCancelled;
  }
}

function logRequest(message, details) {
  console.info(`[api] ${message}`, details);
}

function logFailure(message, details) {
  console.error(`[api] ${message}`, details);
}

function shouldRetry(error, retriesLeft) {
  if (retriesLeft <= 0) {
    return false;
  }

  if (!(error instanceof ApiError)) {
    return true;
  }

  if (error.isCancelled) {
    return false;
  }

  return error.status === null || error.status >= 500;
}

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  const parseStartedAt = performance.now();

  if (contentType.includes('application/json')) {
    const data = await response.json();
    logRequest('response_parse_complete', {
      status: response.status,
      contentType,
      durationMs: Math.round(performance.now() - parseStartedAt),
    });
    return data;
  }

  const text = await response.text();
  logRequest('response_parse_complete', {
    status: response.status,
    contentType,
    durationMs: Math.round(performance.now() - parseStartedAt),
  });
  return text ? { message: text } : null;
}

function buildErrorMessage(data, fallbackMessage) {
  if (data && typeof data === 'object') {
    if (typeof data.error === 'string' && data.error.trim()) {
      return data.error;
    }
    if (typeof data.message === 'string' && data.message.trim()) {
      return data.message;
    }
    if (typeof data.detail === 'string' && data.detail.trim()) {
      return data.detail;
    }
  }

  return fallbackMessage;
}

export function buildApiUrl(path) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

export async function apiRequest(
  path,
  {
    method = 'GET',
    headers = {},
    body,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    retries = method === 'GET' ? DEFAULT_RETRY_COUNT : 0,
    requestId,
    signal,
  } = {},
) {
  const url = buildApiUrl(path);
  const controller = new AbortController();
  const requestStartedAt = performance.now();
  let timeoutTriggered = false;
  const timeoutId = setTimeout(() => {
    timeoutTriggered = true;
    controller.abort();
  }, timeoutMs);

  let removeAbortListener = null;
  if (signal) {
    const abortRequest = () => controller.abort(signal.reason);
    signal.addEventListener('abort', abortRequest, { once: true });
    removeAbortListener = () => signal.removeEventListener('abort', abortRequest);
  }

  const requestHeaders = {
    ...headers,
  };
  const isFormDataBody = typeof FormData !== 'undefined' && body instanceof FormData;
  if (!isFormDataBody) {
    requestHeaders['Content-Type'] = requestHeaders['Content-Type'] || 'application/json';
  }
  if (requestId) {
    requestHeaders['X-Request-ID'] = requestId;
  }

  const requestInit = {
    method,
    headers: requestHeaders,
    signal: controller.signal,
  };

  if (body !== undefined) {
    requestInit.body = isFormDataBody || typeof body === 'string' ? body : JSON.stringify(body);
  }

  logRequest('request_start', { method, path, requestId, timeoutMs, retries });

  try {
    const response = await fetch(url, requestInit);
    logRequest('response_headers_received', {
      method,
      path,
      requestId,
      status: response.status,
      durationMs: Math.round(performance.now() - requestStartedAt),
    });
    const data = await parseResponse(response);

    if (!response.ok) {
      const error = new ApiError(
        buildErrorMessage(data, `Request failed with status ${response.status}`),
        { status: response.status, data },
      );
      logFailure('request_failed', {
        method,
        path,
        requestId,
        status: response.status,
        error: error.message,
      });

      if (shouldRetry(error, retries)) {
        logRequest('request_retry', { method, path, requestId, retriesLeft: retries - 1 });
        return apiRequest(path, { method, headers, body, timeoutMs, retries: retries - 1, requestId, signal });
      }

      throw error;
    }

    logRequest('request_completed', {
      method,
      path,
      requestId,
      status: response.status,
      durationMs: Math.round(performance.now() - requestStartedAt),
    });
    return data;
  } catch (error) {
    const normalizedError = error.name === 'AbortError'
      ? timeoutTriggered
        ? new ApiError('Request timed out', { cause: error })
        : new ApiError('Request cancelled', { cause: error, isCancelled: true })
      : error instanceof ApiError
        ? error
        : new ApiError('Network request failed', { cause: error });

    if (shouldRetry(normalizedError, retries)) {
      logRequest('request_retry', {
        method,
        path,
        requestId,
        retriesLeft: retries - 1,
        reason: normalizedError.message,
      });
      return apiRequest(path, { method, headers, body, timeoutMs, retries: retries - 1, requestId, signal });
    }

    logFailure('request_exception', {
      method,
      path,
      requestId,
      error: normalizedError.message,
      durationMs: Math.round(performance.now() - requestStartedAt),
    });
    throw normalizedError;
  } finally {
    clearTimeout(timeoutId);
    removeAbortListener?.();
  }
}

export { API_BASE_URL };
