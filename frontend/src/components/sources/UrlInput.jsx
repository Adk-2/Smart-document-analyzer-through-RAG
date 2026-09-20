import { useState } from 'react';

import { useAppStore } from '../../store/appStore';


function validateUrlInput(value) {
  const sanitizedValue = value.trim();
  if (!sanitizedValue) {
    return 'URL is required';
  }

  try {
    const parsedUrl = new URL(sanitizedValue);
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      return 'Invalid URL';
    }
  } catch {
    return 'Invalid URL';
  }

  return null;
}


export default function UrlInput() {
  const [urlInput, setUrlInput] = useState('');
  const isLoading = useAppStore((state) => state.loading.ingestion);
  const ingestionStatus = useAppStore((state) => state.ingestionStatus);
  const ingestionError = useAppStore((state) => state.errors.ingestion);
  const ingestUrl = useAppStore((state) => state.ingestUrl);
  const setIngestionFeedback = useAppStore((state) => state.setIngestionFeedback);

  const handleSubmit = async (event) => {
    event.preventDefault();

    const validationError = validateUrlInput(urlInput);
    if (validationError) {
      setIngestionFeedback(validationError, validationError);
      return;
    }

    const result = await ingestUrl(urlInput);
    if (result.success) {
      setUrlInput('');
    }
  };

  const statusColor = ingestionError
    ? '#ff8e8e'
    : isLoading
      ? '#8fa4d6'
      : '#7ea4ff';

  return (
    <>
      <form className="url-input" onSubmit={handleSubmit}>
        <input
          type="url"
          placeholder="Paste a source URL"
          value={urlInput}
          onChange={(event) => setUrlInput(event.target.value)}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !urlInput.trim()}
          style={{ minWidth: '60px' }}
        >
          {isLoading ? 'Indexing...' : 'Add'}
        </button>
      </form>
      {ingestionStatus && (
        <span style={{ fontSize: '0.75rem', color: statusColor }}>
          {ingestionStatus}
        </span>
      )}
    </>
  );
}
