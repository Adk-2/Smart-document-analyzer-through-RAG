import { memo, useMemo } from 'react';
import ChatInput from './ChatInput';

function getUrlDetails(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }

  try {
    const url = new URL(value);
    const segments = url.pathname.split('/').filter(Boolean);
    const filename = segments.length > 0 ? segments[segments.length - 1] : '';

    return {
      domain: url.hostname.replace(/^www\./, ''),
      filename,
    };
  } catch {
    return null;
  }
}

function getFilename(source, metadata, urlDetails) {
  const candidates = [
    metadata?.source_file,
    metadata?.filename,
    metadata?.file_name,
    metadata?.title,
    source,
    urlDetails?.filename,
  ];

  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || !candidate.trim()) {
      continue;
    }

    const normalized = candidate.replace(/\\/g, '/');
    const segments = normalized.split('/').filter(Boolean);
    const value = segments[segments.length - 1] || normalized;
    if (value) {
      return value;
    }
  }

  return 'Source';
}

function getPageLabel(source, metadata) {
  const rawPage = source?.page
    ?? metadata?.page
    ?? metadata?.page_number
    ?? metadata?.pageNumber;

  if (rawPage === undefined || rawPage === null || rawPage === '' || rawPage === '?') {
    return null;
  }

  const pageNumber = Number(rawPage);
  if (Number.isFinite(pageNumber)) {
    return `p.${pageNumber + 1}`;
  }

  return `p.${rawPage}`;
}

function formatCitation(source, index) {
  const metadata = source?.metadata && typeof source.metadata === 'object'
    ? source.metadata
    : {};
  const sourceValue = typeof source?.source === 'string' ? source.source : '';
  const urlValue = metadata?.url || metadata?.source_url || sourceValue;
  const urlDetails = getUrlDetails(urlValue);
  const filename = getFilename(sourceValue, metadata, urlDetails);
  const domain = urlDetails?.domain || null;
  const page = getPageLabel(source, metadata);
  const parts = [filename];

  if (domain && domain !== filename) {
    parts.push(domain);
  }

  if (page) {
    parts.push(page);
  }

  return {
    id: source?.id || `${filename}-${page || 'no-page'}-${index}`,
    label: parts.join(' • '),
  };
}

function renderInlineMarkdown(text) {
  const parts = String(text || '').split(/(\*\*[^*]+\*\*)/g);

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
    }

    return part;
  });
}

function renderMarkdownContent(content) {
  const lines = String(content || '').replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();

    if (!line) {
      index += 1;
      continue;
    }

    if (line.startsWith('```')) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push(
        <pre key={`code-${blocks.length}`} className="markdown-code">
          <code>{codeLines.join('\n')}</code>
        </pre>,
      );
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const HeadingTag = headingMatch[1].length <= 3 ? 'h3' : 'h4';
      blocks.push(
        <HeadingTag key={`heading-${blocks.length}`}>
          {renderInlineMarkdown(headingMatch[2])}
        </HeadingTag>,
      );
      index += 1;
      continue;
    }

    const unorderedMatch = line.match(/^[-*]\s+(.+)$/);
    if (unorderedMatch) {
      const items = [];
      while (index < lines.length) {
        const itemMatch = lines[index].trim().match(/^[-*]\s+(.+)$/);
        if (!itemMatch) {
          break;
        }
        items.push(itemMatch[1]);
        index += 1;
      }
      blocks.push(
        <ul key={`list-${blocks.length}`}>
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    const orderedMatch = line.match(/^\d+\.\s+(.+)$/);
    if (orderedMatch) {
      const items = [];
      while (index < lines.length) {
        const itemMatch = lines[index].trim().match(/^\d+\.\s+(.+)$/);
        if (!itemMatch) {
          break;
        }
        items.push(itemMatch[1]);
        index += 1;
      }
      blocks.push(
        <ol key={`ordered-${blocks.length}`}>
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (index < lines.length) {
      const nextLine = lines[index].trim();
      if (
        !nextLine
        || nextLine.startsWith('```')
        || /^#{1,6}\s+/.test(nextLine)
        || /^[-*]\s+/.test(nextLine)
        || /^\d+\.\s+/.test(nextLine)
      ) {
        break;
      }
      paragraphLines.push(nextLine);
      index += 1;
    }

    blocks.push(
      <p key={`paragraph-${blocks.length}`}>
        {renderInlineMarkdown(paragraphLines.join(' '))}
      </p>,
    );
  }

  return blocks.length > 0 ? blocks : <p>{content}</p>;
}

function ChatArea({
  messages = [],
  input = '',
  isThinking = false,
  onInputChange,
  onSend,
}) {
  const renderedMessages = useMemo(
    () => messages.map((message, index) => ({
      id: message?.id || `message-${index}`,
      role: message?.role === 'user' ? 'user' : 'assistant',
      content: message?.content || '',
      sources: Array.isArray(message?.sources)
        ? message.sources
          .map((source, sourceIndex) => formatCitation(source, sourceIndex))
          .filter((source) => source.label)
        : [],
      isLoading: Boolean(message?.isLoading),
      isError: Boolean(message?.isError),
    })),
    [messages],
  );

  return (
    <main className="chat-area">
      <section className="messages-container" aria-label="Conversation">
        <div className="chat-thread">
          {renderedMessages.length === 0 ? (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              No messages yet
            </span>
          ) : renderedMessages.map((message) => (
            <article
              key={message.id}
              className={[
                'message',
                `message-${message.role}`,
                message.isLoading ? 'message-loading' : '',
                message.isError ? 'message-error' : '',
              ].filter(Boolean).join(' ')}
            >
              <div className="message-content">
                {message.isLoading ? (
                  <>
                    <p>{message.content || 'Thinking...'}</p>
                    <div className="thinking-indicator" aria-label="Thinking">
                      <span />
                      <span />
                      <span />
                    </div>
                  </>
                ) : (
                  renderMarkdownContent(message.content)
                )}
              </div>
              {!message.isLoading && message.sources.length > 0 && (
                <div className="citation-row">
                  {message.sources.map((source) => (
                    <span key={source.id} title={source.label}>{source.label}</span>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>
      </section>

      <ChatInput
        value={input}
        isLoading={isThinking}
        onChange={onInputChange}
        onSend={onSend}
      />
    </main>
  );
}

export default memo(ChatArea);
