import { memo } from 'react';

function ChatInput({
  value = '',
  isLoading = false,
  onChange,
  onSend,
}) {
  const handleSubmit = (event) => {
    event.preventDefault();

    const question = value.trim();
    if (!question || isLoading) {
      return;
    }

    onSend?.(question);
  };

  return (
    <div className="chat-input-wrapper">
      <form className="chat-input" onSubmit={handleSubmit}>
        <textarea
          placeholder="Ask a question about your sources"
          rows="2"
          value={value}
          onChange={(event) => onChange?.(event.target.value)}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !value.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default memo(ChatInput);
