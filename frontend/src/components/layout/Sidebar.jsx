import { memo, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';

import { useAppStore } from '../../store/appStore';
import PdfUpload from '../sources/PdfUpload';
import UrlInput from '../sources/UrlInput';


function formatSourceUrl(source) {
  const sourceUrl = source?.source_url;
  if (!sourceUrl || typeof sourceUrl !== 'string') {
    return 'unknown source';
  }

  try {
    const parsedUrl = new URL(sourceUrl);
    const pathname = parsedUrl.pathname || '/';
    const query = parsedUrl.search || '';
    return `${parsedUrl.host}${pathname}${query}`;
  } catch {
    return sourceUrl;
  }
}


function getSourceCardModel(source, index) {
  const sourceType = source?.source_type || 'source';
  const status = (source?.status || 'unknown').toLowerCase();

  return {
    id: source?.id || `${sourceType}-${index}`,
    icon: sourceType === 'webpage' ? '🌐' : '📄',
    label: formatSourceUrl(source),
    status,
    statusClassName: `status-dot status-dot-${status}`,
  };
}


function Sidebar() {
  const {
    sources,
    isLoadingSources,
    isRetryingSources,
    sourceError,
    backendOffline,
    retryLoadSources,
  } = useAppStore(
    useShallow((state) => ({
      sources: Array.isArray(state.sources) ? state.sources : [],
      isLoadingSources: state.loading.sources,
      isRetryingSources: state.loading.retryingSources,
      sourceError: state.errors.sources,
      backendOffline: state.backendOffline,
      retryLoadSources: state.retryLoadSources,
    })),
  );
  const sourceCount = sources.length;
  const sourceCards = useMemo(
    () => sources.map(getSourceCardModel),
    [sources],
  );

  return (
    <aside className="sidebar">
      <section className="panel">
        <div className="panel-heading">
          <h2>Sources</h2>
          <span>{sourceCount} files</span>
        </div>

        <PdfUpload />

        <UrlInput />
      </section>

      <section className="source-list">
        {isLoadingSources ? (
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            {isRetryingSources ? 'Retrying source refresh...' : 'Loading sources...'}
          </span>
        ) : sourceError ? (
          <article className="source-card">
            <span style={{ fontSize: '0.8rem', color: '#ff8e8e', fontWeight: 600 }}>
              {backendOffline ? 'Backend offline' : 'Source refresh failed'}
            </span>
            <span className="source-meta">
              {sourceError}
            </span>
            <button
              type="button"
              className="navbar-button"
              onClick={retryLoadSources}
              style={{ marginTop: '0.4rem', width: 'fit-content' }}
            >
              Retry
            </button>
          </article>
        ) : sourceCount === 0 ? (
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            No sources indexed yet
          </span>
        ) : (
          sourceCards.map((source) => {
            return (
              <article key={source.id} className="source-card active">
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.8rem', color: '#8fa4d6', fontWeight: 600 }}>
                    {source.icon} {source.label}
                  </span>
                </div>
                <span
                  className="source-meta"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.3, marginTop: '0.15rem' }}
                >
                  <span
                    className={source.statusClassName}
                    aria-hidden="true"
                  />
                  {source.status}
                </span>
              </article>
            );
          })
        )}
      </section>
    </aside>
  );
}

export default memo(Sidebar);
