import { memo } from 'react';
import { useShallow } from 'zustand/react/shallow';

import { useAppStore } from '../../store/appStore';


function InsightsPanel() {
  const {
    sourceCount,
    indexedCount,
    failedCount,
    isLoadingSources,
  } = useAppStore(
    useShallow((state) => {
      const sources = Array.isArray(state.sources) ? state.sources : [];
      return {
        sourceCount: sources.length,
        indexedCount: sources.filter((source) => source?.status === 'indexed').length,
        failedCount: sources.filter((source) => source?.status === 'failed').length,
        isLoadingSources: state.loading.sources,
      };
    }),
  );
  const coverage = sourceCount > 0
    ? Math.round((indexedCount / sourceCount) * 100)
    : 0;

  return (
    <aside className="insights-panel">
      <section className="panel">
        <div className="panel-heading">
          <h2>Insights</h2>
          <span>{isLoadingSources ? 'Refreshing' : 'Live sources'}</span>
        </div>

        <div className="metric-grid">
          <article>
            <strong>{sourceCount}</strong>
            <span>Sources</span>
          </article>
          <article>
            <strong>{coverage}%</strong>
            <span>Coverage</span>
          </article>
        </div>

        <article className="insight-card">
          <span>Status</span>
          <strong>{indexedCount} indexed</strong>
          <p>{sourceCount === 0 ? 'No backend sources are indexed yet.' : 'Source status is derived from the backend source registry.'}</p>
        </article>

        <article className="insight-card">
          <span>Failures</span>
          <strong>{failedCount} failed</strong>
          <p>{failedCount > 0 ? 'Review failed source ingestions before querying.' : 'No failed backend sources are currently listed.'}</p>
        </article>
      </section>
    </aside>
  );
}

export default memo(InsightsPanel);
