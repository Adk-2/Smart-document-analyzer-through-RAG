import { create } from 'zustand';

import { ingestSourcePdfs, ingestSourceUrl, listSources } from '../api/sourceApi';
import { DEFAULT_WORKSPACE_NAME, normalizeWorkspaceName } from '../config/workspace';

export const useAppStore = create((set, get) => ({
  workspaces: [DEFAULT_WORKSPACE_NAME],
  activeWorkspace: DEFAULT_WORKSPACE_NAME,
  sources: [],
  loading: {
    sources: false,
    ingestion: false,
    retryingSources: false,
  },
  errors: {
    sources: null,
    ingestion: null,
  },
  ingestionStatus: '',
  backendOffline: false,
  sourceRequestId: 0,
  ingestionRequestId: 0,

  setActiveWorkspace: (workspaceName) => {
    set({ activeWorkspace: normalizeWorkspaceName(workspaceName, DEFAULT_WORKSPACE_NAME) });
  },

  setIngestionFeedback: (message, errorMessage = null) => {
    set((state) => ({
      ingestionStatus: message,
      errors: {
        ...state.errors,
        ingestion: errorMessage,
      },
    }));
  },

  loadSources: async ({ retrying = false } = {}) => {
    const requestId = get().sourceRequestId + 1;
    const workspaceName = get().activeWorkspace;
    console.info('[store] loadSources_start', {
      requestId,
      workspaceName,
      retrying,
    });

    set((state) => ({
      sourceRequestId: requestId,
      loading: {
        ...state.loading,
        sources: true,
        retryingSources: retrying,
      },
      errors: {
        ...state.errors,
        sources: null,
      },
    }));

    try {
      const data = await listSources(workspaceName);
      console.info('[store] loadSources_completed', {
        requestId,
        requestedWorkspace: workspaceName,
        responseWorkspace: data.workspace || null,
        sourceCount: Array.isArray(data.sources) ? data.sources.length : 0,
      });

      if (get().sourceRequestId !== requestId) {
        return;
      }

      const sources = data.sources || [];
      const workspaceNames = new Set(
        sources
          .map((source) => source.workspace_name || source.workspace)
          .filter(Boolean),
      );

      set((state) => ({
        sources,
        workspaces: Array.from(new Set([...state.workspaces, ...workspaceNames])),
        loading: {
          ...state.loading,
          sources: false,
          retryingSources: false,
        },
        errors: {
          ...state.errors,
          sources: null,
        },
        backendOffline: false,
      }));
    } catch (error) {
      if (get().sourceRequestId !== requestId) {
        return;
      }

      set((state) => ({
        loading: {
          ...state.loading,
          sources: false,
          retryingSources: false,
        },
        errors: {
          ...state.errors,
          sources: error.message,
        },
        backendOffline: true,
      }));
      console.error('[store] loadSources_failed', {
        requestId,
        workspaceName,
        error: error.message,
      });
    }
  },

  retryLoadSources: async () => {
    await get().loadSources({ retrying: true });
  },

  ingestUrl: async (url) => {
    const sourceUrl = url.trim();
    if (!sourceUrl) {
      const errorMessage = 'URL is required';
      get().setIngestionFeedback(errorMessage, errorMessage);
      return { success: false, error: errorMessage };
    }

    const requestId = get().ingestionRequestId + 1;
    const workspaceName = get().activeWorkspace;
    console.info('[store] ingestUrl_start', {
      requestId,
      workspaceName,
      sourceUrl,
    });

    set((state) => ({
      ingestionRequestId: requestId,
      ingestionStatus: '',
      loading: {
        ...state.loading,
        ingestion: true,
      },
      errors: {
        ...state.errors,
        ingestion: null,
      },
    }));

    try {
      const result = await ingestSourceUrl(sourceUrl, workspaceName);
      console.info('[store] ingestUrl_completed', {
        requestId,
        requestedWorkspace: workspaceName,
        responseWorkspace: result.workspace || null,
        chunks: result.chunks || 0,
        vectorInsertion: result.vector_insertion || null,
        vectorTotalDocuments: result.vector_total_documents || null,
      });

      if (get().ingestionRequestId !== requestId) {
        return { success: false, stale: true };
      }

      set((state) => ({
        activeWorkspace: result.workspace || state.activeWorkspace,
        workspaces: result.workspace
          ? Array.from(new Set([...state.workspaces, result.workspace]))
          : state.workspaces,
        ingestionStatus: `Successfully ingested: ${result.title}`,
        loading: {
          ...state.loading,
          ingestion: false,
        },
        errors: {
          ...state.errors,
          ingestion: null,
        },
        backendOffline: false,
      }));

      await get().loadSources();
      return { success: true, data: result };
    } catch (error) {
      if (get().ingestionRequestId !== requestId) {
        return { success: false, stale: true };
      }

      set((state) => ({
        ingestionStatus: `Failed to ingest URL: ${error.message}`,
        loading: {
          ...state.loading,
          ingestion: false,
        },
        errors: {
          ...state.errors,
          ingestion: error.message,
        },
        backendOffline: error.message === 'Network request failed' || error.message === 'Request timed out',
      }));
      console.error('[store] ingestUrl_failed', {
        requestId,
        workspaceName,
        sourceUrl,
        error: error.message,
      });
      return { success: false, error: error.message };
    }
  },

  ingestPdfs: async (files) => {
    const pdfFiles = Array.from(files || []).filter((file) => (
      file?.type === 'application/pdf' || file?.name?.toLowerCase().endsWith('.pdf')
    ));
    if (pdfFiles.length === 0) {
      const errorMessage = 'Select at least one PDF file';
      get().setIngestionFeedback(errorMessage, errorMessage);
      return { success: false, error: errorMessage };
    }

    const requestId = get().ingestionRequestId + 1;
    const workspaceName = get().activeWorkspace;
    console.info('[store] ingestPdfs_start', {
      requestId,
      workspaceName,
      fileCount: pdfFiles.length,
    });

    set((state) => ({
      ingestionRequestId: requestId,
      ingestionStatus: '',
      loading: {
        ...state.loading,
        ingestion: true,
      },
      errors: {
        ...state.errors,
        ingestion: null,
      },
    }));

    try {
      const result = await ingestSourcePdfs(pdfFiles, workspaceName);
      console.info('[store] ingestPdfs_completed', {
        requestId,
        requestedWorkspace: workspaceName,
        responseWorkspace: result.workspace || null,
        fileCount: result.file_count || pdfFiles.length,
        chunks: result.chunks || 0,
      });

      if (get().ingestionRequestId !== requestId) {
        return { success: false, stale: true };
      }

      set((state) => ({
        activeWorkspace: result.workspace || state.activeWorkspace,
        workspaces: result.workspace
          ? Array.from(new Set([...state.workspaces, result.workspace]))
          : state.workspaces,
        ingestionStatus: `Successfully ingested ${result.file_count || pdfFiles.length} PDF${(result.file_count || pdfFiles.length) === 1 ? '' : 's'}`,
        loading: {
          ...state.loading,
          ingestion: false,
        },
        errors: {
          ...state.errors,
          ingestion: null,
        },
        backendOffline: false,
      }));

      await get().loadSources();
      return { success: true, data: result };
    } catch (error) {
      if (get().ingestionRequestId !== requestId) {
        return { success: false, stale: true };
      }

      set((state) => ({
        ingestionStatus: `Failed to ingest PDFs: ${error.message}`,
        loading: {
          ...state.loading,
          ingestion: false,
        },
        errors: {
          ...state.errors,
          ingestion: error.message,
        },
        backendOffline: error.message === 'Network request failed' || error.message === 'Request timed out',
      }));
      console.error('[store] ingestPdfs_failed', {
        requestId,
        workspaceName,
        error: error.message,
      });
      return { success: false, error: error.message };
    }
  },
}));
