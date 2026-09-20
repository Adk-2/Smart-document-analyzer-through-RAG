import { apiRequest } from './client';

export function listSources(workspaceName = null) {
  const params = new URLSearchParams();

  if (workspaceName) {
    params.set('workspace_name', workspaceName);
  }

  const query = params.toString();
  const path = query ? `/sources?${query}` : '/sources';
  console.info('[sourceApi] listSources', {
    workspaceName: workspaceName || null,
    path,
  });

  return apiRequest(path, {
    method: 'GET',
    retries: 1,
  });
}

export function ingestSourceUrl(url, workspaceName = null) {
  console.info('[sourceApi] ingestSourceUrl', {
    workspaceName: workspaceName || null,
    url,
  });
  return apiRequest('/ingest/url', {
    method: 'POST',
    body: {
      url,
      workspace_name: workspaceName,
    },
    retries: 0,
  });
}

export function ingestSourcePdfs(files, workspaceName = null) {
  const formData = new FormData();
  Array.from(files).forEach((file) => {
    formData.append('files', file);
  });

  if (workspaceName) {
    formData.append('workspace_name', workspaceName);
  }

  console.info('[sourceApi] ingestSourcePdfs', {
    workspaceName: workspaceName || null,
    fileCount: files.length,
  });

  return apiRequest('/ingest/pdf', {
    method: 'POST',
    body: formData,
    timeoutMs: 120000,
    retries: 0,
  });
}
