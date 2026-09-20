import { ingestSourceUrl } from '../api/sourceApi';

export async function ingestUrl(url, workspaceName = null) {
  return ingestSourceUrl(url, workspaceName);
}
