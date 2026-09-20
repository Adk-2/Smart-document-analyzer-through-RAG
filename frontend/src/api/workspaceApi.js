import { listSources } from './sourceApi';

export function getWorkspaceSources(workspaceName) {
  return listSources(workspaceName);
}
