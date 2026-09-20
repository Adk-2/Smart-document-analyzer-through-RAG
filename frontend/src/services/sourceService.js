import { listSources } from '../api/sourceApi';

export async function fetchSources(workspaceName = null) {
  return listSources(workspaceName);
}
