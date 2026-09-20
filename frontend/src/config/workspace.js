export const DEFAULT_WORKSPACE_NAME = 'default_workspace';
export const LEGACY_DEFAULT_WORKSPACE_NAME = 'workspace';

export function normalizeWorkspaceName(workspaceName, fallbackWorkspaceName = null) {
  const sanitizedWorkspaceName = (workspaceName || '').trim();

  if (!sanitizedWorkspaceName) {
    return fallbackWorkspaceName;
  }

  if (sanitizedWorkspaceName === LEGACY_DEFAULT_WORKSPACE_NAME) {
    return DEFAULT_WORKSPACE_NAME;
  }

  return sanitizedWorkspaceName;
}
