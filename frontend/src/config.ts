/**
 * Build-time configuration for ApexSupport AI UI.
 */
const env = import.meta.env;

export const API_BASE = (env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");
export const LANDING_URL = env.VITE_LANDING_URL || "/";
export const REPO_URL = env.VITE_REPO_URL || "#";

export const apiUrl = (path: string) => `${API_BASE}${path}`;
