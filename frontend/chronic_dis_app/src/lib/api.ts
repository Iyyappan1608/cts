import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_TOKEN_KEY = '@HealthApp:sessionToken';

// Centralized API base URL. Change here when backend IP/port changes.
export const API_URL = 'http://10.144.149.238:5000';

export async function setSessionToken(token: string): Promise<void> {
  await AsyncStorage.setItem(STORAGE_TOKEN_KEY, token);
}

export async function clearSessionToken(): Promise<void> {
  await AsyncStorage.removeItem(STORAGE_TOKEN_KEY);
}

export async function getSessionToken(): Promise<string | null> {
  return AsyncStorage.getItem(STORAGE_TOKEN_KEY);
}

async function buildHeaders(includeAuth: boolean, extraHeaders?: Record<string, string>): Promise<Record<string, string>> {
  const base: Record<string, string> = { 'Content-Type': 'application/json' };
  if (extraHeaders) {
    Object.assign(base, extraHeaders);
  }
  if (includeAuth) {
    const token = await getSessionToken();
    if (token) {
      base['Authorization'] = `Bearer ${token}`;
    }
  }
  return base;
}

export async function post<T = any>(endpoint: string, body: any, includeAuth: boolean = true): Promise<{ ok: boolean; status: number; data: T }> {
  const url = `${API_URL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
  const headers = await buildHeaders(includeAuth);
  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({} as any));
  return { ok: response.ok, status: response.status, data };
}

export async function get<T = any>(endpoint: string, includeAuth: boolean = true): Promise<{ ok: boolean; status: number; data: T }> {
  const url = `${API_URL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
  const headers = await buildHeaders(includeAuth);
  const response = await fetch(url, {
    method: 'GET',
    headers,
  });
  const data = await response.json().catch(() => ({} as any));
  return { ok: response.ok, status: response.status, data };
}


