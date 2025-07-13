export interface FetchDataOptions {
  headers?: Record<string, string>;
  method?: string;
  body?: BodyInit;
}

/**
 * Fetch JSON data from the provided URL using async/await.
 * The JWT token is read from the environment to avoid hardcoding secrets.
 */
export async function fetchData<T = unknown>(
  url: string,
  options: FetchDataOptions = {}
): Promise<T> {
  const token = process.env.API_TOKEN;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        Authorization: token ? `Bearer ${token}` : '',
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    console.error('Failed to fetch data:', err);
    throw err;
  }
}
