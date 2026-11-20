/** Same-origin API client: never fall back to fabricated or cached analytical results. */
export async function api(path: string, body?: unknown) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120_000);
  try {
    const response = await fetch("/api" + path, {
      signal: controller.signal,
      ...(body === undefined
        ? {}
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
    });
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error(
        `Backend returned an unexpected response (${response.status}). Check the server log and retry.`,
      );
    }
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      throw new Error(
        Array.isArray(detail)
          ? detail
              .map(
                (e: { loc: string[]; msg: string }) =>
                  `${e.loc.slice(1).join(".")}: ${e.msg}`,
              )
              .join("; ")
          : detail || `Server returned ${response.status}`,
      );
    }
    return data;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError")
      throw new Error(
        "The server did not finish within two minutes. Check server availability and the experiment registry before retrying.",
      );
    if (error instanceof TypeError)
      throw new Error(
        "Cannot reach the Python backend. Start the application with ./run.sh and retry.",
      );
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
