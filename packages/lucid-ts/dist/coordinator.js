export class CoordinatorClient {
    #baseUrl;
    #apiKey;
    #fetchImpl;
    #headers;
    constructor(options) {
        this.#baseUrl = options.baseUrl.replace(/\/$/, "");
        this.#apiKey = options.apiKey?.trim() || null;
        this.#fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
        this.#headers = options.headers;
    }
    getModels() {
        return this.#request("/models");
    }
    createSession(modelName) {
        return this.#request("/sessions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ model_name: modelName }),
        });
    }
    getSession(sessionId) {
        return this.#request(`/sessions/${sessionId}`);
    }
    endSession(sessionId) {
        return this.#request(`/sessions/${sessionId}:end`, {
            method: "POST",
        });
    }
    async #request(path, init) {
        const headers = new Headers(this.#headers);
        const overrideHeaders = new Headers(init?.headers);
        overrideHeaders.forEach((value, key) => headers.set(key, value));
        if (this.#apiKey) {
            headers.set("Authorization", `Bearer ${this.#apiKey}`);
        }
        const response = await this.#fetchImpl(`${this.#baseUrl}${path}`, {
            ...init,
            headers,
        });
        if (!response.ok) {
            throw new Error(await parseError(response));
        }
        if (response.status === 204) {
            return undefined;
        }
        return (await response.json());
    }
}
export function createCoordinatorClient(options) {
    return new CoordinatorClient(options);
}
async function parseError(response) {
    try {
        const payload = (await response.json());
        if (payload.error) {
            return payload.error;
        }
    }
    catch {
        // Fall back to the response status when the body is not valid JSON.
    }
    return response.statusText || `request failed with ${response.status}`;
}
