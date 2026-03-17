import type { ModelsResponse, SessionResponse } from "./types.js";
export interface CoordinatorClientOptions {
    baseUrl: string;
    apiKey?: string;
    fetch?: typeof globalThis.fetch;
    headers?: HeadersInit;
}
export declare class CoordinatorClient {
    #private;
    constructor(options: CoordinatorClientOptions);
    getModels(): Promise<ModelsResponse>;
    createSession(modelName: string): Promise<SessionResponse>;
    getSession(sessionId: string): Promise<SessionResponse>;
    endSession(sessionId: string): Promise<void>;
}
export declare function createCoordinatorClient(options: CoordinatorClientOptions): CoordinatorClient;
