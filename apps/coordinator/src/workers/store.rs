use std::sync::Mutex;

use crate::registry::ModelTimeouts;

#[derive(Clone, Debug)]
pub struct RegisteredWorker {
    pub model_id: String,
    pub display_name: String,
    pub dispatch_base_url: String,
    pub dispatch_token: String,
    pub worker_id: String,
    pub timeouts: ModelTimeouts,
}

pub trait WorkerStore: Send + Sync {
    /// Insert or replace a worker by `model_id`.
    fn save(&self, worker: RegisteredWorker);
    fn get(&self, model_id: &str) -> Option<RegisteredWorker>;
    /// Returns workers in insertion order.
    fn list(&self) -> Vec<RegisteredWorker>;
    /// Returns `true` if the worker was present and removed.
    fn remove(&self, model_id: &str) -> bool;
}

/// In-memory worker store that preserves insertion order.
#[derive(Default)]
pub struct InMemoryWorkerStore {
    workers: Mutex<Vec<RegisteredWorker>>,
}

impl InMemoryWorkerStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl WorkerStore for InMemoryWorkerStore {
    fn save(&self, worker: RegisteredWorker) {
        let mut workers = self.workers.lock().unwrap();
        if let Some(existing) = workers.iter_mut().find(|w| w.model_id == worker.model_id) {
            *existing = worker;
        } else {
            workers.push(worker);
        }
    }

    fn get(&self, model_id: &str) -> Option<RegisteredWorker> {
        self.workers
            .lock()
            .unwrap()
            .iter()
            .find(|w| w.model_id == model_id)
            .cloned()
    }

    fn list(&self) -> Vec<RegisteredWorker> {
        self.workers.lock().unwrap().clone()
    }

    fn remove(&self, model_id: &str) -> bool {
        let mut workers = self.workers.lock().unwrap();
        let before = workers.len();
        workers.retain(|w| w.model_id != model_id);
        workers.len() < before
    }
}
