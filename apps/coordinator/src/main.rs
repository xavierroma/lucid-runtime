use std::error::Error;

use coordinator::{build_dispatchers, build_router, config::Config, fetch_capabilities, run_session_reconciler, AppContext};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env()?;
    let bind_addr = config.bind_addr;

    let dispatchers = build_dispatchers(&config);

    tracing::info!("fetching manifests from workers");
    let capabilities = fetch_capabilities(config.model_registry.models(), &dispatchers).await?;
    tracing::info!(models = capabilities.len(), "manifests loaded");

    let ctx = AppContext::with_modal_dispatchers(config, dispatchers, capabilities);
    let app = build_router(ctx.clone());

    tokio::spawn(run_session_reconciler(ctx));

    let listener = TcpListener::bind(bind_addr).await?;
    tracing::info!(%bind_addr, "coordinator listening");
    axum::serve(listener, app).await?;

    Ok(())
}
