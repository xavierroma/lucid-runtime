use std::error::Error;

use coordinator::{build_router, config::Config, run_session_reconciler, AppContext};
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

    let ctx = AppContext::new(config);
    let app = build_router(ctx.clone());

    tokio::spawn(run_session_reconciler(ctx));

    let listener = TcpListener::bind(bind_addr).await?;
    tracing::info!(%bind_addr, "coordinator listening");
    axum::serve(listener, app).await?;

    Ok(())
}
