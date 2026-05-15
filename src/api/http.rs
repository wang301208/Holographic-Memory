use std::sync::Arc;
use std::net::SocketAddr;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::holographic::HolographicMemory;
use crate::foundation::config::HolographicConfig;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<HolographicMemory>>,
}

impl AppState {
    pub fn new(hm: HolographicMemory) -> Self {
        Self {
            inner: Arc::new(Mutex::new(hm)),
        }
    }
}

#[derive(Deserialize)]
pub struct StoreRequest {
    pub data: Vec<f64>,
}

#[derive(Serialize)]
pub struct StoreResponse {
    pub stored: bool,
    pub source_id: u64,
    pub source_hash: u64,
    pub fragment_count: usize,
    pub total_fragments: usize,
}

#[derive(Deserialize)]
pub struct RetrieveRequest {
    pub source_hash: u64,
    pub expected_len: usize,
}

#[derive(Serialize)]
pub struct RetrieveResponse {
    pub data: Vec<f64>,
    pub source_hash: u64,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f64>,
    pub top_k: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
pub struct SearchResultItem {
    pub fragment_id: u64,
    pub similarity: f64,
    pub source_hash: u64,
}

#[derive(Deserialize)]
pub struct IntegrityRequest {
    pub source_hash: u64,
}

#[derive(Serialize)]
pub struct IntegrityResponse {
    pub fragments_total: u32,
    pub fragments_available: u32,
    pub damage_ratio: f64,
    pub recovery_possible: bool,
}

#[derive(Deserialize)]
pub struct RecoverRequest {
    pub source_hash: u64,
    pub damage_pct: f64,
}

#[derive(Serialize)]
pub struct RecoverResponse {
    pub total_fragments: usize,
    pub available_fragments: usize,
    pub damage_ratio: f64,
    pub mse: f64,
    pub recovery_possible: bool,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub service: String,
    pub version: String,
    pub fragment_count: usize,
    pub source_count: usize,
}

#[derive(Serialize)]
pub struct ApiError {
    pub error: String,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/status", get(status))
        .route("/store", post(store))
        .route("/retrieve", post(retrieve))
        .route("/search", post(search))
        .route("/integrity", post(integrity))
        .route("/recover", post(recover))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn root() -> &'static str {
    "{\"service\":\"holographic-memory\",\"version\":\"0.3.0\"}"
}

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let hm = state.inner.lock().await;
    let response = StatusResponse {
        service: "holographic-memory".to_string(),
        version: "0.3.0".to_string(),
        fragment_count: hm.fragment_count(),
        source_count: hm.source_count(),
    };
    (StatusCode::OK, Json(response))
}

async fn store(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> impl IntoResponse {
    if req.data.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(ApiError { error: "数据为空".to_string() })).into_response();
    }

    let mut hm = state.inner.lock().await;
    match hm.store(&req.data) {
        Ok(result) => {
            let resp = StoreResponse {
                stored: true,
                source_id: result.source_id,
                source_hash: result.source_hash,
                fragment_count: result.fragment_count,
                total_fragments: result.total_fragments,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError { error: e.to_string() })).into_response(),
    }
}

async fn retrieve(
    State(state): State<AppState>,
    Json(req): Json<RetrieveRequest>,
) -> impl IntoResponse {
    let mut hm = state.inner.lock().await;
    match hm.retrieve(req.source_hash, req.expected_len) {
        Ok(data) => {
            let resp = RetrieveResponse {
                data,
                source_hash: req.source_hash,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => (StatusCode::NOT_FOUND, Json(ApiError { error: e.to_string() })).into_response(),
    }
}

async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    if req.query.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(ApiError { error: "查询为空".to_string() })).into_response();
    }

    let top_k = req.top_k.unwrap_or(10);
    let mut hm = state.inner.lock().await;
    match hm.search(&req.query, top_k) {
        Ok(results) => {
            let items: Vec<SearchResultItem> = results.iter()
                .map(|r| SearchResultItem {
                    fragment_id: r.fragment_id,
                    similarity: r.similarity,
                    source_hash: r.metadata.source_hash,
                })
                .collect();
            (StatusCode::OK, Json(SearchResponse { results: items })).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError { error: e.to_string() })).into_response(),
    }
}

async fn integrity(
    State(state): State<AppState>,
    Json(req): Json<IntegrityRequest>,
) -> impl IntoResponse {
    let hm = state.inner.lock().await;
    let report = hm.integrity(req.source_hash);
    let resp = IntegrityResponse {
        fragments_total: report.fragments_total,
        fragments_available: report.fragments_available,
        damage_ratio: report.damage_ratio,
        recovery_possible: report.recovery_possible,
    };
    (StatusCode::OK, Json(resp))
}

async fn recover(
    State(state): State<AppState>,
    Json(req): Json<RecoverRequest>,
) -> impl IntoResponse {
    if req.damage_pct < 0.0 || req.damage_pct > 1.0 {
        return (StatusCode::BAD_REQUEST, Json(ApiError { error: "损毁比例需在0-1之间".to_string() })).into_response();
    }

    let mut hm = state.inner.lock().await;
    let fragments = hm.store_with_fault_tolerance(&vec![0.0; 256], req.damage_pct);
    match fragments {
        Ok(ft) => {
            let resp = RecoverResponse {
                total_fragments: ft.total_fragments,
                available_fragments: ft.available_fragments,
                damage_ratio: ft.damage_ratio,
                mse: ft.mse,
                recovery_possible: ft.integrity.recovery_possible,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError { error: e.to_string() })).into_response(),
    }
}

pub async fn serve(config: HolographicConfig, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let hm = HolographicMemory::new(config);
    let state = AppState::new(hm);
    let app = create_router(state);
    let socket_addr: SocketAddr = addr.parse()?;

    let listener = tokio::net::TcpListener::bind(socket_addr).await?;
    println!("全息记忆 API 服务启动于 http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
