use holographic_memory::*;

#[tokio::test]
async fn test_api_store_and_retrieve() {
    let config = HolographicConfig::default();
    let hm = HolographicMemory::new(config);
    let state = AppState::new(hm);

    let app = create_router(state);

    let store_req = StoreRequest {
        data: (0..256).map(|i| (i as f64 * 0.05).sin()).collect(),
    };
    let store_body = serde_json::to_string(&store_req).unwrap();

    let resp = axum::body::to_bytes(
        app.clone().oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/store")
                .header("content-type", "application/json")
                .body(store_body.into()).unwrap()
        ).await.unwrap(),
        4096
    ).await.unwrap();

    let store_resp: StoreResponse = serde_json::from_slice(&resp).unwrap();
    assert!(store_resp.stored);
    assert!(store_resp.total_fragments > 0);
}

#[tokio::test]
async fn test_api_status() {
    let config = HolographicConfig::default();
    let hm = HolographicMemory::new(config);
    let state = AppState::new(hm);
    let app = create_router(state);

    let resp = axum::body::to_bytes(
        app.clone().oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/status")
                .body(axum::body::Empty::<bytes::Bytes>::new().into()).unwrap()
        ).await.unwrap(),
        4096
    ).await.unwrap();

    let status: StatusResponse = serde_json::from_slice(&resp).unwrap();
    assert_eq!(status.service, "holographic-memory");
    assert_eq!(status.version, "0.3.0");
}

#[tokio::test]
async fn test_api_root() {
    let config = HolographicConfig::default();
    let hm = HolographicMemory::new(config);
    let state = AppState::new(hm);
    let app = create_router(state);

    let resp = axum::body::to_bytes(
        app.clone().oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/")
                .body(axum::body::Empty::<bytes::Bytes>::new().into()).unwrap()
        ).await.unwrap(),
        4096
    ).await.unwrap();

    let text = String::from_utf8_lossy(&resp);
    assert!(text.contains("holographic-memory"));
}
