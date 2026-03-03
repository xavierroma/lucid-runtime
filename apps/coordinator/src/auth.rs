use axum::http::{header::AUTHORIZATION, HeaderMap};

pub fn is_bearer_authorized(headers: &HeaderMap, expected_token: &str) -> bool {
    let Some(raw_header) = headers.get(AUTHORIZATION) else {
        return false;
    };

    let Ok(raw_header) = raw_header.to_str() else {
        return false;
    };

    let Some(token) = raw_header.strip_prefix("Bearer ") else {
        return false;
    };

    token == expected_token
}
