use std::time::{SystemTime, UNIX_EPOCH};

use jsonwebtoken::{encode, errors::Error, EncodingKey, Header};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoGrant {
    room_join: bool,
    room: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LiveKitClaims {
    iss: String,
    sub: String,
    nbf: usize,
    exp: usize,
    video: VideoGrant,
}

pub fn mint_access_token(
    api_key: &str,
    api_secret: &str,
    identity: &str,
    room_name: &str,
) -> Result<String, Error> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must be after unix epoch")
        .as_secs() as usize;

    let claims = LiveKitClaims {
        iss: api_key.to_string(),
        sub: identity.to_string(),
        nbf: now,
        exp: now + 60 * 60,
        video: VideoGrant {
            room_join: true,
            room: room_name.to_string(),
        },
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(api_secret.as_bytes()),
    )
}

#[cfg(test)]
mod tests {
    use jsonwebtoken::{decode, DecodingKey, Validation};

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct DecodedClaims {
        iss: String,
        sub: String,
        nbf: usize,
        exp: usize,
        video: VideoGrant,
    }

    #[test]
    fn mints_decodable_token_with_expected_claims() {
        let token = mint_access_token("api-key", "secret", "client-1", "wm-room")
            .expect("token should be minted");

        let decoded = decode::<DecodedClaims>(
            &token,
            &DecodingKey::from_secret("secret".as_bytes()),
            &Validation::default(),
        )
        .expect("token should decode");

        assert_eq!(decoded.claims.iss, "api-key");
        assert_eq!(decoded.claims.sub, "client-1");
        assert_eq!(decoded.claims.video.room, "wm-room");
        assert!(decoded.claims.video.room_join);
    }
}
