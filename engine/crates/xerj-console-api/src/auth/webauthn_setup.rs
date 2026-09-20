//! WebAuthn relying-party setup.
//!
//! Builds a `Webauthn` instance from `ConsoleState.rp` (rp_id, rp_origin,
//! rp_name). Cheap enough to call per-request — `Webauthn::new` just
//! stores config; no I/O.

use webauthn_rs::prelude::*;

use crate::error::{ConsoleApiError, ConsoleResult};
use crate::state::ConsoleState;

pub fn build(state: &ConsoleState) -> ConsoleResult<Webauthn> {
    let origin = Url::parse(&state.rp.rp_origin)
        .map_err(|e| ConsoleApiError::Internal(format!("rp_origin parse: {e}")))?;
    let builder = WebauthnBuilder::new(&state.rp.rp_id, &origin)
        .map_err(|e| ConsoleApiError::Internal(format!("webauthn build: {e}")))?
        .rp_name(&state.rp.rp_name);
    // A loopback deployment may be reached on a port other than the one the
    // origin was derived from (a container port map, or a passkey enrolled
    // when the node listened elsewhere). `webauthn-rs` still requires the
    // scheme and host to match — only the port check is relaxed, and only
    // for `localhost` (#935). Named-host deployments keep strict matching.
    let builder = if state.rp.rp_id == "localhost" {
        builder.allow_any_port(true)
    } else {
        builder
    };
    builder
        .build()
        .map_err(|e| ConsoleApiError::Internal(format!("webauthn finalise: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ClusterMode, RpConfig};
    use tempfile::TempDir;
    use xerj_common::config::Config;
    use xerj_engine::Engine;

    fn state_with_rp(rp: RpConfig) -> (ConsoleState, TempDir) {
        let dir = TempDir::new().unwrap();
        let mut cfg = Config::default();
        cfg.server.data_dir = dir.path().to_str().unwrap().to_string();
        let engine = Engine::new(cfg).expect("engine");
        let state = ConsoleState::new_with_rp(
            engine,
            "local".into(),
            [0u8; 32],
            ClusterMode::Standalone,
            rp,
        );
        (state, dir)
    }

    #[tokio::test]
    async fn a_derived_loopback_rp_builds_on_any_port() {
        // #935: the RP used to be hard-coded to http://localhost:9200, so a
        // node on any other port had no constructible relying party for the
        // origin its browser actually used.
        let rp = RpConfig::from_bind_url("http://127.0.0.1:9550").unwrap();
        assert_eq!(rp.rp_origin, "http://localhost:9550");
        let (state, _dir) = state_with_rp(rp);
        assert!(build(&state).is_ok());
    }

    #[tokio::test]
    async fn an_ip_literal_origin_cannot_build_an_rp() {
        // Why `from_bind_url` rewrites the host instead of using the bind URL
        // verbatim: `WebauthnBuilder` requires the rp_id to be an effective
        // domain of the origin, and an IP-literal origin has no domain.
        let rp = RpConfig {
            rp_id: "localhost".into(),
            rp_origin: "http://127.0.0.1:9550".into(),
            rp_name: "Xerj Console".into(),
        };
        let (state, _dir) = state_with_rp(rp);
        assert!(build(&state).is_err());
    }

    #[tokio::test]
    async fn a_named_host_rp_builds() {
        let rp = RpConfig::from_bind_url("http://console.example.com:9200").unwrap();
        let (state, _dir) = state_with_rp(rp);
        assert!(build(&state).is_ok());
    }
}
