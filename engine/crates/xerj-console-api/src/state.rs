//! Shared state for Xerj Console handlers.
//!
//! `ConsoleState` is `Clone` and lives behind `axum::extract::State`. The
//! types it wraps are individually responsible for their own concurrency
//! (the `Engine` is internally `Arc`-shared, the in-memory caches are
//! `Arc<DashMap<…>>` etc.). This struct is just the join.

use std::sync::Arc;

use dashmap::DashMap;
use xerj_common::net::TrustedProxies;
use xerj_engine::Engine;

use crate::client_ip::TrustedProxySource;
use crate::error::{ConsoleApiError, ConsoleResult};
use crate::time::now_epoch_ms;

/// Server start time, in epoch ms. Used by `/cluster/info` so the SPA can
/// show "node up since …" without a separate /uptime fetch.
#[derive(Debug, Clone, Copy)]
pub struct StartedAt(pub i64);

/// In-memory record of a WebAuthn challenge that the server issued but
/// the client has not yet completed. Indexed by an opaque challenge id
/// (random 32-byte url-safe base64) that the SPA echoes back on
/// `…/finish`. Auto-expires after 5 minutes; entries past their TTL
/// are pruned on every insert.
#[derive(Debug, Clone)]
pub struct PendingChallenge {
    pub kind: ChallengeKind,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone)]
pub enum ChallengeKind {
    /// `POST /auth/passkey/begin` — enrolling a new credential. We hold
    /// the registration state plus the user_id we're enrolling for so
    /// `…/finish` knows where to attach the credential.
    PasskeyEnroll {
        user_id: String,
        state: webauthn_rs::prelude::PasskeyRegistration,
    },
    /// `POST /auth/login/begin` — proving an existing credential.
    /// The `email` is null when we returned a fake challenge for an
    /// unknown email (anti-enumeration).
    Login {
        email: Option<String>,
        state: webauthn_rs::prelude::PasskeyAuthentication,
    },
}

/// In-memory record of an enrollment session — issued when a user
/// redeems a magic link, consumed when they finish passkey enrollment.
/// Lives only in RAM (max 30 minutes), not persisted; if the server
/// restarts mid-bootstrap, the user re-redeems with a fresh link.
#[derive(Debug, Clone)]
pub struct EnrollmentSession {
    pub session_id: String,
    pub email: String,
    pub user_id: String,
    pub role: String,
    pub created_at_ms: i64,
    pub expires_at_ms: i64,
}

/// Coarse "is this node part of a RAFT cluster?" flag, set at startup
/// from `cfg.cluster.enabled` and never mutated. Phase 4 swaps this for
/// a live handle to the cluster runner so `/cluster/raft` can stream
/// real RAFT state; in phase 1 we only need the standalone-vs-raft
/// branch in `/cluster/info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterMode {
    Standalone,
    Raft,
}

/// WebAuthn Relying Party configuration. `rp_id` is the effective domain
/// (e.g. `"localhost"` for dev, `"xerj-console.example.com"` for prod) and
/// `rp_origin` is the URL the SPA was loaded from (e.g.
/// `"http://localhost:9200"`). Both must match between enrolment and
/// login or the browser refuses the assertion.
#[derive(Debug, Clone)]
pub struct RpConfig {
    pub rp_id: String,
    pub rp_origin: String,
    pub rp_name: String,
}

impl Default for RpConfig {
    fn default() -> Self {
        Self {
            rp_id: "localhost".to_string(),
            rp_origin: "http://localhost:9200".to_string(),
            rp_name: default_rp_name(),
        }
    }
}

fn default_rp_name() -> String {
    "Xerj Console".to_string()
}

impl RpConfig {
    /// Derive the relying party from the URL the console is actually
    /// served on (in `xerj-server` that is the bound address plus the
    /// *bound* ES-compat port, so ephemeral ports are already resolved —
    /// issue #469).
    ///
    /// * Loopback and unspecified hosts (`localhost`, `127.0.0.1`, `::1`,
    ///   `0.0.0.0`, `::`) all map to `rp_id = "localhost"` with the origin
    ///   rewritten to `<scheme>://localhost:<port>`. Browsers refuse an
    ///   IP-literal WebAuthn origin outright, so keeping the bind address
    ///   in the origin makes enrolment impossible on any port; and keeping
    ///   `rp_id` on `localhost` preserves credentials enrolled against the
    ///   historical hard-coded default (#935).
    /// * A named host is used verbatim: `rp_id` = host, origin = the URL.
    /// * A non-loopback IP literal is refused: a WebAuthn RP id must be a
    ///   registrable domain, so there is no honest configuration for
    ///   `http://192.168.0.5:9200`. Serve the console behind a named host
    ///   (reverse proxy) or bind loopback to use passkeys.
    pub fn from_bind_url(bind_url: &str) -> ConsoleResult<Self> {
        let url = webauthn_rs::prelude::Url::parse(bind_url).map_err(|e| {
            ConsoleApiError::Internal(format!("console bind url `{bind_url}`: {e}"))
        })?;
        let Some(host) = url.host_str().map(|h| h.to_string()) else {
            return Err(ConsoleApiError::Internal(format!(
                "console bind url `{bind_url}` has no host"
            )));
        };
        // `Url::host_str` keeps the brackets on an IPv6 literal
        // (`http://[::1]:9550` → `"[::1]"`); a domain name can never contain
        // one, so trimming them restores the parseable spelling.
        let host = host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_string();
        let port = url.port().map(|p| format!(":{p}")).unwrap_or_default();
        if bind_host_is_local(&host) {
            Ok(Self {
                rp_id: "localhost".to_string(),
                rp_origin: format!("{}://localhost{port}", url.scheme()),
                rp_name: default_rp_name(),
            })
        } else if host.parse::<std::net::IpAddr>().is_ok() {
            Err(ConsoleApiError::Internal(format!(
                "cannot derive a WebAuthn relying party from `{bind_url}`: a relying-party id \
                 must be a domain, not the IP literal `{host}` — serve the console behind a \
                 named host or bind loopback to use passkeys"
            )))
        } else {
            Ok(Self {
                rp_id: host.clone(),
                rp_origin: format!("{}://{host}{port}", url.scheme()),
                rp_name: default_rp_name(),
            })
        }
    }
}

/// Is this URL host the machine itself, in any of its spellings?
fn bind_host_is_local(host: &str) -> bool {
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback() || ip.is_unspecified(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

/// The host a human should be sent to for the console, given the address
/// the node bound (`server.bind_address`).
///
/// Every loopback or unspecified bind becomes `localhost`: browsers refuse
/// an IP-literal WebAuthn origin, so a first-launch link printed as
/// `http://127.0.0.1:<port>/…` can never complete passkey enrolment even
/// when the relying party is derived correctly (#935). Any other address
/// keeps its spelling, IPv6 bracketed for a URL (`http://[fd00::1]:…`).
pub fn public_bind_host(bind_address: &str) -> String {
    let raw = bind_address.trim();
    match raw.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_loopback() || ip.is_unspecified() => "localhost".to_string(),
        Ok(std::net::IpAddr::V6(v6)) => format!("[{v6}]"),
        Ok(_) => raw.to_string(),
        // `localhost` itself, and any name that slips past the server's
        // IP-literal bind guard, keeps its spelling.
        Err(_) => raw.to_string(),
    }
}

/// Top-level state object shared into every Xerj Console handler.
#[derive(Clone)]
pub struct ConsoleState {
    pub engine: Engine,
    pub started_at: StartedAt,
    pub node_id: Arc<String>,
    pub cluster_mode: ClusterMode,
    pub rp: Arc<RpConfig>,

    /// Live WebAuthn challenges. `pending_challenges[challenge_b64u]
    /// = PendingChallenge`. Capped at 1024 entries; stale entries are
    /// pruned on every insert via `prune_pending_challenges`.
    pub pending_challenges: Arc<DashMap<String, PendingChallenge>>,

    /// Live enrollment sessions (post-magic-link, pre-passkey-finish).
    pub enrollment_sessions: Arc<DashMap<String, EnrollmentSession>>,

    /// Per-IP rate-limit counters for auth endpoints. Reset every minute
    /// on first hit after the window. Map key is `"<ip>:<endpoint>"`.
    pub auth_rate_counters: Arc<DashMap<String, RateWindow>>,

    /// HMAC key used to sign session cookies (derived from the
    /// `data_dir/.xerj_master_key` file or the `XERJ_CONSOLE_KEY`
    /// env var; generated on first start and persisted).
    pub master_key: Arc<[u8; 32]>,

    /// Serializes magic-link redemption (#76 S5-3). The single-use check and
    /// the `mark_magic_link_used` commit are separated by several awaits and
    /// the store's mark-used is itself get→delete→create, so two concurrent
    /// redeems of one token could both pass the used-check and both mint a
    /// session. Holding this gate across the whole check→consume makes the
    /// transition atomic. Redemptions are rare (enrollment/recovery), so a
    /// single gate is cheaper than a per-token map and needs no cleanup.
    pub redeem_gate: Arc<tokio::sync::Mutex<()>>,

    /// Reverse proxies whose `X-Forwarded-For` / `X-Real-IP` we believe
    /// (#76 S5-4). Empty by default — an unconfigured node derives client
    /// identity from the socket peer only, so the auth rate limiter cannot be
    /// side-stepped with a forged header. Populated from
    /// `server.trusted_proxies` by `xerj-server` at startup.
    pub trusted_proxies: Arc<TrustedProxies>,
}

#[derive(Debug, Clone, Copy)]
pub struct RateWindow {
    pub count: u32,
    pub window_start_ms: i64,
}

impl ConsoleState {
    pub fn new(
        engine: Engine,
        node_id: String,
        master_key: [u8; 32],
        cluster_mode: ClusterMode,
    ) -> Self {
        Self::new_with_rp(
            engine,
            node_id,
            master_key,
            cluster_mode,
            RpConfig::default(),
        )
    }

    pub fn new_with_rp(
        engine: Engine,
        node_id: String,
        master_key: [u8; 32],
        cluster_mode: ClusterMode,
        rp: RpConfig,
    ) -> Self {
        Self {
            engine,
            started_at: StartedAt(now_epoch_ms()),
            node_id: Arc::new(node_id),
            cluster_mode,
            rp: Arc::new(rp),
            pending_challenges: Arc::new(DashMap::new()),
            enrollment_sessions: Arc::new(DashMap::new()),
            auth_rate_counters: Arc::new(DashMap::new()),
            master_key: Arc::new(master_key),
            redeem_gate: Arc::new(tokio::sync::Mutex::new(())),
            trusted_proxies: Arc::new(TrustedProxies::none()),
        }
    }

    /// Declare the reverse proxies whose forwarding headers may be believed.
    ///
    /// Builder-style so the safe default (trust nothing) is what you get
    /// unless a caller opts in explicitly.
    pub fn with_trusted_proxies(mut self, trusted: TrustedProxies) -> Self {
        self.trusted_proxies = Arc::new(trusted);
        self
    }
}

impl TrustedProxySource for ConsoleState {
    fn trusted_proxies(&self) -> &TrustedProxies {
        &self.trusted_proxies
    }
}

impl std::fmt::Debug for ConsoleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsoleState")
            .field("node_id", &self.node_id)
            .field("started_at", &self.started_at.0)
            .field("pending_challenges", &self.pending_challenges.len())
            .field("enrollment_sessions", &self.enrollment_sessions.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod rp_config_tests {
    use super::*;

    /// The default bind is `127.0.0.1` — before #935 this was exactly the
    /// node that could not enrol on any port but 9200.
    #[test]
    fn loopback_ipv4_bind_maps_to_localhost_with_the_bound_port() {
        let rp = RpConfig::from_bind_url("http://127.0.0.1:9550").unwrap();
        assert_eq!(rp.rp_id, "localhost");
        assert_eq!(rp.rp_origin, "http://localhost:9550");
    }

    #[test]
    fn loopback_ipv6_bind_maps_to_localhost() {
        let rp = RpConfig::from_bind_url("http://[::1]:9550").unwrap();
        assert_eq!(rp.rp_id, "localhost");
        assert_eq!(rp.rp_origin, "http://localhost:9550");
    }

    #[test]
    fn localhost_name_and_unspecified_binds_map_to_localhost() {
        let a = RpConfig::from_bind_url("http://localhost:9550").unwrap();
        let b = RpConfig::from_bind_url("http://0.0.0.0:9200").unwrap();
        let c = RpConfig::from_bind_url("http://[::]:9200").unwrap();
        assert_eq!(
            (a.rp_id.as_str(), a.rp_origin.as_str()),
            ("localhost", "http://localhost:9550")
        );
        assert_eq!(
            (b.rp_id.as_str(), b.rp_origin.as_str()),
            ("localhost", "http://localhost:9200")
        );
        assert_eq!(
            (c.rp_id.as_str(), c.rp_origin.as_str()),
            ("localhost", "http://localhost:9200")
        );
    }

    #[test]
    fn non_loopback_ip_bind_is_refused_with_the_address_named() {
        let err = RpConfig::from_bind_url("http://192.168.1.5:9200").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("192.168.1.5"),
            "the refusal must name the address so the operator can act: {msg}"
        );
        assert!(RpConfig::from_bind_url("http://[fd00::1]:9200").is_err());
    }

    #[test]
    fn named_host_is_used_verbatim() {
        let rp = RpConfig::from_bind_url("http://console.example.com:9200").unwrap();
        assert_eq!(rp.rp_id, "console.example.com");
        assert_eq!(rp.rp_origin, "http://console.example.com:9200");
    }

    #[test]
    fn portless_url_keeps_a_portless_origin() {
        let rp = RpConfig::from_bind_url("http://localhost").unwrap();
        assert_eq!(rp.rp_id, "localhost");
        assert_eq!(rp.rp_origin, "http://localhost");
    }

    #[test]
    fn unparseable_url_is_refused() {
        assert!(RpConfig::from_bind_url("not a url").is_err());
        assert!(RpConfig::from_bind_url("http://").is_err());
    }

    #[test]
    fn public_bind_host_normalises_every_loopback_spelling() {
        assert_eq!(public_bind_host("127.0.0.1"), "localhost");
        assert_eq!(public_bind_host("::1"), "localhost");
        assert_eq!(public_bind_host("0.0.0.0"), "localhost");
        assert_eq!(public_bind_host("::"), "localhost");
        assert_eq!(public_bind_host("localhost"), "localhost");
    }

    #[test]
    fn public_bind_host_keeps_non_loopback_addresses_bracketing_ipv6() {
        assert_eq!(public_bind_host("192.168.1.5"), "192.168.1.5");
        assert_eq!(public_bind_host("fd00::1"), "[fd00::1]");
    }
}
