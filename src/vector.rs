use std::collections::HashMap;

use time::{OffsetDateTime, Weekday, format_description::well_known::Rfc3339};

use crate::{
    config::{NormalizationConfig, VectorizerConfig},
    domain::FraudRequest,
};

pub type QueryVector = [f32; 14];

#[derive(Clone)]
pub struct Vectorizer {
    normalization: NormalizationConfig,
    mcc_risk: HashMap<String, f32>,
}

impl Vectorizer {
    pub fn new(config: VectorizerConfig) -> Self {
        Self {
            normalization: config.normalization,
            mcc_risk: config.mcc_risk,
        }
    }

    pub fn vectorize(&self, request: &FraudRequest) -> Result<QueryVector, VectorizeError> {
        let requested_at = parse_timestamp(&request.transaction.requested_at)?;

        let amount_vs_avg = if request.customer.avg_amount <= 0.0 {
            1.0
        } else {
            (request.transaction.amount as f32 / request.customer.avg_amount as f32)
                / self.normalization.amount_vs_avg_ratio
        };

        let minutes_since_last_tx = match &request.last_transaction {
            Some(last) => {
                let last_timestamp = parse_timestamp(&last.timestamp)?;
                let seconds = (requested_at - last_timestamp).whole_seconds().max(0) as f32;
                clamp(seconds / 60.0 / self.normalization.max_minutes)
            }
            None => -1.0,
        };

        let km_from_last_tx = match &request.last_transaction {
            Some(last) => clamp(last.km_from_current as f32 / self.normalization.max_km),
            None => -1.0,
        };

        Ok([
            clamp(request.transaction.amount as f32 / self.normalization.max_amount),
            clamp(request.transaction.installments as f32 / self.normalization.max_installments),
            clamp(amount_vs_avg),
            requested_at.hour() as f32 / 23.0,
            weekday_index(requested_at.weekday()) as f32 / 6.0,
            minutes_since_last_tx,
            km_from_last_tx,
            clamp(request.terminal.km_from_home as f32 / self.normalization.max_km),
            clamp(request.customer.tx_count_24h as f32 / self.normalization.max_tx_count_24h),
            bool_to_feature(request.terminal.is_online),
            bool_to_feature(request.terminal.card_present),
            bool_to_feature(!merchant_is_known(request)),
            *self.mcc_risk.get(&request.merchant.mcc).unwrap_or(&0.5),
            clamp(request.merchant.avg_amount as f32 / self.normalization.max_merchant_avg_amount),
        ])
    }

    pub fn vectorize_json_bytes(&self, body: &[u8]) -> Result<QueryVector, VectorizeError> {
        let transaction = find_key(body, b"\"transaction\"").ok_or(VectorizeError::InvalidJson)?;
        let customer = find_key(body, b"\"customer\"").ok_or(VectorizeError::InvalidJson)?;
        let merchant = find_key(body, b"\"merchant\"").ok_or(VectorizeError::InvalidJson)?;
        let terminal = find_key(body, b"\"terminal\"").ok_or(VectorizeError::InvalidJson)?;
        let last_transaction =
            find_key(body, b"\"last_transaction\"").ok_or(VectorizeError::InvalidJson)?;

        let amount = number_after(body, transaction, b"\"amount\"")?;
        let installments = number_after(body, transaction, b"\"installments\"")?;
        let requested_at = string_after(body, transaction, b"\"requested_at\"")?;
        let requested_minutes = timestamp_minutes(requested_at)?;

        let customer_avg_amount = number_after(body, customer, b"\"avg_amount\"")?;
        let tx_count_24h = number_after(body, customer, b"\"tx_count_24h\"")?;
        let known_merchants = array_after(body, customer, b"\"known_merchants\"")?;

        let merchant_id = string_after(body, merchant, b"\"id\"")?;
        let mcc = string_after(body, merchant, b"\"mcc\"")?;
        let merchant_avg_amount = number_after(body, merchant, b"\"avg_amount\"")?;

        let is_online = bool_after(body, terminal, b"\"is_online\"")?;
        let card_present = bool_after(body, terminal, b"\"card_present\"")?;
        let km_from_home = number_after(body, terminal, b"\"km_from_home\"")?;

        let has_last_transaction =
            !starts_with_json_null(body, value_start(body, last_transaction)?);
        let (minutes_since_last_tx, km_from_last_tx) = if has_last_transaction {
            let last_timestamp = string_after(body, last_transaction, b"\"timestamp\"")?;
            let last_minutes = timestamp_minutes(last_timestamp)?;
            let minutes = requested_minutes.saturating_sub(last_minutes) as f32;
            let km = number_after(body, last_transaction, b"\"km_from_current\"")?;
            (
                clamp(minutes / self.normalization.max_minutes),
                clamp(km / self.normalization.max_km),
            )
        } else {
            (-1.0, -1.0)
        };

        let amount_vs_avg = if customer_avg_amount <= 0.0 {
            1.0
        } else {
            (amount / customer_avg_amount) / self.normalization.amount_vs_avg_ratio
        };

        Ok([
            clamp(amount / self.normalization.max_amount),
            clamp(installments / self.normalization.max_installments),
            clamp(amount_vs_avg),
            hour_from_timestamp(requested_at)? as f32 / 23.0,
            weekday_from_ymd(requested_at)? as f32 / 6.0,
            minutes_since_last_tx,
            km_from_last_tx,
            clamp(km_from_home / self.normalization.max_km),
            clamp(tx_count_24h / self.normalization.max_tx_count_24h),
            bool_to_feature(is_online),
            bool_to_feature(card_present),
            bool_to_feature(!array_contains_quoted(known_merchants, merchant_id)),
            *self
                .mcc_risk
                .get(std::str::from_utf8(mcc).map_err(|_| VectorizeError::InvalidJson)?)
                .unwrap_or(&0.5),
            clamp(merchant_avg_amount / self.normalization.max_merchant_avg_amount),
        ])
    }
}

#[derive(Debug)]
pub enum VectorizeError {
    InvalidTimestamp(time::error::Parse),
    InvalidJson,
}

impl std::fmt::Display for VectorizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTimestamp(source) => write!(f, "invalid RFC3339 timestamp: {}", source),
            Self::InvalidJson => write!(f, "invalid fraud request json"),
        }
    }
}

impl std::error::Error for VectorizeError {}

fn parse_timestamp(value: &str) -> Result<OffsetDateTime, VectorizeError> {
    OffsetDateTime::parse(value, &Rfc3339).map_err(VectorizeError::InvalidTimestamp)
}

fn clamp(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn weekday_index(weekday: Weekday) -> u8 {
    match weekday {
        Weekday::Monday => 0,
        Weekday::Tuesday => 1,
        Weekday::Wednesday => 2,
        Weekday::Thursday => 3,
        Weekday::Friday => 4,
        Weekday::Saturday => 5,
        Weekday::Sunday => 6,
    }
}

fn bool_to_feature(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn merchant_is_known(request: &FraudRequest) -> bool {
    request
        .customer
        .known_merchants
        .iter()
        .any(|merchant_id| merchant_id == &request.merchant.id)
}

fn find_key(body: &[u8], key: &[u8]) -> Option<usize> {
    let first = key[0];
    let klen = key.len();
    let mut i = 0;
    while i + klen <= body.len() {
        if body[i] == first && body[i..i + klen] == *key {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_key_from(body: &[u8], from: usize, key: &[u8]) -> Option<usize> {
    let first = key[0];
    let klen = key.len();
    let mut i = from;
    while i + klen <= body.len() {
        if body[i] == first && body[i..i + klen] == *key {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn value_start(body: &[u8], key_pos: usize) -> Result<usize, VectorizeError> {
    let colon = body
        .get(key_pos..)
        .ok_or(VectorizeError::InvalidJson)?
        .iter()
        .position(|byte| *byte == b':')
        .ok_or(VectorizeError::InvalidJson)?
        + key_pos;
    let mut idx = colon + 1;
    while matches!(body.get(idx), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        idx += 1;
    }
    Ok(idx)
}

fn number_after(body: &[u8], from: usize, key: &[u8]) -> Result<f32, VectorizeError> {
    let key_pos = find_key_from(body, from, key).ok_or(VectorizeError::InvalidJson)?;
    let start = value_start(body, key_pos)?;
    parse_f32_fast(&body[start..])
}

/// Fast inline f32 parser — avoids str conversion and generic parse
#[inline(always)]
fn parse_f32_fast(buf: &[u8]) -> Result<f32, VectorizeError> {
    let mut i = 0;
    let neg = if buf.get(i) == Some(&b'-') { i += 1; true } else { false };

    let mut int_part = 0i64;
    while i < buf.len() && buf[i].is_ascii_digit() {
        int_part = int_part * 10 + (buf[i] - b'0') as i64;
        i += 1;
    }

    let mut frac = 0i64;
    let mut frac_div = 1i64;
    if i < buf.len() && buf[i] == b'.' {
        i += 1;
        while i < buf.len() && buf[i].is_ascii_digit() {
            frac = frac * 10 + (buf[i] - b'0') as i64;
            frac_div *= 10;
            i += 1;
        }
    }

    let val = int_part as f32 + frac as f32 / frac_div as f32;
    Ok(if neg { -val } else { val })
}

fn bool_after(body: &[u8], from: usize, key: &[u8]) -> Result<bool, VectorizeError> {
    let key_pos = find_key_from(body, from, key).ok_or(VectorizeError::InvalidJson)?;
    let start = value_start(body, key_pos)?;
    if body
        .get(start..start + 4)
        .is_some_and(|value| value == b"true")
    {
        Ok(true)
    } else if body
        .get(start..start + 5)
        .is_some_and(|value| value == b"false")
    {
        Ok(false)
    } else {
        Err(VectorizeError::InvalidJson)
    }
}

fn string_after<'a>(body: &'a [u8], from: usize, key: &[u8]) -> Result<&'a [u8], VectorizeError> {
    let key_pos = find_key_from(body, from, key).ok_or(VectorizeError::InvalidJson)?;
    let start = value_start(body, key_pos)?;
    if body.get(start) != Some(&b'"') {
        return Err(VectorizeError::InvalidJson);
    }
    let end = body
        .get(start + 1..)
        .ok_or(VectorizeError::InvalidJson)?
        .iter()
        .position(|byte| *byte == b'"')
        .ok_or(VectorizeError::InvalidJson)?
        + start
        + 1;
    Ok(&body[start + 1..end])
}

fn array_after<'a>(body: &'a [u8], from: usize, key: &[u8]) -> Result<&'a [u8], VectorizeError> {
    let key_pos = find_key_from(body, from, key).ok_or(VectorizeError::InvalidJson)?;
    let start = value_start(body, key_pos)?;
    if body.get(start) != Some(&b'[') {
        return Err(VectorizeError::InvalidJson);
    }
    let end = body
        .get(start + 1..)
        .ok_or(VectorizeError::InvalidJson)?
        .iter()
        .position(|byte| *byte == b']')
        .ok_or(VectorizeError::InvalidJson)?
        + start
        + 1;
    Ok(&body[start..=end])
}

fn starts_with_json_null(body: &[u8], start: usize) -> bool {
    body.get(start..start + 4)
        .is_some_and(|value| value == b"null")
}

fn array_contains_quoted(array: &[u8], needle: &[u8]) -> bool {
    array
        .windows(needle.len())
        .position(|window| window == needle)
        .is_some_and(|idx| {
            idx > 0
                && idx + needle.len() < array.len()
                && array[idx - 1] == b'"'
                && array[idx + needle.len()] == b'"'
        })
}

fn timestamp_minutes(timestamp: &[u8]) -> Result<i64, VectorizeError> {
    let (year, month, day, hour, minute) = timestamp_parts(timestamp)?;
    let days = days_from_civil(year, month, day);
    Ok(days * 24 * 60 + hour as i64 * 60 + minute as i64)
}

fn hour_from_timestamp(timestamp: &[u8]) -> Result<u32, VectorizeError> {
    Ok(timestamp_parts(timestamp)?.3)
}

fn weekday_from_ymd(timestamp: &[u8]) -> Result<u32, VectorizeError> {
    let (year, month, day, _, _) = timestamp_parts(timestamp)?;
    let days = days_from_civil(year, month, day);
    Ok((days + 3).rem_euclid(7) as u32)
}

fn timestamp_parts(timestamp: &[u8]) -> Result<(i32, u32, u32, u32, u32), VectorizeError> {
    if timestamp.len() < 16 {
        return Err(VectorizeError::InvalidJson);
    }
    Ok((
        parse_u32(&timestamp[0..4])? as i32,
        parse_u32(&timestamp[5..7])?,
        parse_u32(&timestamp[8..10])?,
        parse_u32(&timestamp[11..13])?,
        parse_u32(&timestamp[14..16])?,
    ))
}

fn parse_u32(value: &[u8]) -> Result<u32, VectorizeError> {
    let mut acc = 0_u32;
    for byte in value {
        if !byte.is_ascii_digit() {
            return Err(VectorizeError::InvalidJson);
        }
        acc = acc * 10 + (byte - b'0') as u32;
    }
    Ok(acc)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - (month <= 2) as i32;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as i64
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        config::{NormalizationConfig, VectorizerConfig},
        domain::{Customer, FraudRequest, LastTransaction, Merchant, Terminal, Transaction},
    };

    use super::Vectorizer;

    #[test]
    fn vectorizes_documented_legit_example() {
        let vectorizer = sample_vectorizer();
        let request = FraudRequest {
            id: "tx-1329056812".into(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".into(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };

        let vector = vectorizer.vectorize(&request).unwrap();

        assert_close(vector[0], 0.004112);
        assert_close(vector[1], 2.0 / 12.0);
        assert_close(vector[2], 0.05);
        assert_close(vector[3], 18.0 / 23.0);
        assert_close(vector[4], 2.0 / 6.0);
        assert_eq!(vector[5], -1.0);
        assert_eq!(vector[6], -1.0);
        assert_close(vector[7], 0.02923);
        assert_close(vector[8], 3.0 / 20.0);
        assert_eq!(vector[9], 0.0);
        assert_eq!(vector[10], 1.0);
        assert_eq!(vector[11], 0.0);
        assert_close(vector[12], 0.15);
        assert_close(vector[13], 0.006025);
    }

    #[test]
    fn vectorizes_documented_fraud_example() {
        let vectorizer = sample_vectorizer();
        let request = FraudRequest {
            id: "tx-3330991687".into(),
            transaction: Transaction {
                amount: 9505.97,
                installments: 10,
                requested_at: "2026-03-14T05:15:12Z".into(),
            },
            customer: Customer {
                avg_amount: 81.28,
                tx_count_24h: 20,
                known_merchants: vec!["MERC-008".into(), "MERC-007".into(), "MERC-005".into()],
            },
            merchant: Merchant {
                id: "MERC-068".into(),
                mcc: "7802".into(),
                avg_amount: 54.86,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 952.27,
            },
            last_transaction: None,
        };

        let vector = vectorizer.vectorize(&request).unwrap();

        assert_close(vector[0], 0.950597);
        assert_close(vector[1], 10.0 / 12.0);
        assert_eq!(vector[2], 1.0);
        assert_close(vector[3], 5.0 / 23.0);
        assert_close(vector[4], 5.0 / 6.0);
        assert_eq!(vector[5], -1.0);
        assert_eq!(vector[6], -1.0);
        assert_close(vector[7], 0.95227);
        assert_eq!(vector[8], 1.0);
        assert_eq!(vector[9], 0.0);
        assert_eq!(vector[10], 1.0);
        assert_eq!(vector[11], 1.0);
        assert_close(vector[12], 0.75);
        assert_close(vector[13], 0.005486);
    }

    #[test]
    fn vectorizes_last_transaction_and_unknown_mcc_defaults() {
        let vectorizer = sample_vectorizer();
        let request = FraudRequest {
            id: "tx-with-last".into(),
            transaction: Transaction {
                amount: 120.0,
                installments: 1,
                requested_at: "2026-03-11T20:23:35Z".into(),
            },
            customer: Customer {
                avg_amount: 60.0,
                tx_count_24h: 2,
                known_merchants: vec!["MERC-001".into()],
            },
            merchant: Merchant {
                id: "MERC-002".into(),
                mcc: "9999".into(),
                avg_amount: 300.0,
            },
            terminal: Terminal {
                is_online: true,
                card_present: false,
                km_from_home: 13.7090520965,
            },
            last_transaction: Some(LastTransaction {
                timestamp: "2026-03-11T14:58:35Z".into(),
                km_from_current: 18.8626479774,
            }),
        };

        let vector = vectorizer.vectorize(&request).unwrap();

        assert_close(vector[5], 325.0 / 1440.0);
        assert_close(vector[6], 18.8626479774_f32 / 1000.0);
        assert_eq!(vector[9], 1.0);
        assert_eq!(vector[10], 0.0);
        assert_eq!(vector[11], 1.0);
        assert_eq!(vector[12], 0.5);
    }

    #[test]
    fn vectorizes_json_bytes_like_typed_request() {
        let vectorizer = sample_vectorizer();
        let json = br#"{"id":"tx-with-last","transaction":{"amount":120.0,"installments":1,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":60.0,"tx_count_24h":2,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-002","mcc":"9999","avg_amount":300.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}"#;

        let vector = vectorizer.vectorize_json_bytes(json).unwrap();

        assert_close(vector[0], 0.012);
        assert_close(vector[2], 0.2);
        assert_close(vector[5], 325.0 / 1440.0);
        assert_close(vector[6], 18.8626479774_f32 / 1000.0);
        assert_eq!(vector[9], 1.0);
        assert_eq!(vector[10], 0.0);
        assert_eq!(vector[11], 1.0);
        assert_eq!(vector[12], 0.5);
    }

    fn sample_vectorizer() -> Vectorizer {
        let mut mcc_risk = HashMap::new();
        mcc_risk.insert("5411".into(), 0.15);
        mcc_risk.insert("7802".into(), 0.75);

        Vectorizer::new(VectorizerConfig {
            normalization: NormalizationConfig {
                max_amount: 10000.0,
                max_installments: 12.0,
                amount_vs_avg_ratio: 10.0,
                max_minutes: 1440.0,
                max_km: 1000.0,
                max_tx_count_24h: 20.0,
                max_merchant_avg_amount: 10000.0,
            },
            mcc_risk,
        })
    }

    fn assert_close(actual: f32, expected: f32) {
        let delta = (actual - expected).abs();
        assert!(
            delta < 0.0001,
            "expected {expected:.6}, got {actual:.6} (delta={delta:.6})"
        );
    }
}
