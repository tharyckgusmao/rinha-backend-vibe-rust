#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct FraudRequest {
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Debug, Deserialize)]
pub struct Transaction {
    pub amount: f64,
    pub installments: u32,
    pub requested_at: String,
}

#[derive(Debug, Deserialize)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f64,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f64,
}

#[derive(Debug, Serialize)]
pub struct FraudResponse {
    pub approved: bool,
    pub fraud_score: f64,
}

#[derive(Debug)]
pub struct FraudDecision {
    pub approved: bool,
    pub fraud_score: f64,
}

impl From<FraudDecision> for FraudResponse {
    fn from(value: FraudDecision) -> Self {
        Self {
            approved: value.approved,
            fraud_score: value.fraud_score,
        }
    }
}
