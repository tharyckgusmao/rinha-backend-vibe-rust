use crate::{index::SearchIndex, vector::Vectorizer};

pub struct AppState {
    index: SearchIndex,
    vectorizer: Vectorizer,
}

impl AppState {
    pub fn new(index: SearchIndex, vectorizer: Vectorizer) -> Self {
        Self { index, vectorizer }
    }

    pub fn index(&self) -> &SearchIndex {
        &self.index
    }

    pub fn vectorizer(&self) -> &Vectorizer {
        &self.vectorizer
    }
}
