//! Fuzzy romaji repair: generate conversion candidates from typo-corrected readings.

use super::*;

impl InputMethodEngine {
    /// Generate fuzzy repair candidates from the current input buffer.
    ///
    /// Called from `start_conversion` when the buffer contains stranded ASCII.
    /// Generates corrected reading hypotheses, filters through dictionary lookup,
    /// converts via the model, and returns candidates sorted by priority.
    ///
    /// Returns empty Vec if no ASCII is found or fuzzy repair is disabled.
    pub(super) fn fuzzy_repair_candidates(&mut self) -> Vec<AnnotatedCandidate> {
        if !self.config.fuzzy_repair_enabled {
            return vec![];
        }

        let buffer = self.input_buf.text.clone();
        let hypotheses: Vec<_> = karukan_engine::generate_hypotheses(&buffer, 100)
            .into_iter()
            .filter(|h| !h.reading.chars().any(|c| c.is_ascii_alphabetic()))
            .collect();
        if hypotheses.is_empty() {
            return vec![];
        }

        self.ensure_kanji_converter();
        let api_context = self.truncate_context_for_api();

        let mut candidates: Vec<AnnotatedCandidate> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for hyp in &hypotheses {
            let reading = &hyp.reading;

            // 1. Learning cache hits for this reading
            if let Some(cache) = &self.learning {
                for (surface, _score) in cache.lookup(reading) {
                    if seen.insert(surface.clone()) {
                        candidates.push(
                            AnnotatedCandidate::new(surface, CandidateSource::Learning)
                                .with_reading(Some(reading.clone())),
                        );
                    }
                }
            }

            // 2. Dictionary hits
            let dict_hits = self.search_dictionaries(reading, 5);
            for ac in dict_hits {
                if seen.insert(ac.text.clone()) {
                    candidates.push(
                        AnnotatedCandidate::new(ac.text, ac.source)
                            .with_reading(Some(reading.clone())),
                    );
                }
            }
        }

        // 3. Model conversion — cap at 5 model calls to bound latency
        let readings_to_convert: Vec<&str> = if candidates.is_empty() {
            hypotheses.iter().take(5).map(|h| h.reading.as_str()).collect()
        } else {
            // Only convert readings that had dict/learning hits
            let dict_readings: std::collections::HashSet<String> = candidates
                .iter()
                .filter_map(|c| c.reading.clone())
                .collect();
            hypotheses
                .iter()
                .filter(|h| dict_readings.contains(&h.reading))
                .take(5)
                .map(|h| h.reading.as_str())
                .collect()
        };

        for reading in readings_to_convert {
            let model_results = self.run_kana_kanji_conversion(reading, &api_context, 1);
            for text in model_results {
                if seen.insert(text.clone()) {
                    candidates.push(
                        AnnotatedCandidate::new(text, CandidateSource::Model)
                            .with_reading(Some(reading.to_string())),
                    );
                }
            }
        }

        // 4. Kana fallback for all hypotheses
        for hyp in &hypotheses {
            if seen.insert(hyp.reading.clone()) {
                let katakana = karukan_engine::hiragana_to_katakana(&hyp.reading);
                candidates.push(
                    AnnotatedCandidate::new(hyp.reading.clone(), CandidateSource::Fallback)
                        .with_reading(Some(hyp.reading.clone())),
                );
                if seen.insert(katakana.clone()) {
                    candidates.push(
                        AnnotatedCandidate::new(katakana, CandidateSource::Fallback)
                            .with_reading(Some(hyp.reading.clone())),
                    );
                }
            }
        }

        candidates
    }

}
