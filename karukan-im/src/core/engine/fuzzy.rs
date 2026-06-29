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

    /// Generate kana-level repair candidates when the reading is valid kana
    /// but doesn't produce meaningful conversion (likely a wrong-key typo).
    ///
    /// Called from `start_conversion` when normal candidates suggest the reading
    /// is broken Japanese (model just returns the reading as-is, no dict hits).
    pub(super) fn kana_repair_candidates(
        &mut self,
        normal_candidates: &[AnnotatedCandidate],
    ) -> Vec<AnnotatedCandidate> {
        if !self.config.fuzzy_repair_enabled {
            return vec![];
        }

        let reading = self.input_buf.text.clone();
        if reading.is_empty() {
            return vec![];
        }

        // Kana repair only makes sense for pure hiragana readings.
        // reverse_romaji only handles hiragana; katakana passes through
        // unchanged and would generate no useful hypotheses.
        if !karukan_engine::is_pure_hiragana(&reading) {
            return vec![];
        }

        // Only trigger when normal conversion looks broken: no dictionary or
        // learning cache hit. Model-only candidates don't count — the model
        // always hallucinates *some* kanji even for nonsensical readings.
        let has_meaningful = normal_candidates.iter().any(|c| {
            matches!(
                c.source,
                CandidateSource::Learning
                    | CandidateSource::Dictionary
                    | CandidateSource::UserDictionary
            )
        });
        if has_meaningful {
            return vec![];
        }

        let hypotheses: Vec<_> = karukan_engine::generate_kana_hypotheses(&reading, 50)
            .into_iter()
            .filter(|h| !h.reading.chars().any(|c| c.is_ascii_alphabetic()))
            .collect();
        if hypotheses.is_empty() {
            return vec![];
        }

        let mut candidates: Vec<AnnotatedCandidate> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Only collect hypotheses that have dictionary or learning cache hits.
        // Without a dict/learning signal, kana repair has no confidence —
        // unlike stranded ASCII, there is no clear typo marker.
        let mut dict_hit_readings_set = std::collections::HashSet::new();
        let mut dict_hit_readings_vec: Vec<String> = Vec::new();

        for hyp in &hypotheses {
            let reading = &hyp.reading;

            if let Some(cache) = &self.learning {
                for (surface, _score) in cache.lookup(reading) {
                    if seen.insert(surface.clone()) {
                        if dict_hit_readings_set.insert(reading.clone()) {
                            dict_hit_readings_vec.push(reading.clone());
                        }
                        candidates.push(
                            AnnotatedCandidate::new(surface, CandidateSource::Learning)
                                .with_reading(Some(reading.clone())),
                        );
                    }
                }
            }

            let dict_hits = self.search_dictionaries(reading, 5);
            for ac in dict_hits {
                if seen.insert(ac.text.clone()) {
                    if dict_hit_readings_set.insert(reading.clone()) {
                        dict_hit_readings_vec.push(reading.clone());
                    }
                    candidates.push(
                        AnnotatedCandidate::new(ac.text, ac.source)
                            .with_reading(Some(reading.clone())),
                    );
                }
            }
        }

        // No dict/learning hits among any hypothesis → no confident repair
        if candidates.is_empty() {
            return vec![];
        }

        // Model conversion only for readings that had dict/learning hits
        self.ensure_kanji_converter();
        let api_context = self.truncate_context_for_api();

        let readings_to_convert: Vec<String> =
            dict_hit_readings_vec.into_iter().take(5).collect();
        for reading in &readings_to_convert {
            let model_results = self.run_kana_kanji_conversion(reading, &api_context, 1);
            for text in model_results {
                if seen.insert(text.clone()) {
                    candidates.push(
                        AnnotatedCandidate::new(text, CandidateSource::Model)
                            .with_reading(Some(reading.clone())),
                    );
                }
            }
        }

        candidates
    }

}
