use super::converter::RomajiConverter;

/// The type of edit that produced a hypothesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditType {
    /// A vowel was inserted after a stranded consonant
    Insertion,
    /// A stranded character was deleted
    Deletion,
    /// A stranded character was replaced with a QWERTY neighbor
    Substitution,
    /// Two adjacent characters were transposed
    Transposition,
}

/// A repair hypothesis: a corrected reading + metadata
#[derive(Debug, Clone)]
pub struct FuzzyHypothesis {
    /// The full corrected buffer text (kana only, no ASCII)
    pub reading: String,
    /// What edit was applied
    pub edit_type: EditType,
}

/// QWERTY keyboard neighbor lookup.
pub fn qwerty_neighbors(c: char) -> &'static [char] {
    match c {
        'q' => &['w', 'a', 's'],
        'w' => &['q', 'e', 'a', 's', 'd'],
        'e' => &['w', 'r', 's', 'd', 'f'],
        'r' => &['e', 't', 'd', 'f', 'g'],
        't' => &['r', 'y', 'f', 'g', 'h'],
        'y' => &['t', 'u', 'g', 'h', 'j'],
        'u' => &['y', 'i', 'h', 'j', 'k'],
        'i' => &['u', 'o', 'j', 'k', 'l'],
        'o' => &['i', 'p', 'k', 'l'],
        'p' => &['o', 'l'],
        'a' => &['q', 'w', 's', 'z', 'x'],
        's' => &['a', 'w', 'e', 'd', 'z', 'x'],
        'd' => &['s', 'e', 'r', 'f', 'x', 'c'],
        'f' => &['d', 'r', 't', 'g', 'c', 'v'],
        'g' => &['f', 't', 'y', 'h', 'v', 'b'],
        'h' => &['g', 'y', 'u', 'j', 'b', 'n'],
        'j' => &['h', 'u', 'i', 'k', 'n', 'm'],
        'k' => &['j', 'i', 'o', 'l', 'm'],
        'l' => &['k', 'o', 'p'],
        'z' => &['a', 's', 'x'],
        'x' => &['z', 's', 'd', 'c'],
        'c' => &['x', 'd', 'f', 'v'],
        'v' => &['c', 'f', 'g', 'b'],
        'b' => &['v', 'g', 'h', 'n'],
        'n' => &['b', 'h', 'j', 'm'],
        'm' => &['n', 'j', 'k'],
        _ => &[],
    }
}

/// Find stranded ASCII alphabetic segments in the buffer.
/// Returns (start_char_index, segment_string) pairs.
/// Only finds lowercase ASCII a-z segments that are surrounded by or adjacent to kana.
pub fn find_stranded_ascii(buffer: &str) -> Vec<(usize, String)> {
    let chars: Vec<char> = buffer.chars().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() {
            let start = i;
            let mut seg = String::new();
            while i < chars.len() && chars[i].is_ascii_alphabetic() {
                seg.push(chars[i].to_ascii_lowercase());
                i += 1;
            }
            // Only report it as "stranded" if surrounded by/adjacent to kana
            let before_is_kana = start > 0 && is_kana(chars[start - 1]);
            let after_is_kana = i < chars.len() && is_kana(chars[i]);
            if before_is_kana || after_is_kana {
                result.push((start, seg));
            }
        } else {
            i += 1;
        }
    }

    result
}

fn is_kana(c: char) -> bool {
    // Hiragana: U+3041–U+309F, Katakana: U+30A0–U+30FF (incl. ー U+30FC)
    ('\u{3041}'..='\u{309F}').contains(&c) || ('\u{30A0}'..='\u{30FF}').contains(&c)
}

/// Validate a romaji-only string: push each char through a fresh RomajiConverter,
/// flush, and return the kana output if it contains no lowercase ASCII.
fn validate_romaji_segment(romaji: &str) -> Option<String> {
    let mut conv = RomajiConverter::new();
    for ch in romaji.chars() {
        conv.push(ch);
    }
    conv.flush();
    let out = conv.output().to_string();
    if out.chars().all(|c| !c.is_ascii_lowercase()) && !out.is_empty() {
        Some(out)
    } else {
        None
    }
}

/// Reconstruct the full buffer with `replacement_kana` substituted for the
/// ASCII segment at `seg_start..seg_start+seg_len` (char indices).
fn rebuild(chars: &[char], seg_start: usize, seg_len: usize, replacement_kana: &str) -> String {
    let before: String = chars[..seg_start].iter().collect();
    let after: String = chars[seg_start + seg_len..].iter().collect();
    format!("{}{}{}", before, replacement_kana, after)
}

const VOWELS: [char; 5] = ['a', 'i', 'u', 'e', 'o'];

/// Generate repair hypotheses for a buffer with stranded ASCII.
/// Returns validated hypotheses (clean kana output from RomajiConverter, no passthrough).
/// Caps total hypotheses at `max_hypotheses` (use 100).
pub fn generate_hypotheses(buffer: &str, max_hypotheses: usize) -> Vec<FuzzyHypothesis> {
    let segments = find_stranded_ascii(buffer);
    if segments.is_empty() {
        return vec![];
    }

    let chars: Vec<char> = buffer.chars().collect();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut hypotheses: Vec<FuzzyHypothesis> = Vec::new();

    let add = |reading: String, edit_type: EditType, seen: &mut std::collections::HashSet<String>, hypotheses: &mut Vec<FuzzyHypothesis>| -> bool {
        if hypotheses.len() >= max_hypotheses {
            return false;
        }
        if seen.insert(reading.clone()) {
            hypotheses.push(FuzzyHypothesis { reading, edit_type });
        }
        true
    };

    for (seg_start, seg) in &segments {
        let seg_start = *seg_start;
        let seg_chars: Vec<char> = seg.chars().collect();
        let seg_len = seg_chars.len();

        // ponytail: skip segments with 3+ ASCII chars — too many edits needed to be useful
        if seg_len >= 3 {
            continue;
        }

        // --- Insertion: insert a vowel after each character in the segment ---
        'insertion: for insert_pos in 0..=seg_chars.len() {
            for &vowel in &VOWELS {
                let mut candidate = seg_chars.clone();
                candidate.insert(insert_pos, vowel);
                let candidate_str: String = candidate.iter().collect();
                if let Some(kana) = validate_romaji_segment(&candidate_str) {
                    let full = rebuild(&chars, seg_start, seg_len, &kana);
                    if !add(full, EditType::Insertion, &mut seen, &mut hypotheses) {
                        break 'insertion;
                    }
                }
            }
        }

        if hypotheses.len() >= max_hypotheses {
            break;
        }

        // --- Deletion: remove each character from the segment ---
        'deletion: for del_pos in 0..seg_chars.len() {
            let mut candidate = seg_chars.clone();
            candidate.remove(del_pos);
            if candidate.is_empty() {
                // Deleting everything — just stitch kana together
                let full = rebuild(&chars, seg_start, seg_len, "");
                if !add(full, EditType::Deletion, &mut seen, &mut hypotheses) {
                    break 'deletion;
                }
                continue;
            }
            let candidate_str: String = candidate.iter().collect();
            if let Some(kana) = validate_romaji_segment(&candidate_str) {
                let full = rebuild(&chars, seg_start, seg_len, &kana);
                if !add(full, EditType::Deletion, &mut seen, &mut hypotheses) {
                    break 'deletion;
                }
            }
        }

        if hypotheses.len() >= max_hypotheses {
            break;
        }

        // --- Substitution: replace each character with a QWERTY neighbor ---
        'substitution: for sub_pos in 0..seg_chars.len() {
            for &neighbor in qwerty_neighbors(seg_chars[sub_pos]) {
                let mut candidate = seg_chars.clone();
                candidate[sub_pos] = neighbor;
                let candidate_str: String = candidate.iter().collect();
                if let Some(kana) = validate_romaji_segment(&candidate_str) {
                    let full = rebuild(&chars, seg_start, seg_len, &kana);
                    if !add(full, EditType::Substitution, &mut seen, &mut hypotheses) {
                        break 'substitution;
                    }
                }
            }
        }

        if hypotheses.len() >= max_hypotheses {
            break;
        }

        // --- Transposition: swap adjacent pairs ---
        if seg_chars.len() >= 2 {
            'transposition: for swap_pos in 0..seg_chars.len() - 1 {
                let mut candidate = seg_chars.clone();
                candidate.swap(swap_pos, swap_pos + 1);
                let candidate_str: String = candidate.iter().collect();
                if let Some(kana) = validate_romaji_segment(&candidate_str) {
                    let full = rebuild(&chars, seg_start, seg_len, &kana);
                    if !add(full, EditType::Transposition, &mut seen, &mut hypotheses) {
                        break 'transposition;
                    }
                }
            }
        }

        if hypotheses.len() >= max_hypotheses {
            break;
        }
    }

    // Drop hypotheses that still contain ASCII alphabetic characters —
    // multi-segment repairs leave the other segment's ASCII intact.
    hypotheses.retain(|h| !h.reading.chars().any(|c| c.is_ascii_alphabetic()));
    hypotheses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_stranded_ascii() {
        // "なmこ" → m is between kana
        let result = find_stranded_ascii("なmこ");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, "m");

        // "wsたし" → ws is before kana
        let result = find_stranded_ascii("wsたし");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, "ws");

        // Pure kana — nothing stranded
        let result = find_stranded_ascii("なまこ");
        assert!(result.is_empty());

        // Pure ASCII with no surrounding kana — not stranded
        let result = find_stranded_ascii("abc");
        assert!(result.is_empty());
    }

    #[test]
    fn test_qwerty_neighbors() {
        let n = qwerty_neighbors('a');
        assert!(n.contains(&'s'));
        assert!(n.contains(&'q'));
        assert!(n.contains(&'w'));

        let n = qwerty_neighbors('m');
        assert!(n.contains(&'n'));
        assert!(n.contains(&'j'));
        assert!(n.contains(&'k'));

        // Unknown char → empty
        assert!(qwerty_neighbors('1').is_empty());
    }

    #[test]
    fn test_generate_hypotheses_missing_vowel() {
        // "なmこ": the 'm' is stranded; inserting a vowel after 'm' should yield "なまこ", "なみこ", etc.
        let hyps = generate_hypotheses("なmこ", 100);
        let readings: Vec<&str> = hyps.iter().map(|h| h.reading.as_str()).collect();

        // ma → ま → "なまこ"
        assert!(readings.contains(&"なまこ"), "expected なまこ in {:?}", readings);
        // mi → み → "なみこ"
        assert!(readings.contains(&"なみこ"), "expected なみこ in {:?}", readings);
        // me → め → "なめこ"
        assert!(readings.contains(&"なめこ"), "expected なめこ in {:?}", readings);
    }

    #[test]
    fn test_generate_hypotheses_substitution() {
        // "wsたし": ws substitution s→a gives "wa" → "わ", reconstructed "わたし"
        let hyps = generate_hypotheses("wsたし", 100);
        let readings: Vec<&str> = hyps.iter().map(|h| h.reading.as_str()).collect();
        assert!(readings.contains(&"わたし"), "expected わたし in {:?}", readings);
    }

    #[test]
    fn test_generate_hypotheses_no_ascii() {
        // Pure kana — nothing to repair
        let hyps = generate_hypotheses("なまこ", 100);
        assert!(hyps.is_empty());
    }

    #[test]
    fn test_generate_hypotheses_long_ascii_skipped() {
        // "なmskこ" has 3 ASCII chars — should be skipped
        let hyps = generate_hypotheses("なmskこ", 100);
        assert!(hyps.is_empty());
    }

    #[test]
    fn test_dedup() {
        // Same reading from multiple edits should only appear once
        let hyps = generate_hypotheses("なmこ", 100);
        let readings: Vec<&str> = hyps.iter().map(|h| h.reading.as_str()).collect();
        let unique: std::collections::HashSet<&str> = readings.iter().copied().collect();
        assert_eq!(readings.len(), unique.len(), "duplicate readings found");
    }

    #[test]
    fn test_multi_segment_ascii_filtered() {
        // Two ASCII segments: "なmたsし" — repairing one leaves the other.
        // All hypotheses should be filtered out since they still contain ASCII.
        let hyps = generate_hypotheses("なmたsし", 100);
        for h in &hyps {
            assert!(
                !h.reading.chars().any(|c| c.is_ascii_alphabetic()),
                "hypothesis {:?} contains residual ASCII",
                h.reading
            );
        }
    }
}
