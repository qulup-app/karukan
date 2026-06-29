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

// ── Kana-level fuzzy repair ──────────────────────────────────────────────────

/// Map a single hiragana character to its shortest romaji.
/// Returns an empty string for characters that are not hiragana
/// (katakana, kanji, symbols, digits, ASCII), which are passed through
/// unchanged by `reverse_romaji`.
fn single_kana_to_romaji(c: char) -> &'static str {
    match c {
        'あ' => "a",  'い' => "i",  'う' => "u",  'え' => "e",  'お' => "o",
        'か' => "ka", 'き' => "ki", 'く' => "ku", 'け' => "ke", 'こ' => "ko",
        'さ' => "sa", 'し' => "si", 'す' => "su", 'せ' => "se", 'そ' => "so",
        'た' => "ta", 'ち' => "ti", 'つ' => "tu", 'て' => "te", 'と' => "to",
        'な' => "na", 'に' => "ni", 'ぬ' => "nu", 'ね' => "ne", 'の' => "no",
        'は' => "ha", 'ひ' => "hi", 'ふ' => "hu", 'へ' => "he", 'ほ' => "ho",
        'ま' => "ma", 'み' => "mi", 'む' => "mu", 'め' => "me", 'も' => "mo",
        'や' => "ya", 'ゆ' => "yu", 'よ' => "yo",
        'ら' => "ra", 'り' => "ri", 'る' => "ru", 'れ' => "re", 'ろ' => "ro",
        'わ' => "wa", 'を' => "wo", 'ん' => "nn",
        'が' => "ga", 'ぎ' => "gi", 'ぐ' => "gu", 'げ' => "ge", 'ご' => "go",
        'ざ' => "za", 'じ' => "zi", 'ず' => "zu", 'ぜ' => "ze", 'ぞ' => "zo",
        'だ' => "da", 'ぢ' => "di", 'づ' => "du", 'で' => "de", 'ど' => "do",
        'ば' => "ba", 'び' => "bi", 'ぶ' => "bu", 'べ' => "be", 'ぼ' => "bo",
        'ぱ' => "pa", 'ぴ' => "pi", 'ぷ' => "pu", 'ぺ' => "pe", 'ぽ' => "po",
        'っ' => "xtu",
        'ゃ' => "xya", 'ゅ' => "xyu", 'ょ' => "xyo",
        'ぁ' => "xa",  'ぃ' => "xi",  'ぅ' => "xu",  'ぇ' => "xe",  'ぉ' => "xo",
        _ => "",
    }
}

/// Map a combo kana pair (拗音: consonant kana + small ゃ/ゅ/ょ) to romaji.
/// Returns `None` if the pair is not a recognised combo.
fn combo_kana_to_romaji(first: char, second: char) -> Option<&'static str> {
    match (first, second) {
        ('き', 'ゃ') => Some("kya"), ('き', 'ゅ') => Some("kyu"), ('き', 'ょ') => Some("kyo"),
        ('し', 'ゃ') => Some("sya"), ('し', 'ゅ') => Some("syu"), ('し', 'ょ') => Some("syo"),
        ('ち', 'ゃ') => Some("tya"), ('ち', 'ゅ') => Some("tyu"), ('ち', 'ょ') => Some("tyo"),
        ('に', 'ゃ') => Some("nya"), ('に', 'ゅ') => Some("nyu"), ('に', 'ょ') => Some("nyo"),
        ('ひ', 'ゃ') => Some("hya"), ('ひ', 'ゅ') => Some("hyu"), ('ひ', 'ょ') => Some("hyo"),
        ('み', 'ゃ') => Some("mya"), ('み', 'ゅ') => Some("myu"), ('み', 'ょ') => Some("myo"),
        ('り', 'ゃ') => Some("rya"), ('り', 'ゅ') => Some("ryu"), ('り', 'ょ') => Some("ryo"),
        ('ぎ', 'ゃ') => Some("gya"), ('ぎ', 'ゅ') => Some("gyu"), ('ぎ', 'ょ') => Some("gyo"),
        ('じ', 'ゃ') => Some("zya"), ('じ', 'ゅ') => Some("zyu"), ('じ', 'ょ') => Some("zyo"),
        ('び', 'ゃ') => Some("bya"), ('び', 'ゅ') => Some("byu"), ('び', 'ょ') => Some("byo"),
        ('ぴ', 'ゃ') => Some("pya"), ('ぴ', 'ゅ') => Some("pyu"), ('ぴ', 'ょ') => Some("pyo"),
        _ => None,
    }
}

/// Convert a hiragana string back to the shortest common romaji representation.
///
/// - Combo kana (拗音) are recognised by lookahead at the next character being
///   small ゃ/ゅ/ょ; they consume two characters and emit a single romaji cluster.
/// - Non-hiragana characters (katakana, kanji, ASCII, digits, symbols) are
///   passed through unchanged.
pub fn reverse_romaji(reading: &str) -> String {
    let chars: Vec<char> = reading.chars().collect();
    let mut result = String::new();
    let mut i = 0;
    while i < chars.len() {
        // Attempt combo (拗音) first: current + next is ゃ/ゅ/ょ
        if i + 1 < chars.len() {
            let next = chars[i + 1];
            if matches!(next, 'ゃ' | 'ゅ' | 'ょ') {
                if let Some(romaji) = combo_kana_to_romaji(chars[i], next) {
                    result.push_str(romaji);
                    i += 2;
                    continue;
                }
            }
        }
        // Single kana → romaji (or pass through if not hiragana)
        let romaji = single_kana_to_romaji(chars[i]);
        if romaji.is_empty() {
            result.push(chars[i]);
        } else {
            result.push_str(romaji);
        }
        i += 1;
    }
    result
}

/// Generate alternative kana readings by applying single QWERTY-key substitutions
/// to the reverse-mapped romaji of `reading`, then re-converting through
/// `RomajiConverter`.
///
/// This handles the case where romaji conversion succeeded but produced the wrong
/// kana (e.g. user typed "warashi" → "わらし" instead of "わたし").  There is no
/// stranded ASCII to signal the error; the kana is valid but wrong.
///
/// Only `EditType::Substitution` is attempted here — insertion/deletion/transposition
/// are already handled by `generate_hypotheses` for the PassThrough (stranded ASCII) case.
pub fn generate_kana_hypotheses(reading: &str, max_hypotheses: usize) -> Vec<FuzzyHypothesis> {
    let romaji = reverse_romaji(reading);
    let romaji_chars: Vec<char> = romaji.chars().collect();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut hypotheses: Vec<FuzzyHypothesis> = Vec::new();

    'outer: for pos in 0..romaji_chars.len() {
        let c = romaji_chars[pos];
        for &neighbor in qwerty_neighbors(c) {
            let mut candidate = romaji_chars.clone();
            candidate[pos] = neighbor;
            let candidate_str: String = candidate.iter().collect();
            if let Some(kana) = validate_romaji_segment(&candidate_str) {
                if kana != reading && seen.insert(kana.clone()) {
                    hypotheses.push(FuzzyHypothesis {
                        reading: kana,
                        edit_type: EditType::Substitution,
                    });
                    if hypotheses.len() >= max_hypotheses {
                        break 'outer;
                    }
                }
            }
        }
    }

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

    // ── reverse_romaji ────────────────────────────────────────────────────────

    #[test]
    fn test_reverse_romaji_basic() {
        assert_eq!(reverse_romaji("わたし"), "watasi");
        // ん→nn, じ→zi
        assert_eq!(reverse_romaji("かんじ"), "kannzi");
        // きょ (combo) → kyo, う→u
        assert_eq!(reverse_romaji("きょう"), "kyou");
    }

    #[test]
    fn test_reverse_romaji_combo() {
        assert_eq!(reverse_romaji("しゃしん"), "syasinn");
        assert_eq!(reverse_romaji("にゃ"), "nya");
        assert_eq!(reverse_romaji("りょ"), "ryo");
    }

    #[test]
    fn test_reverse_romaji_non_hiragana_passthrough() {
        // Pure ASCII passes through unchanged
        assert_eq!(reverse_romaji("abc"), "abc");
        // Mixed: hiragana converted, digits/ASCII pass through
        assert_eq!(reverse_romaji("わ123"), "wa123");
    }

    #[test]
    fn test_reverse_romaji_small_kana() {
        // っ → xtu (standalone)
        assert_eq!(reverse_romaji("っ"), "xtu");
        // Small ゃ outside combo → xya
        assert_eq!(reverse_romaji("ゃ"), "xya");
    }

    // ── generate_kana_hypotheses ──────────────────────────────────────────────

    #[test]
    fn test_generate_kana_hypotheses_substitution() {
        // わらし → reverse "warasi" → substitute r→t at pos 2 → "watasi" → わたし
        let hyps = generate_kana_hypotheses("わらし", 100);
        let readings: Vec<&str> = hyps.iter().map(|h| h.reading.as_str()).collect();
        assert!(readings.contains(&"わたし"), "should find わたし from わらし, got: {:?}", readings);
    }

    #[test]
    fn test_generate_kana_hypotheses_no_self() {
        // The original reading must not appear in the hypotheses
        let hyps = generate_kana_hypotheses("わたし", 100);
        let readings: Vec<&str> = hyps.iter().map(|h| h.reading.as_str()).collect();
        assert!(!readings.contains(&"わたし"), "should not contain the original reading");
    }

    #[test]
    fn test_generate_kana_hypotheses_dedup() {
        // Each reading should appear at most once
        let hyps = generate_kana_hypotheses("わたし", 100);
        let readings: Vec<&str> = hyps.iter().map(|h| h.reading.as_str()).collect();
        let unique: std::collections::HashSet<&str> = readings.iter().copied().collect();
        assert_eq!(readings.len(), unique.len(), "duplicate readings in kana hypotheses");
    }

    #[test]
    fn test_generate_kana_hypotheses_edit_type() {
        // All hypotheses must carry EditType::Substitution
        let hyps = generate_kana_hypotheses("わらし", 100);
        for h in &hyps {
            assert_eq!(h.edit_type, EditType::Substitution);
        }
    }
}
