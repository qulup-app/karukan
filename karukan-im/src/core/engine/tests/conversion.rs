use super::*;

#[test]
fn test_shift_space_inserts_halfwidth_space() {
    let mut engine = InputMethodEngine::new();
    // In empty hiragana state, Shift+Space → half-width space committed
    let result = engine.process_key(&press_shift(' '));
    assert!(result.consumed);
    assert!(result.actions.iter().any(|a| matches!(a, EngineAction::Commit(s) if s == " ")));
}

#[test]
fn test_segment_shrink_and_expand() {
    let mut engine = InputMethodEngine::new();
    // Compose "あいうえお" and start conversion (model not loaded → fallback/dict candidates)
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    engine.process_key(&press('u'));
    engine.process_key(&press('e'));
    engine.process_key(&press('o'));
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // Shift+Left: shrink segment from 5 chars to 4 (creates 2nd segment with 1 char)
    let r = engine.process_key(&press_shift_key(Keysym::LEFT));
    assert!(r.consumed);
    assert_eq!(engine.segments.len(), 2);
    assert_eq!(engine.segments[0].reading, "あいうえ");
    assert_eq!(engine.segments[1].reading, "お");
    assert_eq!(engine.current_segment, 0);

    // Right: move to next segment
    let r = engine.process_key(&press_key(Keysym::RIGHT));
    assert!(r.consumed);
    assert_eq!(engine.current_segment, 1);

    // Left: move back to first segment
    let r = engine.process_key(&press_key(Keysym::LEFT));
    assert!(r.consumed);
    assert_eq!(engine.current_segment, 0);

    // Shift+Right: expand segment back to 5 chars (absorb 2nd segment)
    let r = engine.process_key(&press_shift_key(Keysym::RIGHT));
    assert!(r.consumed);
    assert_eq!(engine.segments.len(), 1);
    assert_eq!(engine.segments[0].reading, "あいうえお");
}

#[test]
fn test_conversion_char_commits_and_continues() {
    let mut engine = InputMethodEngine::new();

    // Type "あい" and enter conversion
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // Type 'k' during conversion → should commit candidate and start new input
    let result = engine.process_key(&press('k'));
    assert!(result.consumed);

    // Should have committed the conversion
    let has_commit = result
        .actions
        .iter()
        .any(|a| matches!(a, EngineAction::Commit(_)));
    assert!(has_commit, "Should have a commit action");

    // Should now be in Composing with 'k' in preedit
    assert!(matches!(engine.state(), InputState::Composing { .. }));
    assert_eq!(engine.preedit().unwrap().text(), "k");
}

#[test]
fn test_conversion_char_commits_and_continues_romaji() {
    let mut engine = InputMethodEngine::new();

    // Type "あ" and enter conversion
    engine.process_key(&press('a'));
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // Type 'k', 'a' → commits conversion, then starts "か"
    engine.process_key(&press('k'));
    assert!(matches!(engine.state(), InputState::Composing { .. }));
    assert_eq!(engine.preedit().unwrap().text(), "k");

    engine.process_key(&press('a'));
    assert_eq!(engine.preedit().unwrap().text(), "か");
}

#[test]
fn test_alphabet_mode_space_inserts_literal_space() {
    let mut engine = InputMethodEngine::new();

    // Enter alphabet mode via Shift+N
    engine.process_key(&press_shift('N'));
    assert!(engine.input_mode == InputMode::Alphabet);

    // Type "ew"
    engine.process_key(&press('e'));
    engine.process_key(&press('w'));
    assert_eq!(engine.preedit().unwrap().text(), "New");

    // Space → should insert literal space, NOT start conversion
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Composing { .. }));
    assert_eq!(engine.preedit().unwrap().text(), "New ");

    // Type "york"
    engine.process_key(&press('y'));
    engine.process_key(&press('o'));
    engine.process_key(&press('r'));
    engine.process_key(&press('k'));
    assert_eq!(engine.preedit().unwrap().text(), "New york");
}
