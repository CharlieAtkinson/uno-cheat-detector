from src.state_machine import UnoReferee


def test_first_card_always_valid():
    referee = UnoReferee()
    referee.process_frame(1, "red_1", frame_idx=1, timestamp=0.03)
    referee.process_frame(1, "red_1", frame_idx=2, timestamp=0.06)
    play = referee.process_frame(1, "red_1", frame_idx=3, timestamp=0.09)
    assert play is not None
    assert play.valid is True


def test_matching_color_is_valid():
    referee = UnoReferee()
    referee.top_color, referee.top_value = "red", "1"
    assert referee.is_valid_move("red", "skip") is True


def test_mismatched_card_flags_cheat():
    referee = UnoReferee()
    referee.top_color, referee.top_value = "red", "1"
    assert referee.is_valid_move("green", "2") is False


def test_wildcard_always_valid():
    referee = UnoReferee()
    referee.top_color, referee.top_value = "red", "1"
    assert referee.is_valid_move("any", "any") is True