"""UNO game state tracking and temporal debounce filter."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PlayedCard:
    class_id: int
    name: str
    color: str
    value: str
    frame_idx: int
    timestamp: float
    valid: bool


class UnoReferee:
    def __init__(self, stability_threshold: int = 3, grace_period: int = 5):
        self.stability_threshold = stability_threshold
        self.grace_period = grace_period

        # Debounce state
        self.current_candidate: Optional[str] = None
        self.candidate_first_frame: int = 0
        self.candidate_first_ts: float = 0.0
        self.stable_frames: int = 0
        self.missing_frames: int = 0

        # Game state
        self.top_color: Optional[str] = None
        self.top_value: Optional[str] = None
        self.history: list[PlayedCard] = []

    @staticmethod
    def parse_card(label: str) -> Tuple[str, str]:
        if label == "wildcard":
            return "any", "any"
        color, value = label.split("_", 1)
        return color, value

    def is_valid_move(self, new_color: str, new_val: str) -> bool:
        if not self.history:
            return True
        if new_color == "any":
            return True
        return new_color == self.top_color or new_val == self.top_value

    def process_frame(
        self,
        detected_id: Optional[int],
        detected_label: Optional[str],
        frame_idx: int,
        timestamp: float,
    ) -> Optional[PlayedCard]:
        """Filters frame noise and updates game state when a play stabilizes."""
        if detected_label is not None:
            self.missing_frames = 0
            last_recorded = self.history[-1].name if self.history else None

            if detected_label != last_recorded:
                if detected_label == self.current_candidate:
                    self.stable_frames += 1
                else:
                    self.current_candidate = detected_label
                    self.candidate_first_frame = frame_idx
                    self.candidate_first_ts = timestamp
                    self.stable_frames = 1

                if self.stable_frames >= self.stability_threshold:
                    color, val = self.parse_card(detected_label)
                    valid = self.is_valid_move(color, val)

                    play = PlayedCard(
                        class_id=detected_id,
                        name=detected_label,
                        color=color,
                        value=val,
                        frame_idx=self.candidate_first_frame,
                        timestamp=self.candidate_first_ts,
                        valid=valid,
                    )
                    self.history.append(play)

                    if valid:
                        self.top_color, self.top_value = color, val

                    self.stable_frames = 0
                    return play
        else:
            self.missing_frames += 1
            if self.missing_frames > self.grace_period:
                self.stable_frames = 0
                self.current_candidate = None

        return None