from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import os
import re
from datetime import datetime


@dataclass
class Message:
    speaker: str
    content: str
    timestamp: str  # Note: this is the time id of the message, not the real timestamp of the message (e.g.: D1:7)
    image_caption: Optional[str] = None

    def to_string(self) -> str:
        """Convert the message to a string, including the image caption when present."""
        if self.image_caption is not None:
            return f"[{self.speaker}]: {self.content} [Showing Image]: {self.image_caption}"
        return f"[{self.speaker}]: {self.content}"


@dataclass
class Session:
    date_time: str
    session_id: str
    messages: List[Message]


@dataclass
class Conversation:
    id: str
    speaker_a: str
    speaker_b: str
    sessions: List[Session]


@dataclass
class Question:
    content: str
    answer: str
    evidence: str
    category: int
    date: Optional[str] = None
    answerable: bool = True


def lme_type_to_int(question_type: str) -> int:
    if question_type == "single-session-user":
        return 0
    elif question_type == "single-session-assistant":
        return 1
    elif question_type == "single-session-preference":
        return 2
    elif question_type == "temporal-reasoning":
        return 3
    elif question_type == "knowledge-update":
        return 4
    elif question_type == "multi-session":
        return 5
    else:
        raise ValueError(f"Invalid question type: {question_type}")


def int_to_lme_type(question_type: int) -> str:
    if question_type == 0:
        return "single-session-user"
    elif question_type == 1:
        return "single-session-assistant"
    elif question_type == 2:
        return "single-session-preference"
    elif question_type == 3:
        return "temporal-reasoning"
    elif question_type == 4:
        return "knowledge-update"
    elif question_type == 5:
        return "multi-session"
    else:
        raise ValueError(f"Invalid question type: {question_type}")


class ConvLoader:
    @staticmethod
    def load_locomo(file_path: str) -> Tuple[List[Conversation], List[List[Question]]]:
        """
        Load the conversations from the locomo dataset. And keep in memory
        """
        conversations = []
        questions = []
        with open(file_path) as f:
            dataset = json.load(f)
            for item in dataset:
                # Load questions
                ques_for_conv = []
                for q in item["qa"]:
                    if q["category"] == 5:
                        continue
                    ques_for_conv.append(
                        Question(
                            content=q["question"],
                            answer=q["answer"],
                            evidence=q["evidence"],
                            category=q["category"],
                        )
                    )
                questions.append(ques_for_conv)
                # Load conversations
                sessions = []
                conv_id = item["sample_id"]
                conversation = Conversation(
                    id=conv_id,
                    sessions=sessions,
                    speaker_a=item["conversation"]["speaker_a"],
                    speaker_b=item["conversation"]["speaker_b"],
                )
                for key, conv in item["conversation"].items():
                    if key.startswith("speaker") or key.endswith("time"):
                        continue
                    time_key = key + "_date_time"
                    session_time = item["conversation"][time_key]
                    session_id = key.split("_")[1]
                    session_messages = []
                    for msg in conv:
                        blip_caption = msg.get("blip_caption", None)
                        session_messages.append(
                            Message(
                                speaker=msg["speaker"],
                                content=msg["text"],
                                timestamp=msg["dia_id"],
                                image_caption=blip_caption,
                            )
                        )
                    sessions.append(
                        Session(
                            date_time=session_time,
                            session_id=session_id,
                            messages=session_messages,
                        )
                    )
                conversations.append(conversation)

        return conversations, questions

    @staticmethod
    def load_longmemeval_s(
        file_path: str,
    ) -> Tuple[List[Conversation], List[List[Question]]]:
        convs = []
        questions = []
        with open(file_path) as f:
            dataset = json.load(f)
            for item in dataset:
                # LongMemEval only has one question per conversation
                question = Question(
                    content=item["question"],
                    answer=item["answer"],
                    evidence=item["answer_session_ids"],
                    category=lme_type_to_int(item["question_type"]),
                    date=item["question_date"],
                    answerable="abstention" not in item["question_id"],
                )
                questions.append([question])
                # Process session by session
                sessions = []
                for session_date, session_id, session_messages in zip(
                    item["haystack_dates"],
                    item["haystack_session_ids"],
                    item["haystack_sessions"],
                ):
                    # formulate time format, long memeval uses xxxx/xx/xx (Weekday) HH:MM
                    # We formulate them as YYYY-MM-DD HH:MM (Weekday) for sorting purposes
                    parts = session_date.split(" ")
                    session_date = f"{parts[0] } {parts[2]} {parts[1]}"
                    messages: List[Message] = [None] * len(session_messages)
                    for i, msg in enumerate(session_messages):
                        messages[i] = Message(
                            speaker=msg["role"],
                            content=msg["content"],
                            timestamp="NA",
                            image_caption=None,
                        )
                    sessions.append(
                        Session(
                            date_time=session_date,
                            session_id=session_id,
                            messages=messages,
                        )
                    )
                convs.append(
                    Conversation(
                        id="NA",
                        sessions=sessions,
                        speaker_a="user",
                        speaker_b="assistant",
                    )
                )
        return convs, questions

    # Month name to number mapping (class-level constant)
    _MONTH_MAP = {
        name: num
        for num, names in enumerate(
            [
                ("january", "jan"),
                ("february", "feb"),
                ("march", "mar"),
                ("april", "apr"),
                ("may",),
                ("june", "jun"),
                ("july", "jul"),
                ("august", "aug"),
                ("september", "sep"),
                ("october", "oct"),
                ("november", "nov"),
                ("december", "dec"),
            ],
            start=1,
        )
        for name in names
    }

    # Regex pattern for parsing "HH:MM am/pm on DD Month, YYYY"
    _TIMESTAMP_PATTERN = re.compile(
        r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})$",
        re.IGNORECASE,
    )

    @staticmethod
    def locomo_time_to_datetime(value: str) -> datetime:
        """Parse dataset timestamps such as "1:56 pm on 8 May, 2023"."""
        value = " ".join(value.split())

        # Try ISO format first if no " on " separator
        if " on " not in value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return datetime.now()

        match = ConvLoader._TIMESTAMP_PATTERN.match(value)
        if not match:
            return datetime.now()

        hour, minute_str, meridiem, day, month_name, year = match.groups()

        hour = int(hour)
        minute = int(minute_str) if minute_str else 0

        is_pm = meridiem.lower() == "pm"
        if is_pm and hour != 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0

        # Parse date components
        month = ConvLoader._MONTH_MAP.get(month_name.lower(), 1)

        return datetime(
            year=int(year), month=month, day=int(day), hour=hour, minute=minute
        )
