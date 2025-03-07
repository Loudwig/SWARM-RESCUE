from typing import List

class DroneMessage:
    class Subject:
        PING = "PING"
        DRONE_POSITION = "DRONE_POSITION"
        GRID_COMMUNICATION = "GRID_COMMUNICATION"

    def __init__(self, subject: str, body=None, sender_id=None):
        if subject not in vars(DroneMessage.Subject).values():
            raise ValueError(f"Invalid subject: {subject}")

        self.subject = subject
        self.body = body
        self.sender_id = sender_id

class DroneMessageBatch:
    """Container for multiple DroneMessage objects"""
    def __init__(self, sender_id=None):
        self.messages: List[DroneMessage] = []
        self.sender_id = sender_id

    def add_message(self, message: DroneMessage):
        if not isinstance(message, DroneMessage):
            raise ValueError("Can only add DroneMessage objects")
        message.sender_id = self.sender_id
        self.messages.append(message)

    def get_messages_by_subject(self, subject: str):
        """Return all messages with a specific subject"""
        return [msg for msg in self.messages if msg.subject == subject]

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)