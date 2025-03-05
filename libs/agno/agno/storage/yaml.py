import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml

from agno.storage.base import Storage
from agno.storage.session.agent import AgentSession
from agno.storage.session.workflow import WorkflowSession
from agno.utils.log import logger


class YamlStorage(Storage):
    def __init__(self, dir_path: Union[str, Path], mode: Optional[Literal["agent", "workflow"]] = "agent"):
        super().__init__(mode)
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def serialize(self, data: dict) -> str:
        return yaml.dump(data, default_flow_style=False)

    def deserialize(self, data: str) -> dict:
        return yaml.safe_load(data)

    def create(self) -> None:
        """Create the storage if it doesn't exist."""
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True, exist_ok=True)

    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[Union[AgentSession, WorkflowSession]]:
        """Read a Session from storage."""
        try:
            with open(self.dir_path / f"{session_id}.yaml", "r", encoding="utf-8") as f:
                data = self.deserialize(f.read())
                if user_id and data["user_id"] != user_id:
                    return None
                if self.mode == "agent":
                    return AgentSession.from_dict(data)
                elif self.mode == "workflow":
                    return WorkflowSession.from_dict(data)
        except FileNotFoundError:
            return None

    def get_all_session_ids(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[str]:
        """Get all session IDs, optionally filtered by user_id and/or entity_id."""
        session_ids = []
        for file in self.dir_path.glob("*.yaml"):
            with open(file, "r", encoding="utf-8") as f:
                data = self.deserialize(f.read())
                if user_id or entity_id:
                    if user_id and entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id and data["user_id"] == user_id:
                            session_ids.append(data["session_id"])
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id and data["user_id"] == user_id:
                            session_ids.append(data["session_id"])
                    elif user_id and data["user_id"] == user_id:
                        session_ids.append(data["session_id"])
                    elif entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id:
                            session_ids.append(data["session_id"])
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id:
                            session_ids.append(data["session_id"])
                else:
                    # No filters applied, add all session_ids
                    session_ids.append(data["session_id"])
        return session_ids

    def get_all_sessions(
        self, user_id: Optional[str] = None, entity_id: Optional[str] = None
    ) -> List[Union[AgentSession, WorkflowSession]]:
        """Get all sessions, optionally filtered by user_id and/or entity_id."""
        sessions: List[Union[AgentSession, WorkflowSession]] = []
        for file in self.dir_path.glob("*.yaml"):
            with open(file, "r", encoding="utf-8") as f:
                data = self.deserialize(f.read())
                if user_id or entity_id:
                    if user_id and entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id and data["user_id"] == user_id:
                            sessions.append(AgentSession.from_dict(data))
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id and data["user_id"] == user_id:
                            sessions.append(WorkflowSession.from_dict(data))
                    elif user_id and data["user_id"] == user_id:
                        if self.mode == "agent":
                            sessions.append(AgentSession.from_dict(data))
                        elif self.mode == "workflow":
                            sessions.append(WorkflowSession.from_dict(data))
                    elif entity_id:
                        if self.mode == "agent" and data["agent_id"] == entity_id:
                            sessions.append(AgentSession.from_dict(data))
                        elif self.mode == "workflow" and data["workflow_id"] == entity_id:
                            sessions.append(WorkflowSession.from_dict(data))
                else:
                    # No filters applied, add all sessions
                    if self.mode == "agent":
                        sessions.append(AgentSession.from_dict(data))
                    elif self.mode == "workflow":
                        sessions.append(WorkflowSession.from_dict(data))
        return sessions

    def upsert(self, session: Union[AgentSession, WorkflowSession]) -> Optional[Union[AgentSession, WorkflowSession]]:
        """Insert or update an Session in storage."""
        try:
            data = asdict(session)
            data["updated_at"] = int(time.time())
            if "created_at" not in data:
                data["created_at"] = data["updated_at"]

            with open(self.dir_path / f"{session.session_id}.yaml", "w", encoding="utf-8") as f:
                f.write(self.serialize(data))
            return session
        except Exception as e:
            logger.error(f"Error upserting session: {e}")
            return None

    def delete_session(self, session_id: Optional[str] = None):
        """Delete a session from storage."""
        if session_id is None:
            return
        try:
            (self.dir_path / f"{session_id}.yaml").unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error deleting session: {e}")

    def drop(self) -> None:
        """Drop all sessions from storage."""
        for file in self.dir_path.glob("*.yaml"):
            file.unlink()

    def upgrade_schema(self) -> None:
        """Upgrade the schema of the storage."""
        pass
