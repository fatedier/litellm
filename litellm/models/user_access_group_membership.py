"""
User access group membership table model.

Canonical definition for ``litellm_useraccessgroupmembership``.
"""

from datetime import datetime
from typing import Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_UserAccessGroupMembershipTable(LiteLLMPydanticObjectBase):
    user_id: str
    access_group_id: str
    created_at: Optional[datetime] = None
