from pydantic import BaseModel

class classificationRequest(BaseModel):
    user_message :str
    session_id :str